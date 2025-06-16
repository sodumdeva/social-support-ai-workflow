"""
Base Agent class for Social Support AI Workflow

Provides common functionality for all AI agents including:
- LLM integration with Ollama
- Logging and observability
- Error handling
- Performance tracking
- Singleton pattern to prevent multiple instances
"""
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import requests
from loguru import logger

from config import settings


class BaseAgent(ABC):
    """Base class for all AI agents in the social support workflow"""
    
    _instances = {}  # Class variable to store singleton instances
    
    def __new__(cls, agent_name: str = None, *args, **kwargs):
        """Implement singleton pattern per agent class"""
        if cls not in cls._instances:
            cls._instances[cls] = super(BaseAgent, cls).__new__(cls)
        return cls._instances[cls]
    
    def __init__(self, agent_name: str, model_name: Optional[str] = None):
        # Only initialize once per class
        if hasattr(self, '_initialized'):
            return
            
        self.agent_name = agent_name
        self.model_name = model_name or settings.ollama_model
        self.ollama_url = settings.ollama_base_url
        self.performance_metrics = []
        
        logger.info(f"Initializing {agent_name} agent with model {self.model_name}")
        self._initialized = True
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method that each agent must implement
        
        Args:
            input_data: Input data specific to the agent's function
            
        Returns:
            Dictionary containing agent's output and metadata
        """
        pass
    
    async def invoke_llm(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke the local LLM using Ollama
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            LLM response with metadata
        """
        start_time = time.time()
        
        try:
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=300  # 2 minute timeout
            )
            response.raise_for_status()
            
            result = response.json()
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Log performance metrics
            self._log_performance(processing_time, len(prompt))
            
            return {
                "response": result.get("response", ""),
                "model": result.get("model", self.model_name),
                "processing_time_ms": processing_time,
                "tokens_generated": result.get("eval_count", 0),
                "tokens_processed": result.get("prompt_eval_count", 0),
                "status": "success"
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed for {self.agent_name}: {e}")
            return {
                "response": "",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "status": "error"
            }
        except Exception as e:
            logger.error(f"Unexpected error in LLM invocation for {self.agent_name}: {e}")
            return {
                "response": "",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "status": "error"
            }
    
    async def invoke_vision_llm(self, prompt: str, image_path: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke vision-capable LLM for image analysis
        
        Args:
            prompt: Text prompt
            image_path: Path to image file
            system_prompt: Optional system prompt
            
        Returns:
            Vision LLM response with metadata
        """
        start_time = time.time()
        
        try:
            import base64
            
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            payload = {
                "model": settings.ollama_vision_model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=300  # 3 minute timeout for vision models
            )
            response.raise_for_status()
            
            result = response.json()
            processing_time = (time.time() - start_time) * 1000
            
            self._log_performance(processing_time, len(prompt), "vision")
            
            return {
                "response": result.get("response", ""),
                "model": result.get("model", settings.ollama_vision_model),
                "processing_time_ms": processing_time,
                "tokens_generated": result.get("eval_count", 0),
                "tokens_processed": result.get("prompt_eval_count", 0),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Vision LLM request failed for {self.agent_name}: {e}")
            return {
                "response": "",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "status": "error"
            }
    
    def _log_performance(self, processing_time_ms: float, prompt_length: int, model_type: str = "text"):
        """Log performance metrics for monitoring"""
        metric = {
            "agent_name": self.agent_name,
            "model_type": model_type,
            "processing_time_ms": processing_time_ms,
            "prompt_length": prompt_length,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.performance_metrics.append(metric)
        
        # Keep only last 100 metrics to prevent memory issues
        if len(self.performance_metrics) > 100:
            self.performance_metrics = self.performance_metrics[-100:]
        
        logger.info(f"{self.agent_name} processed request in {processing_time_ms:.2f}ms")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this agent"""
        if not self.performance_metrics:
            return {"message": "No performance data available"}
        
        processing_times = [m["processing_time_ms"] for m in self.performance_metrics]
        
        return {
            "agent_name": self.agent_name,
            "total_requests": len(self.performance_metrics),
            "average_processing_time_ms": sum(processing_times) / len(processing_times),
            "min_processing_time_ms": min(processing_times),
            "max_processing_time_ms": max(processing_times),
            "last_request": self.performance_metrics[-1]["timestamp"] if self.performance_metrics else None
        }
    
    def create_structured_prompt(self, task: str, context: Dict[str, Any], output_format: str) -> str:
        """
        Create a structured prompt for consistent LLM interactions
        
        Args:
            task: Description of the task
            context: Context data for the task
            output_format: Expected output format
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
TASK: {task}

CONTEXT:
{json.dumps(context, indent=2)}

INSTRUCTIONS:
- Analyze the provided context carefully
- Focus on accuracy and relevance
- Provide clear reasoning for your conclusions
- Output in the specified format

OUTPUT FORMAT:
{output_format}

RESPONSE:
"""
        return prompt.strip()
    
    def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from LLM response, handling common formatting issues
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        try:
            # Try direct JSON parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                # Look for JSON block in response
                start_markers = ['```json', '```JSON', '{']
                end_markers = ['```', '}']
                
                json_start = -1
                for marker in start_markers:
                    idx = response.find(marker)
                    if idx != -1:
                        json_start = idx + len(marker) if marker.startswith('```') else idx
                        break
                
                if json_start == -1:
                    return None
                
                # Find the end of JSON
                json_content = response[json_start:]
                
                # Try to find complete JSON object
                brace_count = 0
                json_end = -1
                
                for i, char in enumerate(json_content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end != -1:
                    json_str = json_content[:json_end]
                    return json.loads(json_str)
                
                return None
                
            except Exception as e:
                logger.warning(f"Failed to extract JSON from response: {e}")
                return None 