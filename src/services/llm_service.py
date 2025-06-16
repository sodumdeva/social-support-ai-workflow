"""
Local LLM Service for Social Support AI Workflow

Integrates with locally hosted LLM models (Ollama) for AI-driven conversation processing,
data extraction, and response generation as required by the solution specifications.
"""
import asyncio
import json
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


class LocalLLMService:
    """Service for interacting with locally hosted LLM models via Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama2"):
        self.base_url = base_url
        self.model_name = model_name
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "stream": False
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def initialize(self):
        """Initialize the LLM service and check model availability"""
        
        try:
            # Check if Ollama is running
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        models = await response.json()
                        available_models = [model["name"] for model in models.get("models", [])]
                        
                        if self.model_name not in available_models:
                            self.logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                            # Try to pull the model
                            await self._pull_model()
                        
                        self.logger.info(f"LLM Service initialized with model: {self.model_name}")
                        return True
                    else:
                        self.logger.error(f"Ollama not accessible at {self.base_url}")
                        return False
        
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM service: {str(e)}")
            return False
    
    async def _pull_model(self):
        """Pull the specified model if not available"""
        
        try:
            async with aiohttp.ClientSession() as session:
                pull_data = {"name": self.model_name}
                
                async with session.post(
                    f"{self.base_url}/api/pull",
                    json=pull_data
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Successfully pulled model: {self.model_name}")
                    else:
                        self.logger.error(f"Failed to pull model: {self.model_name}")
        
        except Exception as e:
            self.logger.error(f"Error pulling model: {str(e)}")
    
    async def generate_response(
        self, 
        user_prompt: str, 
        system_prompt: str = "", 
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate response using local LLM"""
        
        try:
            # Prepare the prompt
            full_prompt = self._build_prompt(user_prompt, system_prompt, context)
            
            # Prepare request data
            request_data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.model_config["temperature"],
                    "top_p": self.model_config["top_p"],
                    "num_predict": self.model_config["max_tokens"]
                }
            }
            
            # Make request to Ollama
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        return {
                            "status": "success",
                            "response": result.get("response", ""),
                            "model": self.model_name,
                            "tokens_used": result.get("eval_count", 0),
                            "generation_time": result.get("total_duration", 0) / 1e9  # Convert to seconds
                        }
                    else:
                        error_text = await response.text()
                        self.logger.error(f"LLM API error: {response.status} - {error_text}")
                        
                        return {
                            "status": "error",
                            "error": f"API error: {response.status}",
                            "response": ""
                        }
        
        except asyncio.TimeoutError:
            self.logger.error("LLM request timed out")
            return {
                "status": "error",
                "error": "Request timeout",
                "response": ""
            }
        
        except Exception as e:
            self.logger.error(f"LLM generation error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "response": ""
            }
    
    async def analyze_intent(self, user_message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Analyze user intent using LLM"""
        
        system_prompt = """You are an AI assistant specialized in analyzing user intent for government social support applications.

Analyze the user's message and classify their intent into one of these categories:
- provide_information: User is providing requested information
- correct_information: User wants to correct previously provided information  
- ask_question: User is asking a question about the process
- navigate: User wants to go back or change conversation flow
- request_help: User needs assistance or clarification

Also extract any structured data from the message (names, numbers, dates, etc.).

Respond in JSON format with:
{
  "intent": "category",
  "confidence": 0.0-1.0,
  "extracted_entities": {
    "field_name": "value"
  },
  "reasoning": "brief explanation",
  "next_action": "suggested next step"
}"""
        
        # Build context from conversation history
        context_str = self._build_conversation_context(conversation_history)
        
        user_prompt = f"""
Conversation Context: {context_str}

User Message: "{user_message}"

Analyze this message and provide your assessment in the specified JSON format."""
        
        result = await self.generate_response(user_prompt, system_prompt)
        
        if result["status"] == "success":
            try:
                # Try to parse JSON response
                intent_data = json.loads(result["response"])
                return {
                    "status": "success",
                    "intent_analysis": intent_data
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "status": "partial_success",
                    "intent_analysis": {
                        "intent": "provide_information",
                        "confidence": 0.5,
                        "reasoning": "JSON parsing failed, using fallback",
                        "raw_response": result["response"]
                    }
                }
        else:
            return result
    
    async def extract_structured_data(self, user_message: str, current_data: Dict) -> Dict[str, Any]:
        """Extract structured data from user message using LLM"""
        
        system_prompt = """You are a data extraction specialist for social support applications.

Extract structured information from the user's message. Look for:
- Personal information (names, IDs, contact details)
- Employment information (status, income, job title)
- Family information (size, dependents)
- Housing information (ownership, rental, living situation)
- Financial information (income, assets, debts)

Respond in JSON format with:
{
  "extracted_data": {
    "field_name": "value"
  },
  "confidence_scores": {
    "field_name": 0.0-1.0
  },
  "validation_notes": {
    "field_name": "any concerns or notes"
  }
}

Only extract information that is clearly stated. Don't make assumptions."""
        
        user_prompt = f"""
Current collected data: {json.dumps(current_data, indent=2)}

User message: "{user_message}"

Extract any new or updated information from this message."""
        
        result = await self.generate_response(user_prompt, system_prompt)
        
        if result["status"] == "success":
            try:
                extraction_data = json.loads(result["response"])
                return {
                    "status": "success",
                    "extraction_result": extraction_data
                }
            except json.JSONDecodeError:
                return {
                    "status": "partial_success",
                    "extraction_result": {
                        "extracted_data": {},
                        "raw_response": result["response"]
                    }
                }
        else:
            return result
    
    async def generate_conversational_response(
        self, 
        context: Dict[str, Any], 
        user_message: str
    ) -> Dict[str, Any]:
        """Generate contextual conversational response using LLM"""
        
        system_prompt = """You are a helpful AI assistant for a government social support application system.

Your role is to:
1. Guide users through the application process naturally and conversationally
2. Collect required information in a friendly, empathetic manner
3. Handle corrections and clarifications professionally
4. Provide clear feedback and next steps
5. Maintain a supportive, government-appropriate tone

Be empathetic, clear, and professional. Always confirm important information.
If there are validation issues, ask for clarification politely.

Respond in JSON format with:
{
  "message": "your conversational response",
  "next_step": "what should happen next",
  "requires_input": true/false,
  "confidence": 0.0-1.0
}"""
        
        user_prompt = f"""
Context Information:
- User Intent: {context.get('intent', 'unknown')}
- Extracted Data: {json.dumps(context.get('extracted_data', {}), indent=2)}
- Validation Issues: {context.get('validation_issues', [])}
- Missing Fields: {context.get('missing_fields', [])}
- Current Data: {json.dumps(context.get('current_data', {}), indent=2)}

User Message: "{user_message}"

Generate an appropriate conversational response that:
1. Acknowledges what the user provided
2. Addresses any validation issues
3. Asks for the next required information or confirms completion
4. Maintains a helpful, conversational tone"""
        
        result = await self.generate_response(user_prompt, system_prompt)
        
        if result["status"] == "success":
            try:
                response_data = json.loads(result["response"])
                return {
                    "status": "success",
                    "response_data": response_data
                }
            except json.JSONDecodeError:
                return {
                    "status": "partial_success",
                    "response_data": {
                        "message": result["response"],
                        "next_step": "continue",
                        "requires_input": True,
                        "confidence": 0.7
                    }
                }
        else:
            return result
    
    async def validate_data_with_reasoning(self, field: str, value: Any, context: Dict) -> Dict[str, Any]:
        """Use LLM to validate data with reasoning"""
        
        system_prompt = f"""You are a data validation specialist for government social support applications.

Validate the provided {field} value considering:
1. Format correctness
2. Reasonableness for the context
3. Consistency with other provided information
4. Government application standards

Respond in JSON format with:
{{
  "is_valid": true/false,
  "confidence": 0.0-1.0,
  "issues": ["list of any issues found"],
  "suggestions": ["suggestions for correction if needed"],
  "reasoning": "explanation of your assessment"
}}"""
        
        user_prompt = f"""
Field: {field}
Value: {value}
Context: {json.dumps(context, indent=2)}

Validate this field value and provide your assessment."""
        
        result = await self.generate_response(user_prompt, system_prompt)
        
        if result["status"] == "success":
            try:
                validation_data = json.loads(result["response"])
                return {
                    "status": "success",
                    "validation_result": validation_data
                }
            except json.JSONDecodeError:
                return {
                    "status": "partial_success",
                    "validation_result": {
                        "is_valid": True,
                        "confidence": 0.5,
                        "reasoning": "JSON parsing failed",
                        "raw_response": result["response"]
                    }
                }
        else:
            return result
    
    def _build_prompt(self, user_prompt: str, system_prompt: str = "", context: Optional[Dict] = None) -> str:
        """Build complete prompt for LLM"""
        
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")
        
        if context:
            prompt_parts.append(f"Context: {json.dumps(context, indent=2)}")
        
        prompt_parts.append(f"User: {user_prompt}")
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def _build_conversation_context(self, conversation_history: List[Dict]) -> str:
        """Build context string from conversation history"""
        
        if not conversation_history:
            return "No previous conversation"
        
        recent_messages = conversation_history[-5:]  # Last 5 messages
        context_parts = []
        
        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # Truncate long messages
            timestamp = msg.get("timestamp", "")
            context_parts.append(f"{role}: {content}")
        
        return " | ".join(context_parts)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the LLM service is healthy and responsive"""
        
        try:
            test_prompt = "Hello, please respond with 'OK' to confirm you are working."
            result = await self.generate_response(test_prompt)
            
            if result["status"] == "success":
                return {
                    "status": "healthy",
                    "model": self.model_name,
                    "response_time": result.get("generation_time", 0),
                    "test_response": result["response"][:50]  # First 50 chars
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": result.get("error", "Unknown error")
                }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global LLM service instance
llm_service = LocalLLMService()


async def get_llm_service() -> LocalLLMService:
    """Get the global LLM service instance"""
    return llm_service


# Add invoke_llm method to BaseAgent
async def invoke_llm(self, user_prompt: str, system_prompt: str = "", context: Optional[Dict] = None) -> Dict[str, Any]:
    """Invoke local LLM for AI processing"""
    
    try:
        service = await get_llm_service()
        return await service.generate_response(user_prompt, system_prompt, context)
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "response": ""
        }


# Monkey patch the BaseAgent class to add LLM capability
from ..agents.base_agent import BaseAgent
BaseAgent.invoke_llm = invoke_llm 