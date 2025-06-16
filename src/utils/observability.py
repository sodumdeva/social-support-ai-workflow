"""
Agent Observability Module for Social Support AI Workflow

Implements end-to-end AI observability using LangSmith and Langfuse as required
by the specification. Provides comprehensive monitoring, tracing, and analytics
for agent performance and workflow execution.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from functools import wraps
import asyncio

from src.utils.logging_config import get_logger

logger = get_logger("observability")

# Try to import observability tools (graceful fallback if not available)
try:
    from langsmith import Client as LangSmithClient
    from langsmith.run_helpers import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    logger.warning("LangSmith not available - observability will use fallback logging")
    LANGSMITH_AVAILABLE = False
    def traceable(func):
        return func

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    logger.warning("Langfuse not available - observability will use fallback logging")
    LANGFUSE_AVAILABLE = False
    def observe(func):
        return func


class ObservabilityManager:
    """
    Comprehensive observability manager for AI agents and workflows
    
    Provides unified interface for LangSmith and Langfuse observability tools,
    with fallback to structured logging when external tools are unavailable.
    """
    
    def __init__(self):
        self.langsmith_client = None
        self.langfuse_client = None
        self.fallback_mode = False
        
        # Initialize LangSmith if available
        if LANGSMITH_AVAILABLE:
            try:
                api_key = os.getenv("LANGSMITH_API_KEY")
                if api_key:
                    self.langsmith_client = LangSmithClient(api_key=api_key)
                    logger.info("✅ LangSmith observability initialized")
                else:
                    logger.warning("LangSmith API key not found - using fallback mode")
                    self.fallback_mode = True
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith: {str(e)}")
                self.fallback_mode = True
        
        # Initialize Langfuse if available
        if LANGFUSE_AVAILABLE:
            try:
                public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
                secret_key = os.getenv("LANGFUSE_SECRET_KEY")
                host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                
                if public_key and secret_key:
                    self.langfuse_client = Langfuse(
                        public_key=public_key,
                        secret_key=secret_key,
                        host=host
                    )
                    logger.info("✅ Langfuse observability initialized")
                else:
                    logger.warning("Langfuse credentials not found - using fallback mode")
                    self.fallback_mode = True
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {str(e)}")
                self.fallback_mode = True
        
        # Metrics storage for fallback mode
        self.metrics_store = {
            "agent_calls": [],
            "workflow_executions": [],
            "performance_metrics": {},
            "error_tracking": []
        }
    
    def trace_agent_execution(self, agent_name: str, operation: str):
        """Decorator for tracing agent execution"""
        
        def decorator(func):
            if not self.fallback_mode and LANGSMITH_AVAILABLE:
                @traceable(name=f"{agent_name}_{operation}")
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    return await self._execute_with_tracing(func, agent_name, operation, *args, **kwargs)
                return wrapper
            elif not self.fallback_mode and LANGFUSE_AVAILABLE:
                @observe(name=f"{agent_name}_{operation}")
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    return await self._execute_with_tracing(func, agent_name, operation, *args, **kwargs)
                return wrapper
            else:
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    return await self._execute_with_fallback_logging(func, agent_name, operation, *args, **kwargs)
                return wrapper
        
        return decorator
    
    async def _execute_with_tracing(self, func, agent_name: str, operation: str, *args, **kwargs):
        """Execute function with full observability tracing"""
        
        start_time = datetime.utcnow()
        execution_id = f"{agent_name}_{operation}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Log execution start
            logger.info(f"🔍 Starting {agent_name}.{operation} - ID: {execution_id}")
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Calculate metrics
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            # Log success
            self._log_execution_success(agent_name, operation, execution_id, execution_time, result)
            
            return result
            
        except Exception as e:
            # Calculate metrics for failed execution
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            # Log error
            self._log_execution_error(agent_name, operation, execution_id, execution_time, str(e))
            
            raise e
    
    async def _execute_with_fallback_logging(self, func, agent_name: str, operation: str, *args, **kwargs):
        """Execute function with fallback logging when observability tools unavailable"""
        
        start_time = datetime.utcnow()
        execution_id = f"{agent_name}_{operation}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"📊 [FALLBACK] Starting {agent_name}.{operation} - ID: {execution_id}")
            
            result = await func(*args, **kwargs)
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            # Store metrics in fallback store
            self.metrics_store["agent_calls"].append({
                "agent_name": agent_name,
                "operation": operation,
                "execution_id": execution_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "execution_time": execution_time,
                "status": "success",
                "result_summary": self._summarize_result(result)
            })
            
            logger.info(f"✅ [FALLBACK] {agent_name}.{operation} completed in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            # Store error in fallback store
            self.metrics_store["error_tracking"].append({
                "agent_name": agent_name,
                "operation": operation,
                "execution_id": execution_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "execution_time": execution_time,
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            logger.error(f"❌ [FALLBACK] {agent_name}.{operation} failed after {execution_time:.2f}s: {str(e)}")
            
            raise e
    
    def _log_execution_success(self, agent_name: str, operation: str, execution_id: str, 
                             execution_time: float, result: Any):
        """Log successful execution with observability tools"""
        
        try:
            if self.langsmith_client:
                # Log to LangSmith
                self.langsmith_client.create_run(
                    name=f"{agent_name}_{operation}",
                    run_type="chain",
                    inputs={"agent": agent_name, "operation": operation},
                    outputs={"result": self._summarize_result(result)},
                    execution_order=1,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow()
                )
            
            if self.langfuse_client:
                # Log to Langfuse
                trace = self.langfuse_client.trace(
                    name=f"{agent_name}_{operation}",
                    metadata={
                        "agent_name": agent_name,
                        "operation": operation,
                        "execution_time": execution_time
                    }
                )
                
                trace.span(
                    name=operation,
                    metadata={"execution_id": execution_id},
                    level="DEFAULT"
                )
            
            logger.info(f"✅ {agent_name}.{operation} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to log execution success: {str(e)}")
    
    def _log_execution_error(self, agent_name: str, operation: str, execution_id: str, 
                           execution_time: float, error: str):
        """Log failed execution with observability tools"""
        
        try:
            if self.langsmith_client:
                # Log error to LangSmith
                self.langsmith_client.create_run(
                    name=f"{agent_name}_{operation}",
                    run_type="chain",
                    inputs={"agent": agent_name, "operation": operation},
                    error=error,
                    execution_order=1,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow()
                )
            
            if self.langfuse_client:
                # Log error to Langfuse
                trace = self.langfuse_client.trace(
                    name=f"{agent_name}_{operation}",
                    metadata={
                        "agent_name": agent_name,
                        "operation": operation,
                        "execution_time": execution_time,
                        "error": error
                    }
                )
                
                trace.span(
                    name=operation,
                    metadata={"execution_id": execution_id, "error": error},
                    level="ERROR"
                )
            
            logger.error(f"❌ {agent_name}.{operation} failed after {execution_time:.2f}s: {error}")
            
        except Exception as e:
            logger.error(f"Failed to log execution error: {str(e)}")
    
    def _summarize_result(self, result: Any) -> Dict[str, Any]:
        """Create a summary of execution result for logging"""
        
        if isinstance(result, dict):
            return {
                "type": "dict",
                "keys": list(result.keys()),
                "status": result.get("status", "unknown"),
                "agent_name": result.get("agent_name", "unknown")
            }
        elif isinstance(result, list):
            return {
                "type": "list",
                "length": len(result),
                "first_item_type": type(result[0]).__name__ if result else "empty"
            }
        else:
            return {
                "type": type(result).__name__,
                "value": str(result)[:100] if len(str(result)) > 100 else str(result)
            }
    
    def track_workflow_execution(self, workflow_name: str, application_id: str, 
                               workflow_data: Dict[str, Any]):
        """Track complete workflow execution"""
        
        try:
            workflow_record = {
                "workflow_name": workflow_name,
                "application_id": application_id,
                "timestamp": datetime.utcnow().isoformat(),
                "workflow_data": workflow_data,
                "agents_involved": workflow_data.get("agents_involved", []),
                "total_processing_time": workflow_data.get("total_processing_time", 0),
                "status": workflow_data.get("status", "unknown")
            }
            
            if not self.fallback_mode:
                if self.langfuse_client:
                    # Create workflow trace in Langfuse
                    trace = self.langfuse_client.trace(
                        name=f"workflow_{workflow_name}",
                        metadata=workflow_record
                    )
                    
                    # Add spans for each agent involved
                    for agent in workflow_data.get("agents_involved", []):
                        trace.span(
                            name=f"agent_{agent}",
                            metadata={"application_id": application_id}
                        )
            
            # Store in fallback regardless
            self.metrics_store["workflow_executions"].append(workflow_record)
            
            logger.info(f"📊 Tracked workflow execution: {workflow_name} for {application_id}")
            
        except Exception as e:
            logger.error(f"Failed to track workflow execution: {str(e)}")
    
    def get_agent_performance_metrics(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for agents"""
        
        if self.fallback_mode:
            # Return metrics from fallback store
            agent_calls = self.metrics_store["agent_calls"]
            
            if agent_name:
                agent_calls = [call for call in agent_calls if call["agent_name"] == agent_name]
            
            if not agent_calls:
                return {"message": f"No metrics found for agent: {agent_name}" if agent_name else "No metrics available"}
            
            total_calls = len(agent_calls)
            successful_calls = len([call for call in agent_calls if call["status"] == "success"])
            avg_execution_time = sum(call["execution_time"] for call in agent_calls) / total_calls
            
            return {
                "agent_name": agent_name or "all_agents",
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
                "average_execution_time": avg_execution_time,
                "recent_calls": agent_calls[-5:]  # Last 5 calls
            }
        
        else:
            # TODO: Implement metrics retrieval from LangSmith/Langfuse
            return {"message": "Metrics retrieval from observability tools not yet implemented"}
    
    def get_workflow_analytics(self) -> Dict[str, Any]:
        """Get analytics for workflow executions"""
        
        workflow_executions = self.metrics_store["workflow_executions"]
        
        if not workflow_executions:
            return {"message": "No workflow executions tracked"}
        
        total_workflows = len(workflow_executions)
        successful_workflows = len([w for w in workflow_executions if w["status"] == "success"])
        
        # Calculate average processing time
        processing_times = [w["total_processing_time"] for w in workflow_executions if w["total_processing_time"] > 0]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Agent usage statistics
        all_agents = []
        for workflow in workflow_executions:
            all_agents.extend(workflow.get("agents_involved", []))
        
        agent_usage = {}
        for agent in all_agents:
            agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        return {
            "total_workflows": total_workflows,
            "successful_workflows": successful_workflows,
            "success_rate": successful_workflows / total_workflows if total_workflows > 0 else 0,
            "average_processing_time": avg_processing_time,
            "agent_usage_statistics": agent_usage,
            "recent_workflows": workflow_executions[-5:]  # Last 5 workflows
        }
    
    def export_metrics(self, file_path: str):
        """Export all metrics to JSON file"""
        
        try:
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "observability_mode": "fallback" if self.fallback_mode else "full",
                "metrics": self.metrics_store,
                "summary": {
                    "total_agent_calls": len(self.metrics_store["agent_calls"]),
                    "total_workflows": len(self.metrics_store["workflow_executions"]),
                    "total_errors": len(self.metrics_store["error_tracking"])
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"📊 Metrics exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")


# Global observability manager instance
observability_manager = ObservabilityManager()

# Convenience decorators
def trace_agent(agent_name: str, operation: str):
    """Convenience decorator for agent tracing"""
    return observability_manager.trace_agent_execution(agent_name, operation)

def track_workflow(workflow_name: str, application_id: str, workflow_data: Dict[str, Any]):
    """Convenience function for workflow tracking"""
    return observability_manager.track_workflow_execution(workflow_name, application_id, workflow_data) 