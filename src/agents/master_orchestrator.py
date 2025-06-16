"""
Master Orchestrator Agent for Social Support AI Workflow

This agent serves as the central coordinator for the entire application processing
workflow, managing agent interactions, decision routing, and overall process flow
as required by the specification.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from src.agents.base_agent import BaseAgent
from src.agents.conversation_agent import ConversationAgent
from src.agents.data_extraction_agent import DataExtractionAgent
from src.agents.eligibility_agent import EligibilityAssessmentAgent
from src.utils.logging_config import get_logger

logger = get_logger("master_orchestrator")


class MasterOrchestratorAgent(BaseAgent):
    """
    Master Orchestrator Agent for Social Support Application Processing
    
    Central coordinator that manages the entire application workflow by orchestrating
    interactions between specialized agents (Conversation, DataExtraction, Eligibility).
    Implements decision routing, process flow control, and agent coordination.
    """
    
    def __init__(self):
        super().__init__("MasterOrchestratorAgent")
        
        # Initialize specialized agents
        self.conversation_agent = ConversationAgent()
        self.data_extraction_agent = DataExtractionAgent()
        self.eligibility_agent = EligibilityAssessmentAgent()
        
        # Workflow state tracking
        self.active_applications = {}
        self.agent_performance_metrics = {
            "conversation_agent": {"calls": 0, "success": 0, "avg_time": 0},
            "data_extraction_agent": {"calls": 0, "success": 0, "avg_time": 0},
            "eligibility_agent": {"calls": 0, "success": 0, "avg_time": 0}
        }
        
        # Decision routing rules
        self.routing_rules = {
            "conversation_flow": self._route_conversation_decisions,
            "document_processing": self._route_document_processing,
            "eligibility_assessment": self._route_eligibility_decisions,
            "error_handling": self._route_error_recovery
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main orchestration method that coordinates the entire application workflow
        
        Args:
            input_data: {
                "workflow_type": str,  # "conversation", "document_processing", "eligibility"
                "application_id": str,
                "user_input": str,
                "conversation_state": Dict,
                "documents": List[str],
                "extracted_data": Dict
            }
            
        Returns:
            Orchestrated workflow result with agent coordination details
        """
        
        workflow_type = input_data.get("workflow_type", "conversation")
        application_id = input_data.get("application_id", "unknown")
        
        try:
            logger.info(f"🎯 Master Orchestrator processing {workflow_type} for application {application_id}")
            
            # Track application state
            if application_id not in self.active_applications:
                self.active_applications[application_id] = {
                    "start_time": datetime.utcnow(),
                    "current_stage": "initiated",
                    "agents_involved": [],
                    "decisions_made": [],
                    "errors_encountered": []
                }
            
            # Route to appropriate workflow
            if workflow_type == "conversation":
                result = await self._orchestrate_conversation_workflow(input_data)
            elif workflow_type == "document_processing":
                result = await self._orchestrate_document_workflow(input_data)
            elif workflow_type == "eligibility_assessment":
                result = await self._orchestrate_eligibility_workflow(input_data)
            elif workflow_type == "full_application":
                result = await self._orchestrate_full_application_workflow(input_data)
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
            
            # Update application tracking
            self.active_applications[application_id]["current_stage"] = "completed"
            self.active_applications[application_id]["completion_time"] = datetime.utcnow()
            
            # Add orchestration metadata
            result["orchestration_metadata"] = {
                "orchestrator": self.agent_name,
                "workflow_type": workflow_type,
                "application_id": application_id,
                "agents_coordinated": self.active_applications[application_id]["agents_involved"],
                "total_processing_time": (
                    self.active_applications[application_id]["completion_time"] - 
                    self.active_applications[application_id]["start_time"]
                ).total_seconds(),
                "decisions_coordinated": len(self.active_applications[application_id]["decisions_made"])
            }
            
            logger.info(f"✅ Master Orchestrator completed {workflow_type} successfully")
            return result
            
        except Exception as e:
            error_msg = f"Master Orchestrator error in {workflow_type}: {str(e)}"
            logger.error(error_msg)
            
            # Track error
            if application_id in self.active_applications:
                self.active_applications[application_id]["errors_encountered"].append({
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                    "workflow_type": workflow_type
                })
            
            return {
                "agent_name": self.agent_name,
                "status": "error",
                "error": error_msg,
                "workflow_type": workflow_type,
                "application_id": application_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _orchestrate_conversation_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate conversation workflow through ConversationAgent"""
        
        application_id = input_data.get("application_id")
        self.active_applications[application_id]["agents_involved"].append("conversation_agent")
        
        start_time = datetime.utcnow()
        
        try:
            # Delegate to ConversationAgent
            conversation_result = await self.conversation_agent.process_message(
                input_data.get("user_input", ""),
                input_data.get("conversation_history", []),
                input_data.get("conversation_state", {})
            )
            
            # Track performance
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_agent_metrics("conversation_agent", True, processing_time)
            
            # Make routing decision
            next_action = self._route_conversation_decisions(conversation_result, input_data)
            
            # Track decision
            self.active_applications[application_id]["decisions_made"].append({
                "agent": "conversation_agent",
                "decision": next_action,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {
                "agent_name": self.agent_name,
                "status": "success",
                "conversation_result": conversation_result,
                "next_action": next_action,
                "processing_time_seconds": processing_time
            }
            
        except Exception as e:
            self._update_agent_metrics("conversation_agent", False, 0)
            raise e
    
    async def _orchestrate_document_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate document processing workflow through DataExtractionAgent"""
        
        application_id = input_data.get("application_id")
        self.active_applications[application_id]["agents_involved"].append("data_extraction_agent")
        
        start_time = datetime.utcnow()
        
        try:
            # Delegate to DataExtractionAgent
            extraction_result = await self.data_extraction_agent.process({
                "documents": input_data.get("documents", []),
                "extraction_mode": "conversational"
            })
            
            # Track performance
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_agent_metrics("data_extraction_agent", True, processing_time)
            
            # Make routing decision
            next_action = self._route_document_processing(extraction_result, input_data)
            
            # Track decision
            self.active_applications[application_id]["decisions_made"].append({
                "agent": "data_extraction_agent",
                "decision": next_action,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {
                "agent_name": self.agent_name,
                "status": "success",
                "extraction_result": extraction_result,
                "next_action": next_action,
                "processing_time_seconds": processing_time
            }
            
        except Exception as e:
            self._update_agent_metrics("data_extraction_agent", False, 0)
            raise e
    
    async def _orchestrate_eligibility_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate eligibility assessment workflow through EligibilityAgent"""
        
        application_id = input_data.get("application_id")
        self.active_applications[application_id]["agents_involved"].append("eligibility_agent")
        
        start_time = datetime.utcnow()
        
        try:
            # Delegate to EligibilityAssessmentAgent
            eligibility_result = await self.eligibility_agent.process({
                "application_data": input_data.get("application_data", {}),
                "extracted_documents": input_data.get("extracted_documents", {}),
                "application_id": application_id
            })
            
            # Track performance
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_agent_metrics("eligibility_agent", True, processing_time)
            
            # Make routing decision
            next_action = self._route_eligibility_decisions(eligibility_result, input_data)
            
            # Track decision
            self.active_applications[application_id]["decisions_made"].append({
                "agent": "eligibility_agent",
                "decision": next_action,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {
                "agent_name": self.agent_name,
                "status": "success",
                "eligibility_result": eligibility_result,
                "next_action": next_action,
                "processing_time_seconds": processing_time
            }
            
        except Exception as e:
            self._update_agent_metrics("eligibility_agent", False, 0)
            raise e
    
    async def _orchestrate_full_application_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate complete application workflow coordinating all agents"""
        
        application_id = input_data.get("application_id")
        logger.info(f"🔄 Starting full application workflow for {application_id}")
        
        workflow_results = {
            "conversation_phase": None,
            "document_phase": None,
            "eligibility_phase": None,
            "final_decision": None
        }
        
        try:
            # Phase 1: Conversation and data collection
            if input_data.get("user_input"):
                conversation_input = {
                    "workflow_type": "conversation",
                    "application_id": application_id,
                    "user_input": input_data.get("user_input"),
                    "conversation_history": input_data.get("conversation_history", []),
                    "conversation_state": input_data.get("conversation_state", {})
                }
                workflow_results["conversation_phase"] = await self._orchestrate_conversation_workflow(conversation_input)
            
            # Phase 2: Document processing (if documents available)
            if input_data.get("documents"):
                document_input = {
                    "workflow_type": "document_processing",
                    "application_id": application_id,
                    "documents": input_data.get("documents")
                }
                workflow_results["document_phase"] = await self._orchestrate_document_workflow(document_input)
            
            # Phase 3: Eligibility assessment (if sufficient data)
            if input_data.get("application_data"):
                eligibility_input = {
                    "workflow_type": "eligibility_assessment",
                    "application_id": application_id,
                    "application_data": input_data.get("application_data"),
                    "extracted_documents": input_data.get("extracted_documents", {})
                }
                workflow_results["eligibility_phase"] = await self._orchestrate_eligibility_workflow(eligibility_input)
            
            # Coordinate final decision
            workflow_results["final_decision"] = self._coordinate_final_decision(workflow_results)
            
            return {
                "agent_name": self.agent_name,
                "status": "success",
                "workflow_type": "full_application",
                "application_id": application_id,
                "workflow_results": workflow_results,
                "coordination_summary": self._generate_coordination_summary(application_id)
            }
            
        except Exception as e:
            logger.error(f"Full workflow orchestration failed: {str(e)}")
            raise e
    
    def _route_conversation_decisions(self, conversation_result: Dict, input_data: Dict) -> str:
        """Route decisions based on conversation agent results"""
        
        if conversation_result.get("application_complete"):
            return "proceed_to_eligibility"
        elif conversation_result.get("documents_needed"):
            return "request_documents"
        elif conversation_result.get("clarification_needed"):
            return "continue_conversation"
        else:
            return "continue_conversation"
    
    def _route_document_processing(self, extraction_result: Dict, input_data: Dict) -> str:
        """Route decisions based on document extraction results"""
        
        if extraction_result.get("status") == "success":
            return "proceed_to_validation"
        elif extraction_result.get("status") == "partial":
            return "request_additional_documents"
        else:
            return "retry_extraction"
    
    def _route_eligibility_decisions(self, eligibility_result: Dict, input_data: Dict) -> str:
        """Route decisions based on eligibility assessment results"""
        
        assessment = eligibility_result.get("assessment_result", {})
        
        if assessment.get("eligible"):
            return "approve_application"
        else:
            return "decline_application"
    
    def _route_error_recovery(self, error_info: Dict, input_data: Dict) -> str:
        """Route error recovery decisions"""
        
        error_type = error_info.get("error_type", "unknown")
        
        if error_type == "agent_timeout":
            return "retry_with_fallback"
        elif error_type == "data_validation":
            return "request_clarification"
        else:
            return "escalate_to_human"
    
    def _coordinate_final_decision(self, workflow_results: Dict) -> Dict[str, Any]:
        """Coordinate final decision based on all workflow phases"""
        
        # Extract key results
        conversation_complete = workflow_results.get("conversation_phase", {}).get("status") == "success"
        documents_processed = workflow_results.get("document_phase", {}).get("status") == "success"
        eligibility_assessed = workflow_results.get("eligibility_phase", {}).get("status") == "success"
        
        # Determine final decision
        if eligibility_assessed:
            eligibility_result = workflow_results["eligibility_phase"]["eligibility_result"]
            assessment = eligibility_result.get("assessment_result", {})
            
            return {
                "decision": "approved" if assessment.get("eligible") else "declined",
                "confidence": assessment.get("confidence", 0.5),
                "support_amount": assessment.get("support_amount", 0),
                "reasoning": assessment.get("reasoning", "Coordinated assessment completed"),
                "workflow_completeness": {
                    "conversation": conversation_complete,
                    "documents": documents_processed,
                    "eligibility": eligibility_assessed
                }
            }
        else:
            return {
                "decision": "incomplete",
                "reasoning": "Insufficient data for final decision",
                "workflow_completeness": {
                    "conversation": conversation_complete,
                    "documents": documents_processed,
                    "eligibility": eligibility_assessed
                }
            }
    
    def _update_agent_metrics(self, agent_name: str, success: bool, processing_time: float):
        """Update performance metrics for coordinated agents"""
        
        metrics = self.agent_performance_metrics[agent_name]
        metrics["calls"] += 1
        
        if success:
            metrics["success"] += 1
        
        # Update average processing time
        if metrics["calls"] == 1:
            metrics["avg_time"] = processing_time
        else:
            metrics["avg_time"] = (metrics["avg_time"] * (metrics["calls"] - 1) + processing_time) / metrics["calls"]
    
    def _generate_coordination_summary(self, application_id: str) -> Dict[str, Any]:
        """Generate summary of orchestration activities"""
        
        app_data = self.active_applications.get(application_id, {})
        
        return {
            "total_agents_involved": len(app_data.get("agents_involved", [])),
            "agents_list": app_data.get("agents_involved", []),
            "total_decisions_made": len(app_data.get("decisions_made", [])),
            "errors_encountered": len(app_data.get("errors_encountered", [])),
            "processing_duration": (
                app_data.get("completion_time", datetime.utcnow()) - 
                app_data.get("start_time", datetime.utcnow())
            ).total_seconds() if app_data.get("start_time") else 0,
            "agent_performance": self.agent_performance_metrics
        }
    
    def get_orchestration_status(self, application_id: str) -> Dict[str, Any]:
        """Get current orchestration status for an application"""
        
        if application_id not in self.active_applications:
            return {"status": "not_found", "message": "Application not being orchestrated"}
        
        app_data = self.active_applications[application_id]
        
        return {
            "status": "active" if "completion_time" not in app_data else "completed",
            "current_stage": app_data.get("current_stage", "unknown"),
            "agents_involved": app_data.get("agents_involved", []),
            "decisions_made": len(app_data.get("decisions_made", [])),
            "errors_count": len(app_data.get("errors_encountered", [])),
            "start_time": app_data.get("start_time", "").isoformat() if app_data.get("start_time") else None,
            "completion_time": app_data.get("completion_time", "").isoformat() if app_data.get("completion_time") else None
        } 