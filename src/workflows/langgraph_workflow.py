"""
LangGraph Workflow for Social Support AI System

Implements state-based conversation workflow using LangGraph for managing
the conversational application process with proper state transitions.
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional, Any, Annotated
import asyncio
from datetime import datetime
import json
import operator

# Import agents with correct paths
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.agents.conversation_agent import ConversationAgent, ConversationStep
from src.agents.data_extraction_agent import DataExtractionAgent
from src.agents.eligibility_agent import EligibilityAssessmentAgent

# Import logging configuration
from src.utils.logging_config import get_logger
logger = get_logger("langgraph_workflow")


class ConversationState(TypedDict):
    """State structure for the conversation workflow"""
    messages: Annotated[List[Dict], operator.add]
    collected_data: Dict
    current_step: str
    eligibility_result: Optional[Dict]
    final_decision: Optional[Dict]
    uploaded_documents: List[str]
    workflow_history: Annotated[List[Dict], operator.add]
    application_id: Optional[str]
    processing_status: str
    error_messages: Annotated[List[str], operator.add]
    user_input: Optional[str]
    last_agent_response: Optional[str]


def create_conversation_workflow():
    """Create LangGraph workflow for conversational application processing"""
    
    workflow = StateGraph(ConversationState)
    
    # Add nodes for each step of the conversation
    workflow.add_node("initialize_conversation", initialize_conversation)
    workflow.add_node("handle_user_message", handle_user_message)
    workflow.add_node("process_documents", process_documents)
    workflow.add_node("validate_information", validate_information)
    workflow.add_node("assess_eligibility", assess_eligibility)
    workflow.add_node("generate_recommendations", generate_recommendations)
    workflow.add_node("finalize_application", finalize_application)
    workflow.add_node("handle_completion_chat", handle_completion_chat)
    
    # Define conditional routing
    workflow.add_conditional_edges(
        "initialize_conversation",
        should_continue_conversation,
        {
            "continue": "handle_user_message",
            "waiting": END
        }
    )
    
    workflow.add_conditional_edges(
        "handle_user_message", 
        determine_next_action,
        {
            "process_documents": "process_documents",
            "validate_info": "validate_information", 
            "assess_eligibility": "assess_eligibility",
            "continue_conversation": "handle_user_message",
            "completion_chat": "handle_completion_chat",
            "finalize": "finalize_application",
            "waiting": END
        }
    )
    
    workflow.add_conditional_edges(
        "process_documents",
        after_document_processing,
        {
            "continue_conversation": "handle_user_message",
            "validate_info": "validate_information"
        }
    )
    
    workflow.add_conditional_edges(
        "validate_information",
        after_validation,
        {
            "assess_eligibility": "assess_eligibility",
            "continue_conversation": "handle_user_message"
        }
    )
    
    workflow.add_conditional_edges(
        "assess_eligibility",
        after_eligibility_assessment,
        {
            "generate_recommendations": "generate_recommendations",
            "finalize": "finalize_application"
        }
    )
    
    workflow.add_edge("generate_recommendations", "finalize_application")
    workflow.add_edge("finalize_application", "handle_completion_chat")
    
    workflow.add_conditional_edges(
        "handle_completion_chat",
        after_completion_chat,
        {
            "restart": "initialize_conversation",
            "waiting": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("initialize_conversation")
    
    # Compile workflow (recursion_limit is set during invoke, not compile)
    return workflow.compile()


async def initialize_conversation(state: ConversationState) -> ConversationState:
    """Initialize the conversation workflow"""
    
    logger.info("Initializing conversation workflow")
    
    # Generate application ID if not exists
    if not state.get("application_id"):
        state["application_id"] = f"APP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Initialize conversation if empty or restarting
    if not state.get("messages") or state.get("processing_status") == "restarting":
        state["messages"] = [
            {
                "role": "assistant",
                "content": "Hello! I'm your Social Support AI Assistant. I'll help you apply for financial support through an easy conversation. Let's start - what's your full name?",
                "timestamp": datetime.now().isoformat()
            }
        ]
    
    # Initialize other state components
    state["collected_data"] = state.get("collected_data", {})
    state["current_step"] = state.get("current_step", ConversationStep.NAME_COLLECTION)
    state["uploaded_documents"] = state.get("uploaded_documents", [])
    state["workflow_history"] = state.get("workflow_history", [])
    state["error_messages"] = state.get("error_messages", [])
    
    # Set processing status based on whether we have user input
    if state.get("user_input"):
        state["processing_status"] = "in_progress"
    else:
        state["processing_status"] = "waiting_for_input"
    
    # Log workflow initialization
    state["workflow_history"].append({
        "step": "initialize_conversation",
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
        "application_id": state["application_id"]
    })
    
    logger.info(f"Conversation initialized with application ID: {state['application_id']}")
    return state


async def handle_user_message(state: ConversationState) -> ConversationState:
    """Handle user message through conversation agent"""
    
    try:
        user_input = state.get("user_input")
        if not user_input:
            logger.warning("No user input provided, setting status to waiting")
            state["processing_status"] = "waiting_for_input"
            return state
        
        logger.info(f"Processing user message: '{user_input}' at step: {state['current_step']}")
        
        # Initialize conversation agent
        conversation_agent = ConversationAgent()
        
        # Process the message
        conversation_state = {
            "current_step": state["current_step"],
            "collected_data": state["collected_data"],
            "uploaded_documents": state["uploaded_documents"],
            "eligibility_result": state.get("eligibility_result")
        }
        
        response = await conversation_agent.process_message(
            user_input,
            state["messages"],
            conversation_state
        )
        
        # Update state with response
        if "message" in response:
            state["messages"].append({
                "role": "assistant",
                "content": response["message"],
                "timestamp": datetime.now().isoformat()
            })
            state["last_agent_response"] = response["message"]
        
        # Update conversation state
        if "state_update" in response:
            for key, value in response["state_update"].items():
                if key in state:
                    state[key] = value
        
        # CRITICAL: Handle restart scenario when step changes to NAME_COLLECTION
        if state["current_step"] == ConversationStep.NAME_COLLECTION and "state_update" in response:
            # This is a restart - reset all relevant state
            if response["state_update"].get("collected_data") == {}:
                logger.info("Detected restart request - resetting conversation state")
                state["collected_data"] = {}
                state["eligibility_result"] = None
                state["final_decision"] = None
                state["uploaded_documents"] = []
                state["processed_documents"] = []
                state["processing_status"] = "restarting"
                
                # Log the restart
                state["workflow_history"].append({
                    "step": "restart_conversation",
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "message": "Conversation restarted"
                })
                
                # Clear user_input and return immediately to prevent further processing
                state["user_input"] = None
                return state
        
        # CRITICAL: Determine processing status based on response
        if response.get("application_complete"):
            state["final_decision"] = response.get("final_decision")
            state["processing_status"] = "completed"
        elif state["current_step"] == ConversationStep.COMPLETION:
            state["processing_status"] = "completion_chat"
        elif state["current_step"] == ConversationStep.DOCUMENT_COLLECTION:
            # Check if we need more information or documents
            if has_minimum_required_data(state["collected_data"]):
                state["processing_status"] = "ready_for_validation"
            else:
                state["processing_status"] = "waiting_for_input"
        else:
            # Default to waiting for more input to prevent loops
            state["processing_status"] = "waiting_for_input"
        
        # Clear user_input after processing to prevent reprocessing
        state["user_input"] = None
        
        # Log the interaction
        state["workflow_history"].append({
            "step": "handle_user_message",
            "user_message": user_input,
            "assistant_response": response.get("message", ""),
            "current_step": state["current_step"],
            "processing_status": state["processing_status"],
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
        logger.info(f"User message processed successfully. New step: {state['current_step']}, Status: {state['processing_status']}")
        
    except Exception as e:
        error_msg = f"Error handling user message: {str(e)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        state["messages"].append({
            "role": "assistant", 
            "content": "I apologize, I encountered an error processing your message. Could you please try again?",
            "timestamp": datetime.now().isoformat()
        })
        state["processing_status"] = "error"
        state["user_input"] = None  # Clear input on error too
    
    return state


async def process_documents(state: ConversationState) -> ConversationState:
    """Process uploaded documents"""
    
    try:
        logger.info("Processing uploaded documents")
        
        # Get unprocessed documents
        processed_docs = state.get("processed_documents", [])
        uploaded_docs = state.get("uploaded_documents", [])
        unprocessed_docs = [doc for doc in uploaded_docs if doc not in processed_docs]
        
        if not unprocessed_docs:
            state["processing_status"] = "no_documents_to_process"
            return state
        
        # Initialize data extraction agent
        data_extraction_agent = DataExtractionAgent()
        all_extraction_results = {}
        
        for doc_path in unprocessed_docs:
            try:
                # Determine document type from filename
                doc_type = determine_document_type(doc_path)
                
                # Process document
                extraction_result = await data_extraction_agent.process({
                    "documents": [{"file_path": doc_path, "document_type": doc_type}],
                    "extraction_mode": "conversational"
                })
                
                if extraction_result.get("status") == "success":
                    extracted_data = extraction_result.get("extraction_results", {}).get(doc_type, {})
                    if extracted_data.get("status") == "success":
                        structured_data = extracted_data.get("structured_data", {})
                        all_extraction_results[doc_type] = structured_data
                        
                        # Update collected data with extracted information
                        state["collected_data"].update(structured_data)
                
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {str(e)}")
        
        # Update processed documents list
        if "processed_documents" not in state:
            state["processed_documents"] = []
        state["processed_documents"].extend(unprocessed_docs)
        
        # Generate response about document processing
        if all_extraction_results:
            doc_types = list(all_extraction_results.keys())
            response_msg = f"I've successfully processed your {', '.join(doc_types).replace('_', ' ')}. The extracted information has been added to your application."
            
            state["messages"].append({
                "role": "assistant",
                "content": response_msg,
                "timestamp": datetime.now().isoformat()
            })
        
        state["processing_status"] = "documents_processed"
        
        # Log document processing
        state["workflow_history"].append({
            "step": "process_documents",
            "documents_processed": len(all_extraction_results),
            "extraction_results": all_extraction_results,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
        logger.info(f"Document processing completed. Processed {len(all_extraction_results)} documents")
        
    except Exception as e:
        error_msg = f"Error processing documents: {str(e)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        state["processing_status"] = "document_error"
    
    return state


async def validate_information(state: ConversationState) -> ConversationState:
    """Validate collected information for consistency"""
    
    try:
        logger.info("Validating collected information")
        
        collected_data = state["collected_data"]
        validation_results = {}
        
        # Basic validation checks
        required_fields = ["name", "employment_status", "monthly_income", "family_size"]
        missing_fields = [field for field in required_fields if field not in collected_data]
        
        if missing_fields:
            validation_results["missing_fields"] = missing_fields
            validation_results["status"] = "incomplete"
            
            state["messages"].append({
                "role": "assistant",
                "content": f"I still need the following information: {', '.join(missing_fields)}. Could you provide this?",
                "timestamp": datetime.now().isoformat()
            })
            
        else:
            # Perform consistency checks
            validation_results = perform_data_consistency_checks(collected_data)
            validation_results["status"] = "complete"
            
            if validation_results.get("inconsistencies"):
                inconsistency_msg = "I found some inconsistencies in the information: " + "; ".join(validation_results["inconsistencies"])
                state["messages"].append({
                    "role": "assistant",
                    "content": inconsistency_msg + " Could you help clarify these points?",
                    "timestamp": datetime.now().isoformat()
                })
        
        state["validation_results"] = validation_results
        state["processing_status"] = "validated"
        
        # Log validation
        state["workflow_history"].append({
            "step": "validate_information",
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
        logger.info(f"Information validation completed. Status: {validation_results['status']}")
        
    except Exception as e:
        error_msg = f"Error during validation: {str(e)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        state["processing_status"] = "validation_error"
    
    return state


async def assess_eligibility(state: ConversationState) -> ConversationState:
    """Assess eligibility for social support"""
    
    try:
        logger.info("Starting eligibility assessment")
        
        # Initialize eligibility agent
        eligibility_agent = EligibilityAssessmentAgent()
        
        # Run eligibility assessment
        eligibility_result = await eligibility_agent.process({
            "application_data": state["collected_data"],
            "assessment_mode": "conversational"
        })
        
        if eligibility_result.get("status") == "success":
            # Extract the decision
            decision = eligibility_result.get("eligibility_result") or eligibility_result.get("assessment_result")
            if not decision:
                decision = {k: v for k, v in eligibility_result.items() if k not in {"status", "agent_name", "application_id", "assessed_at", "assessment_method", "reasoning"}}
            
            state["eligibility_result"] = decision
            state["processing_status"] = "eligibility_assessed"
            
            logger.info(f"Eligibility assessment completed. Eligible: {decision.get('eligible', False)}")
            
        else:
            # Generate fallback decision
            logger.warning("Eligibility assessment failed, using fallback decision")
            state["eligibility_result"] = generate_fallback_eligibility_decision(state["collected_data"])
            state["processing_status"] = "eligibility_fallback"
        
        # Log eligibility assessment
        state["workflow_history"].append({
            "step": "assess_eligibility",
            "eligibility_result": state["eligibility_result"],
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
    except Exception as e:
        error_msg = f"Error during eligibility assessment: {str(e)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        state["processing_status"] = "eligibility_error"
        
        # Generate fallback decision
        state["eligibility_result"] = generate_fallback_eligibility_decision(state["collected_data"])
    
    return state


async def generate_recommendations(state: ConversationState) -> ConversationState:
    """Generate economic enablement recommendations"""
    
    try:
        logger.info("Generating economic enablement recommendations")
        
        # Initialize conversation agent for LLM-powered recommendations
        conversation_agent = ConversationAgent()
        
        # Generate recommendations using the new LLM method
        recommendations_result = await conversation_agent._generate_llm_economic_recommendations(
            state["collected_data"], 
            state["eligibility_result"]
        )
        
        if recommendations_result.get("status") == "success":
            # Add recommendations to eligibility result
            if "economic_enablement" not in state["eligibility_result"]:
                state["eligibility_result"]["economic_enablement"] = {}
            
            state["eligibility_result"]["economic_enablement"]["llm_generated"] = True
            state["eligibility_result"]["economic_enablement"]["recommendations_text"] = recommendations_result["response"]
            
            logger.info("LLM-powered recommendations generated successfully")
        else:
            logger.warning("LLM recommendations failed, using fallback")
            # Fallback to basic recommendations
            state["eligibility_result"]["economic_enablement"] = {
                "summary": "Basic economic enablement recommendations based on your profile.",
                "recommendations": [
                    "Explore skills development programs",
                    "Consider career advancement opportunities", 
                    "Look into financial literacy resources"
                ]
            }
        
        state["processing_status"] = "recommendations_generated"
        
        # Log recommendation generation
        state["workflow_history"].append({
            "step": "generate_recommendations",
            "recommendations_status": recommendations_result.get("status", "fallback"),
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
    except Exception as e:
        error_msg = f"Error generating recommendations: {str(e)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        state["processing_status"] = "recommendation_error"
    
    return state


async def finalize_application(state: ConversationState) -> ConversationState:
    """Finalize the application process"""
    
    try:
        logger.info("Finalizing application")
        
        # Prepare final decision
        final_decision = state.get("eligibility_result", {})
        state["final_decision"] = final_decision
        
        # Generate final summary message using conversation agent
        conversation_agent = ConversationAgent()
        final_response = conversation_agent._generate_eligibility_response(final_decision)
        
        state["messages"].append({
            "role": "assistant",
            "content": final_response,
            "timestamp": datetime.now().isoformat()
        })
        
        state["processing_status"] = "completed"
        state["current_step"] = ConversationStep.COMPLETION
        
        # Log finalization
        state["workflow_history"].append({
            "step": "finalize_application",
            "final_decision": final_decision,
            "completion_time": datetime.now().isoformat(),
            "status": "completed"
        })
        
        logger.info(f"Application finalized. Application ID: {state['application_id']}")
        
    except Exception as e:
        error_msg = f"Error finalizing application: {str(e)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        state["processing_status"] = "finalization_error"
    
    return state


async def handle_completion_chat(state: ConversationState) -> ConversationState:
    """Handle post-completion conversation"""
    
    try:
        # Get user input for completion chat
        user_input = state.get("user_input")
        if not user_input:
            # Get the latest user message
            user_messages = [msg for msg in state["messages"] if msg["role"] == "user"]
            if user_messages:
                user_input = user_messages[-1]["content"]
        
        if user_input:
            logger.info(f"Handling completion chat: '{user_input}'")
            
            # Initialize conversation agent
            conversation_agent = ConversationAgent()
            
            # Process completion conversation
            response = await conversation_agent._handle_completion_conversation(
                user_input, 
                {
                    "current_step": state["current_step"],
                    "collected_data": state["collected_data"],
                    "eligibility_result": state["eligibility_result"]
                }
            )
            
            # Add response to messages
            if "message" in response:
                state["messages"].append({
                    "role": "assistant",
                    "content": response["message"],
                    "timestamp": datetime.now().isoformat()
                })
                
                # CRITICAL FIX: Set the last_agent_response so it gets returned to user
                state["last_agent_response"] = response["message"]
            
            # Handle state updates (like restart)
            if "state_update" in response:
                for key, value in response["state_update"].items():
                    if key in state:
                        state[key] = value
            
            # Clear user input after processing
            state["user_input"] = None
            
            # Check if user wants to restart
            if response.get("state_update", {}).get("current_step") == ConversationStep.NAME_COLLECTION:
                state["processing_status"] = "restarting"
            else:
                state["processing_status"] = "completion_chat"
        
        else:
            state["processing_status"] = "waiting_for_completion_input"
        
    except Exception as e:
        error_msg = f"Error in completion chat: {str(e)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        state["processing_status"] = "completion_error"
        
        # Provide fallback response
        state["messages"].append({
            "role": "assistant", 
            "content": "I apologize, but I encountered an error processing your question. Please try asking again or start a new application.",
            "timestamp": datetime.now().isoformat()
        })
        state["last_agent_response"] = "I apologize, but I encountered an error processing your question. Please try asking again or start a new application."
    
    return state


# Conditional routing functions
def should_continue_conversation(state: ConversationState) -> str:
    """Determine if conversation should continue"""
    processing_status = state.get("processing_status", "")
    
    # If we're waiting for input, end the workflow
    if processing_status in ["waiting_for_input", "waiting_for_completion_input"]:
        return "waiting"
    
    # If completed, continue to completion chat
    if processing_status == "completed":
        return "continue"
    
    return "continue"


def determine_next_action(state: ConversationState) -> str:
    """Determine next action based on conversation state"""
    
    current_step = state.get("current_step", "")
    processing_status = state.get("processing_status", "")
    workflow_history = state.get("workflow_history", [])
    
    logger.debug(f"Determining next action. Step: {current_step}, Status: {processing_status}")
    
    # CRITICAL: Prevent infinite loops by checking workflow history
    recent_steps = [entry.get("step", "") for entry in workflow_history[-10:]]  # Last 10 steps
    if len(recent_steps) >= 5:
        # Check for repeated patterns
        if recent_steps[-1] == recent_steps[-2] == recent_steps[-3]:
            logger.warning(f"Detected loop in workflow: {recent_steps[-3:]}, ending workflow")
            return "waiting"
    
    # Handle waiting states - end workflow to prevent recursion
    if processing_status in [
        "waiting_for_input", 
        "waiting_for_completion_input", 
        "completed", 
        "error",
        "validation_error",
        "eligibility_error",
        "recommendation_error",
        "completion_error",
        "restarting"  # Add restarting to end workflow
    ]:
        logger.debug(f"Ending workflow due to status: {processing_status}")
        return "waiting"
    
    # Check if we're in completion phase
    if current_step == ConversationStep.COMPLETION or processing_status == "completion_chat":
        return "completion_chat"
    
    # Check if we have new documents to process
    uploaded_docs = len(state.get("uploaded_documents", []))
    processed_docs = len(state.get("processed_documents", []))
    if uploaded_docs > processed_docs:
        return "process_documents"
    
    # Handle validation flow
    if processing_status == "ready_for_validation":
        return "validate_info"
    elif processing_status == "validated":
        return "assess_eligibility"
    
    # Check if we need to validate information
    if current_step in [ConversationStep.DOCUMENT_COLLECTION, ConversationStep.ELIGIBILITY_PROCESSING]:
        validation_results = state.get("validation_results")
        if not validation_results:
            return "validate_info"
        elif validation_results.get("status") == "complete":
            return "assess_eligibility"
    
    # Check if ready for eligibility assessment
    if current_step == ConversationStep.ELIGIBILITY_PROCESSING:
        if not state.get("eligibility_result"):
            return "assess_eligibility"
        else:
            return "finalize"
    
    # Check if application is complete
    if processing_status == "completed":
        return "finalize"
    
    # If we have user input to process, continue conversation
    user_input = state.get("user_input")
    if user_input:
        return "continue_conversation"
    
    # SAFETY: If no clear next action, end workflow to prevent infinite loops
    logger.debug("No clear next action determined, ending workflow to prevent loops")
    return "waiting"


def after_document_processing(state: ConversationState) -> str:
    """Determine action after document processing"""
    if has_minimum_required_data(state["collected_data"]):
        return "validate_info"
    return "continue_conversation"


def after_validation(state: ConversationState) -> str:
    """Determine action after validation"""
    validation_results = state.get("validation_results", {})
    if validation_results.get("status") == "complete" and not validation_results.get("inconsistencies"):
        return "assess_eligibility"
    return "continue_conversation"


def after_eligibility_assessment(state: ConversationState) -> str:
    """Determine action after eligibility assessment"""
    if state.get("eligibility_result"):
        # Check if we should generate recommendations
        eligibility_result = state.get("eligibility_result", {})
        if not eligibility_result.get("economic_enablement"):
            return "generate_recommendations"
        else:
            return "finalize"
    return "finalize"


def after_completion_chat(state: ConversationState) -> str:
    """Determine action after completion chat"""
    processing_status = state.get("processing_status", "")
    
    if processing_status == "restarting":
        return "restart"
    elif processing_status in ["completion_chat"]:
        # CRITICAL FIX: End workflow after processing completion chat to prevent loops
        return "waiting"  # End workflow - response has been processed and added to messages
    elif processing_status in ["waiting_for_completion_input"]:
        return "waiting"  # End workflow when waiting for input
    else:
        return "waiting"


# Helper functions
def determine_document_type(file_path: str) -> str:
    """Determine document type from file path"""
    filename = os.path.basename(file_path).lower()
    
    if any(word in filename for word in ['bank', 'statement', 'transaction']):
        return "bank_statement"
    elif any(word in filename for word in ['emirates', 'id', 'identity']):
        return "emirates_id"
    elif any(word in filename for word in ['resume', 'cv', 'curriculum']):
        return "resume"
    elif any(word in filename for word in ['credit', 'report', 'score']):
        return "credit_report"
    elif any(word in filename for word in ['asset', 'liability', 'wealth']):
        return "assets_liabilities"
    else:
        return "other"


def perform_data_consistency_checks(collected_data: Dict) -> Dict:
    """Perform consistency checks on collected data"""
    
    inconsistencies = []
    
    # Check income vs employment status
    employment_status = collected_data.get("employment_status", "")
    monthly_income = collected_data.get("monthly_income", 0)
    
    if employment_status == "unemployed" and monthly_income > 1000:
        inconsistencies.append("High income reported for unemployed status")
    
    if employment_status == "employed" and monthly_income == 0:
        inconsistencies.append("Zero income reported for employed status")
    
    # Check family size reasonableness
    family_size = collected_data.get("family_size", 1)
    if family_size > 15:
        inconsistencies.append("Unusually large family size")
    
    # Check income per person
    if family_size > 0:
        income_per_person = monthly_income / family_size
        if income_per_person > 50000:
            inconsistencies.append("Very high per-person income")
    
    return {
        "inconsistencies": inconsistencies,
        "total_checks": 4,
        "passed_checks": 4 - len(inconsistencies)
    }


def has_minimum_required_data(collected_data: Dict) -> bool:
    """Check if we have minimum required data for assessment"""
    required_fields = ["name", "employment_status", "monthly_income", "family_size"]
    return all(field in collected_data for field in required_fields)


def generate_fallback_eligibility_decision(collected_data: Dict) -> Dict:
    """Generate a fallback eligibility decision when assessment fails"""
    
    monthly_income = collected_data.get("monthly_income", 0)
    family_size = collected_data.get("family_size", 1)
    
    # Simple rule-based assessment
    income_threshold = 3000 * family_size  # 3000 AED per person threshold
    
    if monthly_income < income_threshold:
        support_amount = max(500, (income_threshold - monthly_income) * 0.5)
        return {
            "eligible": True,
            "decision": "approved",
            "support_amount": support_amount,
            "breakdown": {
                "Base Support": 500,
                "Family Size Supplement": (family_size - 1) * 200,
                "Income Gap Support": support_amount - 500 - ((family_size - 1) * 200)
            },
            "reason": "Approved based on income threshold assessment (fallback decision)"
        }
    else:
        return {
            "eligible": False,
            "decision": "declined",
            "support_amount": 0,
            "reason": "Monthly income exceeds the threshold for direct financial support (fallback decision)"
        } 