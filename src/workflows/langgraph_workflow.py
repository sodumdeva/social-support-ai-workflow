"""
LangGraph Workflow for Social Support AI System

State-based conversation workflow using LangGraph for managing social support applications.
Integrates conversational AI (Ollama LLMs), document processing (OCR + LLM), ML-based 
eligibility assessment, and PostgreSQL database storage.

Main components:
- Conversation flow management through defined steps
- Document processing with OCR and LLM analysis  
- ML model integration for eligibility decisions
- Database storage with audit trails
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional, Any, Annotated
import asyncio
from datetime import datetime
import json
import operator
import re

# Import agents with correct paths
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.agents.conversation_agent import ConversationAgent, ConversationStep
from src.agents.data_extraction_agent import DataExtractionAgent
from src.agents.eligibility_agent import EligibilityAssessmentAgent

# Import database components
from src.models.database import (
    SessionLocal, 
    DatabaseManager, 
    Application, 
    Document,
    ApplicationReview,
    ApplicationStatus,
    DocumentType,
    EmploymentStatus,
    HousingStatus
)

# Import logging configuration
from src.utils.logging_config import get_logger, WorkflowLogger
logger = get_logger("langgraph_workflow")
demo_logger = WorkflowLogger("workflow")


class ConversationState(TypedDict):
    """
    State structure for the LangGraph conversation workflow.
    
    Contains conversation messages, user data, processing status, and workflow history
    for managing the complete application process through state transitions.
    """
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
    processed_documents: List[str]
    validation_results: Dict


def create_conversation_workflow():
    """
    Create LangGraph workflow for conversational application processing.
    
    Builds state-based workflow with nodes for conversation, document processing,
    validation, eligibility assessment, and database storage. Uses conditional
    routing based on processing status and conversation steps.
    """
    
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
    # CRITICAL FIX: Don't override important processing statuses
    current_status = state.get("processing_status", "")
    if current_status in ["documents_need_processing", "ready_for_validation", "validated", "completed", "restarting"]:
        # Preserve important processing statuses
        logger.info(f"Preserving processing status: {current_status}")
    elif state.get("user_input"):
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
    """
    Process user messages through ConversationAgent and manage state transitions.
    
    Handles message processing, conversation flow, restart scenarios, and determines
    next processing status for workflow routing. Clears user input after processing.
    """
    
    try:
        user_input = state.get("user_input")
        if not user_input:
            # CRITICAL FIX: Don't override important processing statuses when there's no user input
            current_status = state.get("processing_status", "")
            if current_status in ["validated", "eligibility_assessed", "documents_need_processing", "ready_for_validation"]:
                logger.info(f"No user input, but preserving important processing status: {current_status}")
                return state  # Return without changing status
            else:
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
        # FIXED: Only trigger restart if explicitly requested, not on normal flow changes
        if (state["current_step"] == ConversationStep.NAME_COLLECTION and 
            "state_update" in response and 
            response["state_update"].get("collected_data") == {} and
            user_input and "start new" in user_input.lower()):
            # This is an explicit restart request - reset all relevant state
            logger.info("Detected explicit restart request - resetting conversation state")
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
            # CRITICAL FIX: Check for unprocessed documents first
            uploaded_docs = len(state.get("uploaded_documents", []))
            processed_docs = len(state.get("processed_documents", []))
            
            if uploaded_docs > processed_docs:
                # We have unprocessed documents - trigger document processing
                state["processing_status"] = "documents_need_processing"
                logger.info(f"Found {uploaded_docs - processed_docs} unprocessed documents, will trigger processing")
            else:
                # CRITICAL FIX: In document collection, wait for user to decide whether to upload documents or proceed
                # Don't automatically proceed to validation just because we have minimum data
                state["processing_status"] = "waiting_for_input"
                logger.info("In document collection, waiting for user decision to upload documents or proceed")
        elif state["current_step"] == ConversationStep.ELIGIBILITY_PROCESSING:
            # CRITICAL FIX: Handle eligibility processing step
            uploaded_docs = len(state.get("uploaded_documents", []))
            processed_docs = len(state.get("processed_documents", []))
            
            if uploaded_docs > processed_docs:
                # Process documents first before eligibility assessment
                state["processing_status"] = "documents_need_processing"
                logger.info(f"ELIGIBILITY_PROCESSING: Found {uploaded_docs - processed_docs} unprocessed documents, processing first")
            elif has_minimum_required_data(state["collected_data"]):
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
    """
    Process uploaded documents using OCR and LLM analysis for data extraction.
    
    Combines Tesseract OCR with local LLM models to extract structured data from
    documents, performs verification against user data, and updates workflow state.
    """
    
    try:
        demo_logger.log_step("DOCUMENT_PROCESSING", "🚀 Starting document processing workflow")
        logger.info("🔍 Starting document processing with detailed extraction logging")
        
        # Get unprocessed documents
        processed_docs = state.get("processed_documents", [])
        uploaded_docs = state.get("uploaded_documents", [])
        unprocessed_docs = [doc for doc in uploaded_docs if doc not in processed_docs]
        
        demo_logger.log_step("DOCUMENT_ANALYSIS", f"📊 Found {len(uploaded_docs)} uploaded, {len(processed_docs)} processed, {len(unprocessed_docs)} pending")
        logger.info(f"📄 Document status: {len(uploaded_docs)} uploaded, {len(processed_docs)} processed, {len(unprocessed_docs)} pending")
        
        if not unprocessed_docs:
            demo_logger.log_step("DOCUMENT_PROCESSING", "ℹ️ No new documents to process")
            logger.info("ℹ️  No new documents to process")
            state["processing_status"] = "no_documents_to_process"
            return state
        
        # Initialize data extraction agent
        data_extraction_agent = DataExtractionAgent()
        all_extraction_results = {}
        user_provided_data = state.get("collected_data", {})
        
        demo_logger.log_step("USER_DATA", f"👤 User provided: {', '.join(user_provided_data.keys())}")
        logger.info(f"👤 User-provided data for verification: {json.dumps(user_provided_data, indent=2)}")
        
        for doc_path in unprocessed_docs:
            try:
                filename = os.path.basename(doc_path)
                demo_logger.log_document_processing("UNKNOWN", filename, "STARTING")
                logger.info(f"🔍 Processing document: {doc_path}")
                
                # Determine document type from filename
                doc_type = determine_document_type(doc_path)
                demo_logger.log_document_processing(doc_type.upper(), filename, "TYPE_DETECTED")
                logger.info(f"📋 Detected document type: {doc_type}")
                
                # Process document
                demo_logger.log_step("MULTIMODAL_AI", f"🤖 Sending {doc_type} to LLaVA for analysis...")
                extraction_result = await data_extraction_agent.process({
                    "documents": [{"file_path": doc_path, "document_type": doc_type}],
                    "extraction_mode": "conversational"
                })
                
                demo_logger.log_step("EXTRACTION_RESULT", f"📊 LLaVA processing completed: {extraction_result.get('status')}")
                logger.info(f"📊 Extraction result status: {extraction_result.get('status')}")
                
                if extraction_result.get("status") == "success":
                    extracted_data = extraction_result.get("extraction_results", {}).get(doc_type, {})
                    demo_logger.log_step("DATA_EXTRACTION", f"✅ Successfully extracted data from {doc_type}")
                    logger.info(f"📄 Raw extraction data for {doc_type}: {json.dumps(extracted_data, indent=2)}")
                    
                    # IMPROVED: Handle multiple status checking scenarios
                    extraction_successful = False
                    structured_data = None
                    confidence = 0.0
                    processing_time = 0
                    
                    # Check for success in multiple ways
                    if extracted_data.get("status") == "success":
                        extraction_successful = True
                        structured_data = extracted_data.get("structured_data", {})
                        confidence = extracted_data.get("extraction_confidence", 0.0)
                        processing_time = extracted_data.get("processing_time_ms", 0)
                    elif "structured_data" in extracted_data:
                        extraction_successful = True
                        structured_data = extracted_data.get("structured_data", {})
                        confidence = extracted_data.get("extraction_confidence", 0.0)
                        processing_time = extracted_data.get("processing_time_ms", 0)
                    elif extracted_data:  # Any data extracted
                        extraction_successful = True
                        structured_data = extracted_data
                        confidence = extracted_data.get("extraction_confidence", 0.0)
                        processing_time = extracted_data.get("processing_time_ms", 0)
                    
                    if extraction_successful and structured_data:
                        demo_logger.log_step("EXTRACTION_SUCCESS", f"📋 Extracted {len(structured_data)} data fields (Confidence: {confidence:.2f}, Time: {processing_time:.0f}ms)")
                        logger.info(f"✅ Successfully extracted structured data from {doc_type}")
                        logger.info(f"📋 Structured data: {json.dumps(structured_data, indent=2)}")
                        
                        # SIMPLIFIED: Perform only name verification for bank statements
                        demo_logger.log_step("DATA_VERIFICATION", f"🔍 Starting simplified name verification for {doc_type}...")
                        verification_results = verify_name_only(structured_data, user_provided_data, doc_type)
                        
                        # Log verification results
                        summary = verification_results.get("_verification_summary", {})
                        matches = summary.get("matches", 0)
                        mismatches = summary.get("mismatches", 0)
                        overall_score = summary.get("overall_score", 0)
                        quality = summary.get("verification_quality", "Unknown")
                        
                        demo_logger.log_data_verification(doc_type.upper(), matches, mismatches, overall_score)
                        demo_logger.log_step("VERIFICATION_QUALITY", f"🎯 Data quality: {quality} ({overall_score:.2f} score)")
                        
                        logger.info(f"🔍 Name verification results for {doc_type}:")
                        for field, result in verification_results.items():
                            if isinstance(result, dict) and "status" in result:
                                if result["status"] == "match":
                                    logger.info(f"  ✅ {field}: MATCH - User: '{result.get('user_value')}' | Document: '{result.get('extracted_value')}'")
                                elif result["status"] == "mismatch":
                                    logger.warning(f"  ❌ {field}: MISMATCH - User: '{result.get('user_value')}' | Document: '{result.get('extracted_value')}'")
                                elif result["status"] == "missing_user":
                                    logger.info(f"  ➕ {field}: NEW DATA from document - '{result.get('extracted_value')}'")
                                elif result["status"] == "missing_document":
                                    logger.info(f"  ⚠️  {field}: User provided '{result.get('user_value')}' but not found in document")
                        
                        all_extraction_results[doc_type] = {
                            "structured_data": structured_data,
                            "verification_results": verification_results,
                            "extraction_confidence": confidence
                        }
                        
                        demo_logger.log_step("DATA_MERGE", f"🔄 Name verification completed for {doc_type}")
                        logger.info(f"🔄 Name verification completed for {doc_type}")
                    
                    else:
                        error_msg = extracted_data.get("error", "No structured data found")
                        demo_logger.log_document_processing(doc_type.upper(), filename, f"FAILED: {error_msg}")
                        logger.error(f"❌ Document extraction failed for {doc_type}: {error_msg}")
                
                else:
                    demo_logger.log_document_processing("UNKNOWN", filename, f"FAILED: {extraction_result.get('error', 'Unknown error')}")
                    logger.error(f"❌ Document processing failed for {doc_path}: {extraction_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                demo_logger.log_document_processing("UNKNOWN", filename, f"ERROR: {str(e)}")
                logger.error(f"❌ Error processing document {doc_path}: {str(e)}")
                # Continue processing other documents even if one fails
                continue
        
        # Update processed documents list - ONLY add documents that were successfully processed
        if "processed_documents" not in state:
            state["processed_documents"] = []
        
        # Only add documents that were actually processed successfully
        successfully_processed = []
        for doc_path in unprocessed_docs:
            filename = os.path.basename(doc_path)
            doc_type = determine_document_type(doc_path)
            if doc_type in all_extraction_results:
                successfully_processed.append(doc_path)
                logger.info(f"✅ Marking {filename} as successfully processed")
            else:
                logger.warning(f"⚠️ {filename} was not successfully processed, will retry later")
        
        state["processed_documents"].extend(successfully_processed)
        logger.info(f"📊 Updated processed_documents: {len(state['processed_documents'])} total, {len(successfully_processed)} newly processed")
        
        # Generate response about document processing
        if all_extraction_results:
            doc_types = list(all_extraction_results.keys())
            demo_logger.log_step("PROCESSING_COMPLETE", f"✅ Successfully processed {len(all_extraction_results)} documents: {', '.join(doc_types)}")
            
            # SIMPLIFIED: Create simple response about name verification
            response_parts = [f"I've processed your {', '.join(doc_types).replace('_', ' ')}. Here's what I found:"]
            
            for doc_type, results in all_extraction_results.items():
                verification = results.get("verification_results", {})
                name_result = verification.get("name", {})
                
                if name_result.get("status") == "match":
                    response_parts.append(f"✅ {doc_type.replace('_', ' ').title()}: Name confirmed")
                elif name_result.get("status") == "mismatch":
                    response_parts.append(f"⚠️ {doc_type.replace('_', ' ').title()}: Name differs from document - using your provided name")
                elif name_result.get("status") == "missing_user":
                    extracted_name = name_result.get("extracted_value", "")
                    response_parts.append(f"➕ {doc_type.replace('_', ' ').title()}: Found name '{extracted_name}' in document")
                elif name_result.get("status") == "missing_document":
                    response_parts.append(f"ℹ️ {doc_type.replace('_', ' ').title()}: Using your provided name (not found in document)")
                else:
                    response_parts.append(f"ℹ️ {doc_type.replace('_', ' ').title()}: Document processed successfully")
            
            response_msg = "\n".join(response_parts)
            
            state["messages"].append({
                "role": "assistant",
                "content": response_msg,
                "timestamp": datetime.now().isoformat()
            })
            
            demo_logger.log_step("USER_RESPONSE", f"📝 Generated simplified response for user")
            logger.info(f"📝 Generated response about document processing: {response_msg}")
        
        state["processing_status"] = "ready_for_validation"
        
        # Log document processing summary
        state["workflow_history"].append({
            "step": "process_documents",
            "documents_processed": len(all_extraction_results),
            "extraction_results": all_extraction_results,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
        demo_logger.log_step("DOCUMENT_PROCESSING", f"🏁 Document processing completed successfully")
        logger.info(f"✅ Document processing completed. Successfully processed {len(all_extraction_results)} documents")
        
    except Exception as e:
        error_msg = f"Error processing documents: {str(e)}"
        demo_logger.log_step("DOCUMENT_PROCESSING", f"❌ FAILED: {error_msg}", "ERROR")
        logger.error(f"❌ {error_msg}")
        state["error_messages"].append(error_msg)
        state["processing_status"] = "document_error"
    
    return state


def verify_name_only(extracted_data: Dict, user_data: Dict, doc_type: str) -> Dict:
    """Simplified verification focusing only on name matching"""
    
    verification_results = {}
    user_name = user_data.get("name", "")
    
    logger.info(f"🔍 Starting simplified name verification for {doc_type}")
    logger.info(f"👤 User name: '{user_name}'")
    
    # Try to find name in different ways based on document type
    extracted_name = None
    
    if doc_type == "bank_statement":
        # For bank statements, try multiple approaches
        # 1. Check structured data for account holder name
        if isinstance(extracted_data, dict):
            # Try nested structure first
            account_info = extracted_data.get("account_info", {})
            if account_info and isinstance(account_info, dict):
                extracted_name = account_info.get("account_holder_name")
            
            # If not found, check raw text for fallback parsing
            if not extracted_name and extracted_data.get("parsing_method") == "text_fallback":
                raw_text = extracted_data.get("raw_text", "")
                if user_name and raw_text and user_name.upper() in raw_text.upper():
                    extracted_name = f"Found in raw text: {user_name}"
                    logger.info(f"🔍 FALLBACK: Found name '{user_name}' in raw text")
    
    elif doc_type == "emirates_id":
        # For Emirates ID
        personal_info = extracted_data.get("personal_info", {})
        if personal_info:
            extracted_name = personal_info.get("full_name")
    
    elif doc_type == "resume":
        # For resume
        personal_contact = extracted_data.get("personal_contact", {})
        if personal_contact:
            extracted_name = personal_contact.get("full_name")
    
    elif doc_type == "salary_certificate":
        # For salary certificate
        employee_info = extracted_data.get("employee_info", {})
        if employee_info:
            extracted_name = employee_info.get("employee_name")
    
    # Perform name comparison
    if extracted_name and user_name:
        # Both values exist - perform comparison
        match_result = compare_names(extracted_name.lower().strip(), user_name.lower().strip())
        verification_results["name"] = {
            "status": match_result["status"],
            "user_value": user_name,
            "extracted_value": extracted_name,
            "confidence": match_result["confidence"],
            "match_score": match_result.get("match_score", 0.0),
            "comparison_notes": match_result.get("notes", "")
        }
        
        if match_result["status"] == "match":
            logger.info(f"  ✅ NAME MATCH: '{user_name}' ≈ '{extracted_name}'")
        else:
            logger.warning(f"  ❌ NAME MISMATCH: '{user_name}' ≠ '{extracted_name}'")
            
    elif extracted_name and not user_name:
        # Found name in document but user didn't provide one
        verification_results["name"] = {
            "status": "missing_user",
            "extracted_value": extracted_name,
            "confidence": "medium",
            "notes": "Found name in document but user didn't provide one"
        }
        logger.info(f"  ➕ NEW NAME: Found '{extracted_name}' in document")
        
    elif user_name and not extracted_name:
        # User provided name but not found in document
        verification_results["name"] = {
            "status": "missing_document",
            "user_value": user_name,
            "confidence": "low",
            "notes": "User provided name but not found in document"
        }
        logger.warning(f"  ⚠️  MISSING: User name '{user_name}' not found in document")
    
    else:
        # Neither found
        verification_results["name"] = {
            "status": "missing_both",
            "confidence": "low",
            "notes": "No name found in either user data or document"
        }
        logger.warning(f"  ⚠️  NO NAMES: Neither user nor document has name information")
    
    # Calculate simple verification score
    if verification_results.get("name", {}).get("status") == "match":
        overall_score = verification_results["name"].get("match_score", 1.0)
        matches = 1
        mismatches = 0
        quality = "High" if overall_score > 0.8 else "Medium" if overall_score > 0.5 else "Low"
    elif verification_results.get("name", {}).get("status") in ["missing_user", "missing_document"]:
        overall_score = 0.5
        matches = 0
        mismatches = 0
        quality = "Partial"
    else:
        overall_score = 0.0
        matches = 0
        mismatches = 1
        quality = "Low"
    
    verification_results["_verification_summary"] = {
        "overall_score": overall_score,
        "total_fields_checked": 1,
        "matches": matches,
        "mismatches": mismatches,
        "new_data_found": 1 if verification_results.get("name", {}).get("status") == "missing_user" else 0,
        "verification_quality": quality
    }
    
    logger.info(f"📊 Simplified Name Verification Summary for {doc_type}:")
    logger.info(f"   Overall Score: {overall_score:.2f}")
    logger.info(f"   Quality: {quality}")
    
    return verification_results


def verify_extracted_data_against_user_input(extracted_data: Dict, user_data: Dict, doc_type: str) -> Dict:
    """Enhanced verification of extracted document data against user-provided information"""
    
    verification_results = {}
    
    logger.info(f"🔍 Starting detailed verification for {doc_type}")
    logger.info(f"📄 Document data keys: {list(extracted_data.keys())}")
    logger.info(f"👤 User data keys: {list(user_data.keys())}")
    
    # Define comprehensive field mappings for different document types
    field_mappings = {
        "emirates_id": {
            # Map document fields to user data fields
            "personal_info.full_name": "name",
            "personal_info.emirates_id_number": "emirates_id",
            "personal_info.nationality": "nationality",
            "personal_info.date_of_birth": "date_of_birth",
            "personal_info.gender": "gender"
        },
        "bank_statement": {
            "account_info.account_holder_name": "name",
            # Temporarily focusing only on name verification
            # "income_verification.monthly_salary": "monthly_income",
            # "income_verification.total_monthly_income": "monthly_income",
            # "income_verification.employer_name": "employer_name",
            # "financial_data.closing_balance": "savings_amount"
        },
        "resume": {
            "personal_contact.full_name": "name",
            "personal_contact.email_address": "email",
            "personal_contact.phone_number": "phone_number",
            "current_employment.current_employer": "employer_name",
            "current_employment.current_position": "job_title",
            "current_employment.employment_status": "employment_status",
            "professional_summary.total_experience_months": "work_experience_months"
        },
        "salary_certificate": {
            "employee_info.employee_name": "name",
            "employee_info.position": "job_title",
            "employer_info.company_name": "employer_name",
            "salary_details.gross_salary": "monthly_income",
            "salary_details.basic_salary": "monthly_income",
            "employment_details.employment_type": "employment_status"
        }
    }
    
    mappings = field_mappings.get(doc_type, {})
    
    # Perform field-by-field verification
    for doc_field, user_field in mappings.items():
        extracted_value = get_nested_value(extracted_data, doc_field)
        user_value = user_data.get(user_field)
        
        logger.info(f"🔍 Verifying {user_field}: Document='{extracted_value}' vs User='{user_value}'")
        
        if extracted_value is not None and user_value is not None:
            # Both values exist - perform detailed comparison
            match_result = compare_values(extracted_value, user_value, user_field)
            verification_results[user_field] = {
                "status": match_result["status"],
                "user_value": user_value,
                "extracted_value": extracted_value,
                "confidence": match_result["confidence"],
                "match_score": match_result.get("match_score", 0.0),
                "comparison_notes": match_result.get("notes", "")
            }
            
            if match_result["status"] == "match":
                logger.info(f"  ✅ MATCH: {user_field}")
            elif match_result["status"] == "mismatch":
                logger.warning(f"  ❌ MISMATCH: {user_field} - {match_result.get('notes', '')}")
            else:
                logger.info(f"  ⚠️  PARTIAL: {user_field} - {match_result.get('notes', '')}")
                
        elif extracted_value is not None and user_value is None:
            # New data from document
            verification_results[user_field] = {
                "status": "missing_user",
                "extracted_value": extracted_value,
                "confidence": "medium",
                "notes": "Found in document but not provided by user"
            }
            logger.info(f"  ➕ NEW DATA: {user_field} = '{extracted_value}'")
            
        elif extracted_value is None and user_value is not None:
            # User provided data not found in document
            verification_results[user_field] = {
                "status": "missing_document",
                "user_value": user_value,
                "confidence": "low",
                "notes": "User provided but not found in document"
            }
            logger.warning(f"  ⚠️  MISSING: {user_field} = '{user_value}' not found in document")
    
    # SPECIAL HANDLING: For fallback parsing cases, check if name appears in raw_text
    if extracted_data.get("parsing_method") == "text_fallback" and doc_type == "bank_statement":
        raw_text = extracted_data.get("raw_text", "")
        user_name = user_data.get("name", "")
        
        if user_name and raw_text and user_name.upper() in raw_text.upper():
            logger.info(f"🔍 FALLBACK: Found name '{user_name}' in raw text for fallback parsing")
            verification_results["name"] = {
                "status": "match",
                "user_value": user_name,
                "extracted_value": f"Found in raw text: {user_name}",
                "confidence": "medium",
                "match_score": 0.8,
                "comparison_notes": "Name found in fallback parsing raw text"
            }
            logger.info(f"  ✅ FALLBACK MATCH: name")
    
    # Perform document-specific additional verifications
    additional_checks = perform_document_specific_checks(extracted_data, user_data, doc_type)
    if additional_checks:
        verification_results.update(additional_checks)
    
    # Calculate overall verification score
    overall_score = calculate_verification_score(verification_results)
    verification_results["_verification_summary"] = {
        "overall_score": overall_score,
        "total_fields_checked": len([r for r in verification_results.values() if isinstance(r, dict) and "status" in r]),
        "matches": len([r for r in verification_results.values() if isinstance(r, dict) and r.get("status") == "match"]),
        "mismatches": len([r for r in verification_results.values() if isinstance(r, dict) and r.get("status") == "mismatch"]),
        "new_data_found": len([r for r in verification_results.values() if isinstance(r, dict) and r.get("status") == "missing_user"]),
        "verification_quality": "High" if overall_score > 0.8 else "Medium" if overall_score > 0.5 else "Low"
    }
    
    logger.info(f"📊 Verification Summary for {doc_type}:")
    logger.info(f"   Overall Score: {overall_score:.2f}")
    logger.info(f"   Matches: {verification_results['_verification_summary']['matches']}")
    logger.info(f"   Mismatches: {verification_results['_verification_summary']['mismatches']}")
    logger.info(f"   New Data: {verification_results['_verification_summary']['new_data_found']}")
    
    return verification_results


def compare_values(extracted_value: Any, user_value: Any, field_name: str) -> Dict:
    """Enhanced value comparison with field-specific logic"""
    
    # Handle None or empty values
    if not extracted_value or not user_value:
        return {"status": "partial", "confidence": "low", "notes": "One or both values are empty"}
    
    # Convert to strings for comparison
    extracted_str = str(extracted_value).strip().lower()
    user_str = str(user_value).strip().lower()
    
    # Field-specific comparison logic
    if field_name in ["monthly_income", "salary", "gross_salary", "basic_salary"]:
        return compare_financial_values(extracted_value, user_value)
    elif field_name in ["name", "full_name", "employee_name"]:
        return compare_names(extracted_str, user_str)
    elif field_name in ["emirates_id", "emirates_id_number"]:
        return compare_id_numbers(extracted_str, user_str)
    elif field_name in ["employer_name", "company_name"]:
        return compare_company_names(extracted_str, user_str)
    elif field_name in ["employment_status"]:
        return compare_employment_status(extracted_str, user_str)
    else:
        # Generic string comparison
        return compare_generic_strings(extracted_str, user_str)


def compare_financial_values(extracted_value: Any, user_value: Any) -> Dict:
    """Compare financial values with intelligent tolerance for salary ranges"""
    
    try:
        # Extract numeric values
        extracted_num = extract_numeric_value(str(extracted_value))
        user_num = extract_numeric_value(str(user_value))
        
        if extracted_num is None or user_num is None:
            return {"status": "partial", "confidence": "low", "notes": "Could not extract numeric values"}
        
        # Calculate percentage difference
        diff_percent = abs(extracted_num - user_num) / max(extracted_num, user_num) * 100
        
        # Enhanced tolerance for salary ranges
        if diff_percent <= 3:  # Within 3% - exact match
            return {
                "status": "match",
                "confidence": "high",
                "match_score": 1.0,
                "notes": f"Values match within {diff_percent:.1f}% tolerance"
            }
        elif diff_percent <= 10:  # Within 10% - very good match
            return {
                "status": "match",
                "confidence": "high",
                "match_score": 0.95,
                "notes": f"Values match within acceptable range ({diff_percent:.1f}% difference)"
            }
        elif diff_percent <= 20:  # Within 20% - acceptable for salary variations
            return {
                "status": "partial",
                "confidence": "medium",
                "match_score": 0.8,
                "notes": f"Values within salary range ({diff_percent:.1f}% difference) - possible bonus/allowance variation"
            }
        elif diff_percent <= 35:  # Within 35% - possible different salary components
            return {
                "status": "partial",
                "confidence": "medium",
                "match_score": 0.6,
                "notes": f"Values differ by {diff_percent:.1f}% - possible basic vs gross salary difference"
            }
        else:
            return {
                "status": "mismatch",
                "confidence": "high",
                "match_score": 0.0,
                "notes": f"Significant difference: {diff_percent:.1f}%"
            }
            
    except Exception as e:
        return {"status": "partial", "confidence": "low", "notes": f"Comparison error: {str(e)}"}


def compare_names(extracted_name: str, user_name: str) -> Dict:
    """Compare names with intelligent tolerance for variations and cultural naming patterns"""
    
    # Normalize names
    extracted_clean = normalize_name(extracted_name)
    user_clean = normalize_name(user_name)
    
    # Exact match
    if extracted_clean == user_clean:
        return {"status": "match", "confidence": "high", "match_score": 1.0}
    
    # Split into parts for analysis
    extracted_parts = [part for part in extracted_clean.split() if len(part) > 1]  # Ignore single letters
    user_parts = [part for part in user_clean.split() if len(part) > 1]
    
    # Check for exact substring matches (handles middle name variations)
    if extracted_clean in user_clean or user_clean in extracted_clean:
        return {
            "status": "match",
            "confidence": "high",
            "match_score": 0.95,
            "notes": "Name is a subset of the other - likely middle name variation"
        }
    
    # Calculate matching parts
    common_parts = []
    for ext_part in extracted_parts:
        for user_part in user_parts:
            # Exact match
            if ext_part == user_part:
                common_parts.append(ext_part)
            # Partial match for longer names (handles abbreviations)
            elif len(ext_part) >= 4 and len(user_part) >= 4:
                if ext_part.startswith(user_part[:3]) or user_part.startswith(ext_part[:3]):
                    common_parts.append(ext_part)
    
    # Remove duplicates
    common_parts = list(set(common_parts))
    total_unique_parts = len(set(extracted_parts + user_parts))
    
    # Enhanced matching logic for Arabic/Middle Eastern names
    if len(common_parts) >= 2:  # At least 2 parts match
        match_ratio = len(common_parts) / max(len(extracted_parts), len(user_parts))
        
        if match_ratio >= 0.8:  # 80% of parts match
            return {
                "status": "match",
                "confidence": "high",
                "match_score": 0.9,
                "notes": f"Strong name match: {len(common_parts)} common parts ({match_ratio:.1%} similarity)"
            }
        elif match_ratio >= 0.6:  # 60% of parts match
            return {
                "status": "match",
                "confidence": "medium",
                "match_score": 0.85,
                "notes": f"Good name match: {len(common_parts)} common parts ({match_ratio:.1%} similarity)"
            }
        else:
            return {
                "status": "partial",
                "confidence": "medium",
                "match_score": 0.7,
                "notes": f"Partial name match: {len(common_parts)} common parts"
            }
    
    # Check for single strong match (first name or last name)
    elif len(common_parts) == 1:
        # If it's a longer name part (likely significant)
        if len(common_parts[0]) >= 4:
            return {
                "status": "partial",
                "confidence": "medium",
                "match_score": 0.6,
                "notes": f"Single significant name part matches: '{common_parts[0]}'"
            }
        else:
            return {
                "status": "partial",
                "confidence": "low",
                "match_score": 0.4,
                "notes": f"Only short name part matches: '{common_parts[0]}'"
            }
    
    # Check for initials match (common in documents)
    extracted_initials = ''.join([part[0] for part in extracted_parts if part])
    user_initials = ''.join([part[0] for part in user_parts if part])
    
    if len(extracted_initials) >= 2 and extracted_initials == user_initials:
        return {
            "status": "partial",
            "confidence": "medium",
            "match_score": 0.5,
            "notes": f"Name initials match: {extracted_initials}"
        }
    
    # Use fuzzy string matching as fallback
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, extracted_clean, user_clean).ratio()
    
    if similarity >= 0.7:
        return {
            "status": "partial",
            "confidence": "low",
            "match_score": similarity,
            "notes": f"Names have {similarity:.1%} character similarity"
        }
    
    return {
        "status": "mismatch",
        "confidence": "high",
        "match_score": 0.0,
        "notes": "Names do not match sufficiently"
    }


def compare_id_numbers(extracted_id: str, user_id: str) -> Dict:
    """Compare ID numbers with exact matching"""
    
    # Remove formatting characters
    extracted_clean = re.sub(r'[^0-9]', '', extracted_id)
    user_clean = re.sub(r'[^0-9]', '', user_id)
    
    if extracted_clean == user_clean:
        return {"status": "match", "confidence": "high", "match_score": 1.0}
    
    # Check for partial matches (useful for redacted IDs)
    if len(extracted_clean) >= 10 and len(user_clean) >= 10:
        # Compare first and last few digits
        if extracted_clean[:4] == user_clean[:4] and extracted_clean[-4:] == user_clean[-4:]:
            return {
                "status": "partial",
                "confidence": "medium",
                "match_score": 0.7,
                "notes": "Partial ID match - first and last digits match"
            }
    
    return {"status": "mismatch", "confidence": "high", "match_score": 0.0}


def compare_company_names(extracted_company: str, user_company: str) -> Dict:
    """Compare company names with tolerance for variations"""
    
    # Normalize company names
    extracted_norm = normalize_company_name(extracted_company)
    user_norm = normalize_company_name(user_company)
    
    if extracted_norm == user_norm:
        return {"status": "match", "confidence": "high", "match_score": 1.0}
    
    # Check for substring matches
    if extracted_norm in user_norm or user_norm in extracted_norm:
        return {
            "status": "partial",
            "confidence": "medium",
            "match_score": 0.8,
            "notes": "Company names partially match"
        }
    
    # Check for common abbreviations
    if check_company_abbreviations(extracted_norm, user_norm):
        return {
            "status": "partial",
            "confidence": "medium",
            "match_score": 0.7,
            "notes": "Possible company name abbreviation match"
        }
    
    return {"status": "mismatch", "confidence": "high", "match_score": 0.0}


def compare_employment_status(extracted_status: str, user_status: str) -> Dict:
    """Compare employment status with normalization"""
    
    status_mappings = {
        "employed": ["employed", "working", "full-time", "part-time", "permanent", "contract"],
        "unemployed": ["unemployed", "not working", "jobless", "seeking work"],
        "self-employed": ["self-employed", "freelance", "consultant", "business owner", "entrepreneur"]
    }
    
    def normalize_status(status):
        status = status.lower().strip()
        for normalized, variants in status_mappings.items():
            if any(variant in status for variant in variants):
                return normalized
        return status
    
    extracted_norm = normalize_status(extracted_status)
    user_norm = normalize_status(user_status)
    
    if extracted_norm == user_norm:
        return {"status": "match", "confidence": "high", "match_score": 1.0}
    
    return {"status": "mismatch", "confidence": "high", "match_score": 0.0}


def compare_generic_strings(extracted_str: str, user_str: str) -> Dict:
    """Generic string comparison with fuzzy matching"""
    
    if extracted_str == user_str:
        return {"status": "match", "confidence": "high", "match_score": 1.0}
    
    # Calculate similarity ratio
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, extracted_str, user_str).ratio()
    
    if similarity >= 0.9:
        return {"status": "match", "confidence": "high", "match_score": similarity}
    elif similarity >= 0.7:
        return {"status": "partial", "confidence": "medium", "match_score": similarity}
    else:
        return {"status": "mismatch", "confidence": "high", "match_score": similarity}


def perform_document_specific_checks(extracted_data: Dict, user_data: Dict, doc_type: str) -> Dict:
    """Perform additional document-specific verification checks"""
    
    additional_checks = {}
    
    if doc_type == "bank_statement":
        # Check income consistency
        income_verification = extracted_data.get("income_verification", {})
        if income_verification.get("income_consistency") == "Irregular":
            additional_checks["income_stability_concern"] = {
                "status": "warning",
                "confidence": "medium",
                "notes": "Document indicates irregular income pattern"
            }
        
        # Check for financial stress indicators
        spending_patterns = extracted_data.get("spending_patterns", {})
        stress_indicators = spending_patterns.get("financial_stress_indicators", [])
        if stress_indicators:
            additional_checks["financial_stress_detected"] = {
                "status": "warning",
                "confidence": "high",
                "notes": f"Financial stress indicators found: {', '.join(stress_indicators)}"
            }
    
    elif doc_type == "salary_certificate":
        # Check document authenticity
        doc_verification = extracted_data.get("document_verification", {})
        if doc_verification.get("document_authenticity") == "Questionable":
            additional_checks["document_authenticity_concern"] = {
                "status": "warning",
                "confidence": "high",
                "notes": "Document authenticity appears questionable"
            }
    
    elif doc_type == "resume":
        # Check for employment gaps
        professional_summary = extracted_data.get("professional_summary", {})
        employment_gaps = professional_summary.get("employment_gaps", [])
        if employment_gaps:
            additional_checks["employment_gaps_detected"] = {
                "status": "info",
                "confidence": "medium",
                "notes": f"Employment gaps detected: {', '.join(employment_gaps)}"
            }
    
    return additional_checks


# Helper functions
def extract_numeric_value(value_str: str) -> Optional[float]:
    """Extract numeric value from string, handling various formats"""
    try:
        # Remove common non-numeric characters
        cleaned = re.sub(r'[^\d.,]', '', str(value_str))
        # Handle comma as thousands separator
        cleaned = cleaned.replace(',', '')
        return float(cleaned)
    except:
        return None


def normalize_name(name: str) -> str:
    """Normalize name for comparison"""
    return re.sub(r'[^a-z\s]', '', name.lower()).strip()


def normalize_company_name(company: str) -> str:
    """Normalize company name for comparison"""
    # Remove common suffixes and prefixes
    company = re.sub(r'\b(llc|ltd|inc|corp|company|co|group|holdings)\b', '', company.lower())
    return re.sub(r'[^a-z\s]', '', company).strip()


def check_company_abbreviations(name1: str, name2: str) -> bool:
    """Check if company names might be abbreviations of each other"""
    # Simple check for initials
    words1 = name1.split()
    words2 = name2.split()
    
    if len(words1) > 1 and len(words2) == 1:
        initials = ''.join([word[0] for word in words1 if word])
        return initials == words2[0]
    elif len(words2) > 1 and len(words1) == 1:
        initials = ''.join([word[0] for word in words2 if word])
        return initials == words1[0]
    
    return False


def calculate_verification_score(verification_results: Dict) -> float:
    """Calculate overall verification score"""
    
    scores = []
    for key, result in verification_results.items():
        if isinstance(result, dict) and "match_score" in result:
            scores.append(result["match_score"])
        elif isinstance(result, dict) and result.get("status") == "match":
            scores.append(1.0)
        elif isinstance(result, dict) and result.get("status") == "partial":
            scores.append(0.5)
    
    return sum(scores) / len(scores) if scores else 0.0


def get_nested_value(data: Dict, path: str):
    """Get value from nested dictionary using dot notation"""
    keys = path.split('.')
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    
    return current


def normalize_value(value) -> str:
    """Normalize values for comparison"""
    if value is None:
        return ""
    
    # Convert to string and normalize
    str_value = str(value).strip().lower()
    
    # Remove common formatting
    str_value = str_value.replace("-", "").replace(" ", "").replace("_", "")
    
    return str_value


def merge_extracted_data_with_user_data(user_data: Dict, extracted_data: Dict, verification_results: Dict) -> Dict:
    """Merge extracted data with user data, prioritizing user input for conflicts"""
    
    merged_data = {}
    
    for field, result in verification_results.items():
        if result["status"] == "missing_user":
            # Add new data from document
            merged_data[field] = result["extracted_value"]
        elif result["status"] == "match":
            # Keep user data (already matches)
            continue
        elif result["status"] == "mismatch":
            # Keep user data but log the discrepancy
            continue
        # For missing_document, we already have user data
    
    return merged_data


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
        
        # DEBUG: Log what we're setting in the state
        logger.info(f"🔍 VALIDATION DEBUG: Setting validation_results = {validation_results}")
        logger.info(f"🔍 VALIDATION DEBUG: Setting processing_status = 'validated'")
        
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
    """Assess eligibility for social support and store ML results"""
    
    try:
        demo_logger.log_step("ELIGIBILITY_ASSESSMENT", "🚀 Starting eligibility assessment with ML models")
        logger.info("Starting eligibility assessment")
        
        # Initialize eligibility agent
        eligibility_agent = EligibilityAssessmentAgent()
        
        # Log user data being assessed
        collected_data = state["collected_data"]
        demo_logger.log_step("ASSESSMENT_INPUT", f"📊 Assessing user: {collected_data.get('name', 'Unknown')} - Income: {collected_data.get('monthly_income', 0)} AED, Family: {collected_data.get('family_size', 1)}")
        
        # Run eligibility assessment
        demo_logger.log_step("ML_PROCESSING", "🤖 Running ML models for eligibility and support amount prediction...")
        eligibility_result = await eligibility_agent.process({
            "application_data": state["collected_data"],
            "assessment_mode": "conversational"
        })
        
        if eligibility_result.get("status") == "success":
            # Extract the decision
            decision = eligibility_result.get("eligibility_result") or eligibility_result.get("assessment_result")
            if not decision:
                decision = {k: v for k, v in eligibility_result.items() if k not in {"status", "agent_name", "application_id", "assessed_at", "assessment_method", "reasoning"}}
            
            # Log ML predictions
            ml_prediction = decision.get("ml_prediction", {})
            if ml_prediction:
                eligible = ml_prediction.get("eligible", False)
                confidence = ml_prediction.get("confidence", 0.0)
                support_amount = ml_prediction.get("support_amount", 0)
                
                demo_logger.log_eligibility_assessment(eligible, support_amount, confidence)
                demo_logger.log_step("ML_FEATURES", f"🔢 Used features: {', '.join(ml_prediction.get('features_used', {}).keys())}")
            
            # Log overall decision
            overall_eligible = decision.get("eligible", False)
            overall_amount = decision.get("support_amount", 0)
            reason = decision.get("reason", "Assessment completed")
            
            demo_logger.log_step("FINAL_DECISION", f"⚖️ Final Decision: {'APPROVED' if overall_eligible else 'DECLINED'} - {overall_amount} AED")
            demo_logger.log_step("DECISION_REASON", f"📝 Reason: {reason}")
            
            state["eligibility_result"] = decision
            state["processing_status"] = "eligibility_assessed"
            
            # CRITICAL: Store ML model predictions in database if we have an application_id
            if state.get("application_id") and decision.get("ml_prediction"):
                try:
                    demo_logger.log_database_operation("ML_STORAGE", "STARTING", "Storing ML predictions")
                    db = SessionLocal()
                    try:
                        ml_prediction = decision.get("ml_prediction", {})
                        
                        # Store eligibility prediction
                        if "eligible" in ml_prediction:
                            eligibility_prediction_data = {
                                "model_version": "v2.1",
                                "prediction_type": "eligibility",
                                "input_features": ml_prediction.get("features_used", {}),
                                "prediction_result": {
                                    "eligible": ml_prediction.get("eligible"),
                                    "probability": ml_prediction.get("confidence", 0.5)
                                },
                                "confidence_score": ml_prediction.get("confidence", 0.5),
                                "processing_time_ms": ml_prediction.get("processing_time_ms", 50.0)
                            }
                            
                            DatabaseManager.store_ml_prediction(
                                db, 
                                state["application_id"], 
                                "eligibility_classifier",
                                eligibility_prediction_data
                            )
                            demo_logger.log_database_operation("ELIGIBILITY_ML", "STORED", f"Confidence: {ml_prediction.get('confidence', 0.5):.3f}")
                        
                        # Store support amount prediction
                        if "support_amount" in ml_prediction:
                            support_prediction_data = {
                                "model_version": "v2.1",
                                "prediction_type": "support_amount",
                                "input_features": ml_prediction.get("features_used", {}),
                                "prediction_result": {
                                    "support_amount": ml_prediction.get("support_amount")
                                },
                                "confidence_score": ml_prediction.get("confidence", 0.5),
                                "processing_time_ms": ml_prediction.get("processing_time_ms", 40.0)
                            }
                            
                            DatabaseManager.store_ml_prediction(
                                db, 
                                state["application_id"], 
                                "support_amount_regressor",
                                support_prediction_data
                            )
                            demo_logger.log_database_operation("SUPPORT_AMOUNT_ML", "STORED", f"Amount: {ml_prediction.get('support_amount', 0)} AED")
                        
                        db.commit()
                        demo_logger.log_database_operation("ML_STORAGE", "SUCCESS", "All ML predictions stored")
                        logger.info("✅ ML predictions stored in database")
                        
                    except Exception as e:
                        demo_logger.log_database_operation("ML_STORAGE", "FAILED", str(e))
                        logger.error(f"❌ Failed to store ML predictions: {str(e)}")
                        db.rollback()
                    finally:
                        db.close()
                        
                except Exception as e:
                    demo_logger.log_database_operation("ML_STORAGE", "ERROR", str(e))
                    logger.error(f"❌ Database connection failed for ML storage: {str(e)}")
            
            demo_logger.log_step("ELIGIBILITY_ASSESSMENT", f"✅ Assessment completed successfully")
            logger.info(f"Eligibility assessment completed. Eligible: {decision.get('eligible', False)}")
            
        else:
            # Generate fallback decision
            demo_logger.log_step("ELIGIBILITY_FALLBACK", "⚠️ ML assessment failed, using rule-based fallback")
            logger.warning("Eligibility assessment failed, using fallback decision")
            state["eligibility_result"] = generate_fallback_eligibility_decision(state["collected_data"])
            state["processing_status"] = "eligibility_fallback"
        
        # Log eligibility assessment
        state["workflow_history"].append({
            "step": "assess_eligibility",
            "eligibility_result": state["eligibility_result"],
            "ml_predictions_stored": state.get("application_id") is not None,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
    except Exception as e:
        error_msg = f"Error in eligibility assessment: {str(e)}"
        demo_logger.log_step("ELIGIBILITY_ASSESSMENT", f"❌ FAILED: {error_msg}", "ERROR")
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        state["processing_status"] = "eligibility_error"
    
    return state


async def generate_recommendations(state: ConversationState) -> ConversationState:
    """Generate economic enablement recommendations"""
    
    try:
        logger.info("🎯 Generating economic enablement recommendations")
        
        # Create conversation agent for LLM recommendations
        conversation_agent = ConversationAgent()
        
        # Verify the agent has the required method
        if not hasattr(conversation_agent, '_generate_llm_economic_recommendations'):
            logger.error("❌ ConversationAgent missing _generate_llm_economic_recommendations method")
            raise AttributeError("ConversationAgent missing _generate_llm_economic_recommendations method")
        
        logger.info(f"🔄 Calling LLM recommendations with data: {len(state.get('collected_data', {}))} fields")
        
        # Generate recommendations using the new LLM method
        recommendations_result = await conversation_agent._generate_llm_economic_recommendations(
            state["collected_data"], 
            state["eligibility_result"]
        )
        
        logger.info(f"📝 LLM recommendations result status: {recommendations_result.get('status')}")
        
        if recommendations_result.get("status") == "success":
            # Add recommendations to eligibility result
            if "economic_enablement" not in state["eligibility_result"]:
                state["eligibility_result"]["economic_enablement"] = {}
            
            state["eligibility_result"]["economic_enablement"]["llm_generated"] = True
            state["eligibility_result"]["economic_enablement"]["recommendations_text"] = recommendations_result["response"]
            
            logger.info("✅ LLM-powered recommendations generated successfully")
        else:
            error_msg = recommendations_result.get("error", "Unknown LLM error")
            logger.warning(f"⚠️ LLM recommendations failed: {error_msg}")
            # Fallback to basic recommendations
            state["eligibility_result"]["economic_enablement"] = {
                "summary": "Basic economic enablement recommendations based on your profile.",
                "recommendations": [
                    "Explore skills development programs",
                    "Consider career advancement opportunities", 
                    "Look into financial literacy resources"
                ],
                "fallback_used": True,
                "llm_error": error_msg
            }
        
        state["processing_status"] = "recommendations_generated"
        
        # Log recommendation generation
        state["workflow_history"].append({
            "step": "generate_recommendations",
            "recommendations_status": recommendations_result.get("status", "fallback"),
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
    except AttributeError as e:
        error_msg = f"AttributeError generating recommendations: {str(e)}"
        logger.error(f"❌ {error_msg}")
        state["error_messages"].append(error_msg)
        state["processing_status"] = "recommendation_error"
        
        # Add fallback recommendations
        state["eligibility_result"]["economic_enablement"] = {
            "summary": "Basic recommendations due to system error.",
            "recommendations": ["Contact support for personalized guidance"],
            "error": error_msg
        }
        
    except Exception as e:
        error_msg = f"Error generating recommendations: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"🔍 Error type: {type(e).__name__}")
        state["error_messages"].append(error_msg)
        state["processing_status"] = "recommendation_error"
        
        # Add fallback recommendations
        state["eligibility_result"]["economic_enablement"] = {
            "summary": "Basic recommendations due to system error.",
            "recommendations": ["Contact support for personalized guidance"],
            "error": error_msg
        }
    
    return state


async def finalize_application(state: ConversationState) -> ConversationState:
    """Finalize the application process and store in database"""
    
    try:
        logger.info("Finalizing application and storing in database")
        
        # Prepare final decision
        final_decision = state.get("eligibility_result", {})
        state["final_decision"] = final_decision
        
        # CRITICAL: Store application in PostgreSQL database
        db = SessionLocal()
        try:
            # Prepare application data for database storage
            collected_data = state.get("collected_data", {})
            
            # Map conversation data to database schema
            application_data = {
                "full_name": collected_data.get("name", "Unknown"),
                "emirates_id": collected_data.get("emirates_id"),
                "phone_number": collected_data.get("phone_number") or collected_data.get("phone"),
                "email": collected_data.get("email"),
                "nationality": collected_data.get("nationality", "UAE"),
                
                # Employment Information
                "employment_status": _map_employment_status(collected_data.get("employment_status")),
                "employer_name": collected_data.get("employer_name"),
                "job_title": collected_data.get("job_title"),
                "monthly_income": float(collected_data.get("monthly_income", 0)),
                "employment_duration_months": collected_data.get("employment_duration_months"),
                
                # Family Information
                "family_size": int(collected_data.get("family_size", 1)),
                "dependents_count": collected_data.get("dependents_count", 0),
                "spouse_employment_status": _map_employment_status(collected_data.get("spouse_employment_status")),
                "spouse_monthly_income": float(collected_data.get("spouse_monthly_income", 0)),
                
                # Housing Information
                "housing_status": _map_housing_status(collected_data.get("housing_status")),
                "monthly_rent": float(collected_data.get("monthly_rent", 0)),
                "housing_allowance": float(collected_data.get("housing_allowance", 0)),
                
                # Financial Information
                "total_assets": float(collected_data.get("total_assets", 0)),
                "total_liabilities": float(collected_data.get("total_liabilities", 0)),
                "monthly_expenses": float(collected_data.get("monthly_expenses", 0)),
                "savings_amount": float(collected_data.get("savings_amount", 0)),
                "credit_score": collected_data.get("credit_score"),
                
                # Application Status and Processing
                "status": ApplicationStatus.COMPLETED,
                "submission_date": datetime.utcnow(),
                "decision_date": datetime.utcnow(),
                
                # Eligibility Results
                "is_eligible": final_decision.get("eligible", False),
                "eligibility_score": final_decision.get("eligibility_score"),
                "recommended_support_amount": final_decision.get("support_amount", 0),
                "eligibility_reason": final_decision.get("reason", "Assessment completed"),
                
                # ML Model Results (if available)
                "ml_eligibility_prediction": final_decision.get("ml_prediction", {}).get("eligible"),
                "ml_support_amount_prediction": final_decision.get("ml_prediction", {}).get("support_amount"),
                "ml_model_confidence": final_decision.get("ml_prediction", {}).get("confidence"),
                "ml_features_used": final_decision.get("ml_prediction", {}).get("features_used"),
                
                # Economic Enablement
                "recommended_programs": final_decision.get("economic_enablement", {}).get("programs", []),
                "enablement_recommendations": final_decision.get("economic_enablement", {}).get("recommendations_text"),
                
                # Audit Trail
                "created_by": "AI_WORKFLOW",
                "updated_by": "AI_WORKFLOW",
                
                # Additional metadata
                "application_metadata": {
                    "workflow_version": "v2.0",
                    "processing_method": "conversational_ai",
                    "completion_time": datetime.utcnow().isoformat(),
                    "total_messages": len(state.get("messages", [])),
                    "documents_processed": len(state.get("uploaded_documents", []))
                },
                "conversation_history": state.get("messages", [])
            }
            
            # Check if application with this Emirates ID already exists
            emirates_id = collected_data.get("emirates_id")
            existing_application = None
            if emirates_id:
                existing_application = DatabaseManager.get_application_by_emirates_id(db, emirates_id)
            
            # TESTING MODE: For testing, create new applications instead of updating
            # This can be controlled by environment variable or configuration
            force_new_application = os.getenv("FORCE_NEW_APPLICATION", "false").lower() == "true"
            
            if existing_application and not force_new_application:
                # Update existing application instead of creating new one
                logger.info(f"Updating existing application for Emirates ID: {emirates_id}")
                
                # Update the existing application with new data
                for key, value in application_data.items():
                    if hasattr(existing_application, key) and key not in ['id', 'application_id', 'created_at', 'reference_number']:
                        setattr(existing_application, key, value)
                
                existing_application.updated_at = datetime.utcnow()
                existing_application.updated_by = "AI_WORKFLOW"
                
                db.commit()
                db.refresh(existing_application)
                
                db_application = existing_application
                logger.info(f"✅ Application updated in database: {db_application.application_id}")
            else:
                # Create new application
                logger.info(f"Creating new application for Emirates ID: {emirates_id}")
                db_application = DatabaseManager.create_application(db, application_data)
                logger.info(f"✅ Application created in database: {db_application.application_id}")
            
            # Store documents if any were uploaded
            uploaded_documents = state.get("uploaded_documents", [])
            processed_documents = state.get("processed_documents", [])
            
            for doc_path in uploaded_documents:
                try:
                    # Determine document type
                    doc_type = _determine_document_type_enum(doc_path)
                    
                    # Create document record
                    document_data = {
                        "document_id": f"DOC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(doc_path)}",
                        "filename": os.path.basename(doc_path),
                        "original_filename": os.path.basename(doc_path),
                        "document_type": doc_type,
                        "file_path": doc_path,
                        "file_size": os.path.getsize(doc_path) if os.path.exists(doc_path) else 0,
                        "mime_type": _get_mime_type(doc_path),
                        "is_processed": doc_path in processed_documents,
                        "processing_status": "processed" if doc_path in processed_documents else "uploaded",
                        "uploaded_by": "AI_WORKFLOW"
                    }
                    
                    document = Document(
                        application_id=db_application.id,
                        **document_data
                    )
                    db.add(document)
                    
                except Exception as e:
                    logger.error(f"Error storing document {doc_path}: {str(e)}")
            
            # Create application review record
            review_data = {
                "reviewer_id": "AI_SYSTEM",
                "reviewer_name": "AI Assessment System",
                "review_type": "automated",
                "decision": "approved" if final_decision.get("eligible", False) else "rejected",
                "decision_reason": final_decision.get("reason", "Automated AI assessment completed"),
                "support_amount_recommended": final_decision.get("support_amount", 0),
                "conditions": final_decision.get("conditions", {}),
                "review_notes": f"Automated assessment completed via conversational AI workflow. Total conversation messages: {len(state.get('messages', []))}",
                "risk_assessment": final_decision.get("risk_assessment", {"overall_risk_score": 0.5}),
                "compliance_check": {
                    "document_verification": "completed" if uploaded_documents else "not_required",
                    "identity_verification": "completed" if collected_data.get("emirates_id") else "pending",
                    "income_verification": "completed" if collected_data.get("monthly_income") else "pending",
                    "eligibility_criteria": "passed" if final_decision.get("eligible", False) else "failed"
                }
            }
            
            review = ApplicationReview(
                application_id=db_application.id,
                **review_data
            )
            db.add(review)
            
            # Commit all database changes
            db.commit()
            
            # Update state with database application ID
            state["application_id"] = db_application.application_id
            state["database_stored"] = True
            
            logger.info(f"✅ Application stored in database: {db_application.application_id}")
            logger.info(f"   - Documents: {len(uploaded_documents)} stored")
            logger.info(f"   - Eligibility: {final_decision.get('eligible', False)}")
            logger.info(f"   - Support Amount: {final_decision.get('support_amount', 0)} AED")
            
        except Exception as e:
            logger.error(f"❌ Database storage failed: {str(e)}")
            db.rollback()
            # Continue with workflow even if database storage fails
            state["database_error"] = str(e)
        finally:
            db.close()
        
        # Generate final summary message using conversation agent
        conversation_agent = ConversationAgent()
        final_response = conversation_agent._generate_eligibility_response(final_decision)
        
        # Add database confirmation to response if successful
        if state.get("database_stored"):
            # Get the stored application to access reference number
            db = SessionLocal()
            try:
                db_app = DatabaseManager.get_application_by_id(db, state["application_id"])
                if db_app:
                    final_response += f"\n\n📋 **Your Application References:**"
                    final_response += f"\n🔢 **Reference Number:** {db_app.reference_number} (Easy to remember!)"
                    final_response += f"\n📱 **Quick Lookup:** Use your name + last 4 digits of phone ({db_app.phone_reference or 'Not provided'})"
                    final_response += f"\n🆔 **Full Application ID:** {state['application_id']}"
                    final_response += f"\n\n💡 **How to Check Status Later:**"
                    final_response += f"\n• Use Reference Number: {db_app.reference_number}"
                    final_response += f"\n• Use your Emirates ID: {db_app.emirates_id}"
                    final_response += f"\n• Use Name + Phone: {db_app.full_name} + {db_app.phone_reference or 'Not provided'}"
                    final_response += f"\n\n📞 **Save this message** or take a screenshot for your records!"
                else:
                    final_response += f"\n\n📋 Your application has been saved with ID: {state['application_id']}"
            finally:
                db.close()
        
        state["messages"].append({
            "role": "assistant",
            "content": final_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # CRITICAL FIX: Set last_agent_response so frontend gets the final message
        state["last_agent_response"] = final_response
        
        state["processing_status"] = "completed"
        state["current_step"] = ConversationStep.COMPLETION
        
        # Log finalization
        state["workflow_history"].append({
            "step": "finalize_application",
            "final_decision": final_decision,
            "database_stored": state.get("database_stored", False),
            "application_id": state.get("application_id"),
            "completion_time": datetime.now().isoformat(),
            "status": "completed"
        })
        
        logger.info(f"Application finalized. Application ID: {state.get('application_id', 'N/A')}")
        
    except Exception as e:
        error_msg = f"Error finalizing application: {str(e)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        state["processing_status"] = "finalization_error"
    
    return state


def _map_employment_status(status_str: str) -> EmploymentStatus:
    """Map string employment status to enum"""
    if not status_str:
        return EmploymentStatus.UNEMPLOYED
    
    status_lower = status_str.lower()
    if "employ" in status_lower and "un" not in status_lower:
        return EmploymentStatus.EMPLOYED
    elif "self" in status_lower:
        return EmploymentStatus.SELF_EMPLOYED
    elif "retire" in status_lower:
        return EmploymentStatus.RETIRED
    elif "student" in status_lower:
        return EmploymentStatus.STUDENT
    else:
        return EmploymentStatus.UNEMPLOYED


def _map_housing_status(status_str: str) -> HousingStatus:
    """Map string housing status to enum"""
    if not status_str:
        return HousingStatus.RENTED
    
    status_lower = status_str.lower()
    if "own" in status_lower:
        return HousingStatus.OWNED
    elif "family" in status_lower:
        return HousingStatus.FAMILY_OWNED
    elif "government" in status_lower:
        return HousingStatus.GOVERNMENT_HOUSING
    else:
        return HousingStatus.RENTED


def _determine_document_type_enum(file_path: str) -> DocumentType:
    """Determine document type enum from file path"""
    filename_lower = os.path.basename(file_path).lower()
    
    if "emirates" in filename_lower or "id" in filename_lower:
        return DocumentType.EMIRATES_ID
    elif "bank" in filename_lower or "statement" in filename_lower:
        return DocumentType.BANK_STATEMENT
    elif "salary" in filename_lower or "certificate" in filename_lower:
        return DocumentType.SALARY_CERTIFICATE
    elif "resume" in filename_lower or "cv" in filename_lower:
        return DocumentType.RESUME
    elif "asset" in filename_lower or "liabilit" in filename_lower:
        return DocumentType.ASSETS_LIABILITIES
    elif "credit" in filename_lower or "report" in filename_lower:
        return DocumentType.CREDIT_REPORT
    elif "family" in filename_lower:
        return DocumentType.FAMILY_CERTIFICATE
    elif "housing" in filename_lower or "contract" in filename_lower:
        return DocumentType.HOUSING_CONTRACT
    else:
        return DocumentType.BANK_STATEMENT  # Default


def _get_mime_type(file_path: str) -> str:
    """Get MIME type from file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.pdf': 'application/pdf',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xls': 'application/vnd.ms-excel',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }
    return mime_types.get(ext, 'application/octet-stream')


async def handle_completion_chat(state: ConversationState) -> ConversationState:
    """Handle post-completion conversation"""
    
    try:
        # CRITICAL FIX: Don't process the user's last input if we just completed the application
        # Check if this is the first time entering completion phase
        workflow_history = state.get("workflow_history", [])
        finalization_steps = [step for step in workflow_history if step.get("step") == "finalize_application"]
        
        if finalization_steps:
            # We just completed finalization - the final response should already be in messages
            # Don't process any user input as completion chat yet
            last_message = state.get("messages", [])[-1] if state.get("messages") else None
            if last_message and last_message.get("role") == "assistant":
                # The finalization response is already there, just preserve it
                state["last_agent_response"] = last_message.get("content", "")
                state["processing_status"] = "completion_chat"
                state["user_input"] = None  # Clear any user input
                return state
        
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
    """Determine the next action based on current state"""
    
    current_step = state.get("current_step", "")
    processing_status = state.get("processing_status", "")
    uploaded_docs = len(state.get("uploaded_documents", []))
    processed_docs = len(state.get("processed_documents", []))
    workflow_history = state.get("workflow_history", [])
    
    logger.info(f"🔍 ROUTING DEBUG: current_step='{current_step}', processing_status='{processing_status}', uploaded={uploaded_docs}, processed={processed_docs}")
    
    # CRITICAL LOOP PREVENTION: Check workflow history to prevent infinite document processing
    recent_steps = [entry.get("step", "") for entry in workflow_history[-5:]]
    document_processing_count = recent_steps.count("process_documents")
    validation_count = recent_steps.count("validate_information")
    
    if document_processing_count >= 2 and validation_count >= 1:
        logger.warning(f"🚨 LOOP PREVENTION: Detected repeated document processing ({document_processing_count}x) and validation ({validation_count}x), forcing progression to eligibility assessment")
        return "assess_eligibility"
    
    # PRIORITY 1: Handle document processing flow
    if uploaded_docs > processed_docs or processing_status == "documents_need_processing":
        logger.info(f"🔍 DOCUMENT PROCESSING TRIGGERED: {uploaded_docs} uploaded > {processed_docs} processed OR status = {processing_status}")
        return "process_documents"
    
    # PRIORITY 2: Handle validation flow - CRITICAL FIX
    if processing_status == "ready_for_validation":
        logger.info(f"🔍 ROUTING: Processing status is 'ready_for_validation', routing to validate_info")
        return "validate_info"
    elif processing_status == "validated":
        logger.info("🔍 ROUTING: Processing status is 'validated', routing to assess_eligibility")
        return "assess_eligibility"
    
    # PRIORITY 3: Check validation results
    validation_results = state.get("validation_results", {})
    if validation_results:
        if validation_results.get("status") == "complete":
            logger.info("🔍 ROUTING: Validation complete, routing to assess_eligibility")
            return "assess_eligibility"
    
    # CRITICAL FIX: Handle DOCUMENT_COLLECTION step properly
    # Don't automatically proceed to validation - wait for user decision
    if current_step == ConversationStep.DOCUMENT_COLLECTION:
        # Only proceed if user explicitly indicated they want to proceed (via processing_status)
        if processing_status == "ready_for_assessment":
            logger.info("🔍 ROUTING: User ready for assessment from document collection, routing to validate_info")
            return "validate_info"
        else:
            # Stay in conversation to let user decide what to do
            user_input = state.get("user_input")
            if user_input:
                logger.info("🔍 ROUTING: In document collection with user input, continuing conversation")
                return "continue_conversation"
            else:
                logger.info("🔍 ROUTING: In document collection waiting for user input")
                return "waiting"
    
    # PRIORITY 4: Handle other steps that need validation
    elif current_step == ConversationStep.ELIGIBILITY_PROCESSING and has_minimum_required_data(state.get("collected_data", {})):
        # If we have minimum data but no validation results, validate first
        if not validation_results:
            logger.info("🔍 ROUTING: Have minimum data but no validation results, routing to validate_info")
            return "validate_info"
    
    # PRIORITY 5: Check if ready for eligibility assessment
    if current_step == ConversationStep.ELIGIBILITY_PROCESSING:
        if not state.get("eligibility_result"):
            logger.info("🔍 ROUTING: In eligibility_processing step without result, routing to assess_eligibility")
            return "assess_eligibility"
        else:
            return "finalize"
    
    # PRIORITY 6: If we have user input to process, continue conversation
    user_input = state.get("user_input")
    if user_input:
        logger.info(f"🔍 ROUTING: Have user input '{user_input[:50]}...', routing to continue_conversation")
        return "continue_conversation"
    
    # SAFETY: If no clear next action, end workflow to prevent infinite loops
    logger.info("🔍 ROUTING: No clear next action determined, ending workflow to prevent loops")
    return "waiting"


def after_validation(state: ConversationState) -> str:
    """Determine action after validation"""
    validation_results = state.get("validation_results", {})
    
    # DEBUG: Log the validation results and decision
    logger.info(f"🔍 AFTER_VALIDATION DEBUG: validation_results = {validation_results}")
    
    status = validation_results.get("status")
    inconsistencies = validation_results.get("inconsistencies", [])
    
    logger.info(f"🔍 AFTER_VALIDATION DEBUG: status = '{status}', inconsistencies = {inconsistencies}")
    
    # CRITICAL FIX: If validation results are empty but we have minimum data, proceed to eligibility
    if not validation_results and has_minimum_required_data(state.get("collected_data", {})):
        logger.warning(f"🔍 AFTER_VALIDATION DEBUG: validation_results empty but have minimum data, proceeding to assess_eligibility")
        return "assess_eligibility"
    
    if status == "complete" and not inconsistencies:
        logger.info(f"🔍 AFTER_VALIDATION DEBUG: Routing to assess_eligibility")
        return "assess_eligibility"
    else:
        logger.info(f"🔍 AFTER_VALIDATION DEBUG: Routing to continue_conversation")
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
    
    # Check for bank-related keywords first (before emirates check)
    # This prevents "Emirates NBD" bank statements from being classified as Emirates ID
    if any(word in filename for word in ['bank', 'statement', 'transaction', 'nbd', 'account']):
        return "bank_statement"
    elif any(word in filename for word in ['emirates', 'id', 'identity']) and 'nbd' not in filename:
        # Only classify as emirates_id if it contains emirates/id keywords but NOT bank-related terms
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