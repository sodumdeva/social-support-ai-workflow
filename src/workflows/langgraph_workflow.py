"""
LangGraph Workflow for Social Support AI System

Implements state-based conversation workflow using LangGraph for managing
the conversational application process with proper state transitions.
"""
#try:
from langgraph.graph import StateGraph, END
#except ImportError:
    # Fallback for different langgraph versions
    # try:
    #from langgraph import StateGraph, END
    # except ImportError:
    #     # Create mock classes if langgraph is not available
    #     class StateGraph:
    #         def __init__(self, state_type):
    #             self.state_type = state_type
    #         def add_node(self, name, func):
    #             pass
    #         def add_conditional_edges(self, source, condition, mapping):
    #             pass
    #         def add_edge(self, source, target):
    #             pass
    #         def set_entry_point(self, node):
    #             pass
    #         def compile(self):
    #             return MockWorkflow()
        
    #     class MockWorkflow:
    #         async def ainvoke(self, state):
    #             return state
        
    #     END = "END"

from typing import TypedDict, List, Dict, Optional, Any
import asyncio
from datetime import datetime
import json

# Import agents with correct paths
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.agents.conversation_agent import ConversationAgent, ConversationStep
from src.agents.data_extraction_agent import DataExtractionAgent
from src.agents.eligibility_agent import EligibilityAssessmentAgent


class ConversationState(TypedDict):
    """State structure for the conversation workflow"""
    messages: List[Dict]
    collected_data: Dict
    current_step: str
    eligibility_result: Optional[Dict]
    final_decision: Optional[Dict]
    uploaded_documents: List[str]
    workflow_history: List[Dict]
    application_id: Optional[str]
    processing_status: str
    error_messages: List[str]


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
    
    # Define conditional routing
    workflow.add_conditional_edges(
        "initialize_conversation",
        should_continue_conversation,
        {
            "continue": "handle_user_message",
            "end": END
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
            "finalize": "finalize_application"
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
    workflow.add_edge("finalize_application", END)
    
    # Set entry point
    workflow.set_entry_point("initialize_conversation")
    
    return workflow.compile()


async def initialize_conversation(state: ConversationState) -> ConversationState:
    """Initialize the conversation workflow"""
    
    # Generate application ID if not exists
    if not state.get("application_id"):
        state["application_id"] = f"APP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Initialize conversation if empty
    if not state.get("messages"):
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
    state["processing_status"] = "initialized"
    state["error_messages"] = state.get("error_messages", [])
    
    # Log workflow initialization
    state["workflow_history"].append({
        "step": "initialize_conversation",
        "timestamp": datetime.now().isoformat(),
        "status": "completed"
    })
    
    return state


async def handle_user_message(state: ConversationState) -> ConversationState:
    """Handle user message using conversation agent"""
    
    try:
        # Get the latest user message
        user_messages = [msg for msg in state["messages"] if msg["role"] == "user"]
        if not user_messages:
            state["processing_status"] = "waiting_for_input"
            return state
        
        latest_message = user_messages[-1]["content"]
        
        # Initialize conversation agent
        conversation_agent = ConversationAgent()
        
        # Process the message
        conversation_state = {
            "current_step": state["current_step"],
            "collected_data": state["collected_data"],
            "uploaded_documents": state["uploaded_documents"]
        }
        
        response = await conversation_agent.process_message(
            latest_message,
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
        
        # Update conversation state
        if "state_update" in response:
            state.update(response["state_update"])
        
        # Check if application is complete
        if response.get("application_complete"):
            state["final_decision"] = response.get("final_decision")
            state["processing_status"] = "completed"
        else:
            state["processing_status"] = "in_progress"
        
        # Log the interaction
        state["workflow_history"].append({
            "step": "handle_user_message",
            "user_message": latest_message,
            "assistant_response": response.get("message", ""),
            "current_step": state["current_step"],
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
    except Exception as e:
        error_msg = f"Error handling user message: {str(e)}"
        state["error_messages"].append(error_msg)
        state["messages"].append({
            "role": "assistant", 
            "content": "I apologize, I encountered an error processing your message. Could you please try again?",
            "timestamp": datetime.now().isoformat()
        })
        state["processing_status"] = "error"
    
    return state


async def process_documents(state: ConversationState) -> ConversationState:
    """Process uploaded documents"""
    
    try:
        if not state["uploaded_documents"]:
            state["processing_status"] = "no_documents"
            return state
        
        # Initialize document processing agent
        extraction_agent = DataExtractionAgent()
        
        # Process each document
        all_extraction_results = {}
        
        for doc_path in state["uploaded_documents"]:
            if doc_path not in state.get("processed_documents", []):
                # Determine document type from filename
                doc_type = determine_document_type(doc_path)
                
                # Process document
                extraction_result = await extraction_agent.process({
                    "documents": [{"path": doc_path, "type": doc_type}],
                    "extraction_mode": "conversational"
                })
                
                if extraction_result["status"] == "success":
                    extracted_data = extraction_result["extraction_results"].get(doc_type, {})
                    all_extraction_results[doc_type] = extracted_data
                    
                    # Update collected_data with extracted information
                    state["collected_data"].update(extracted_data)
                    
                    # Mark document as processed
                    if "processed_documents" not in state:
                        state["processed_documents"] = []
                    state["processed_documents"].append(doc_path)
        
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
        
    except Exception as e:
        error_msg = f"Error processing documents: {str(e)}"
        state["error_messages"].append(error_msg)
        state["processing_status"] = "document_error"
    
    return state


async def validate_information(state: ConversationState) -> ConversationState:
    """Validate collected information for consistency"""
    
    try:
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
        
    except Exception as e:
        error_msg = f"Error during validation: {str(e)}"
        state["error_messages"].append(error_msg)
        state["processing_status"] = "validation_error"
    
    return state


async def assess_eligibility(state: ConversationState) -> ConversationState:
    """Assess eligibility for social support"""
    
    try:
        # Initialize eligibility agent
        eligibility_agent = EligibilityAssessmentAgent()
        
        # Perform eligibility assessment
        assessment_result = await eligibility_agent.process({
            "application_data": state["collected_data"],
            "validation_results": state.get("validation_results", {}),
            "assessment_mode": "conversational"
        })
        
        if assessment_result["status"] == "success":
            eligibility_result = assessment_result["eligibility_result"]
            state["eligibility_result"] = eligibility_result
            
            # Generate response message
            if eligibility_result.get("eligible"):
                support_amount = eligibility_result.get("support_amount", 0)
                response_msg = f"🎉 Great news! You are eligible for {support_amount:,.0f} AED per month in social support."
            else:
                reason = eligibility_result.get("reason", "You don't meet the current eligibility criteria.")
                response_msg = f"Based on the assessment, {reason}"
            
            state["messages"].append({
                "role": "assistant",
                "content": response_msg,
                "timestamp": datetime.now().isoformat()
            })
            
        else:
            # Fallback assessment
            state["eligibility_result"] = generate_fallback_eligibility_decision(state["collected_data"])
            state["messages"].append({
                "role": "assistant",
                "content": "I've completed a basic eligibility assessment. Let me provide you with the results.",
                "timestamp": datetime.now().isoformat()
            })
        
        state["processing_status"] = "eligibility_assessed"
        
        # Log eligibility assessment
        state["workflow_history"].append({
            "step": "assess_eligibility",
            "eligibility_result": state["eligibility_result"],
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
    except Exception as e:
        error_msg = f"Error during eligibility assessment: {str(e)}"
        state["error_messages"].append(error_msg)
        state["processing_status"] = "eligibility_error"
        
        # Generate fallback decision
        state["eligibility_result"] = generate_fallback_eligibility_decision(state["collected_data"])
    
    return state


async def generate_recommendations(state: ConversationState) -> ConversationState:
    """Generate economic enablement recommendations"""
    
    try:
        collected_data = state["collected_data"]
        eligibility_result = state.get("eligibility_result", {})
        
        # Generate basic recommendations based on profile
        recommendations = generate_economic_enablement_recommendations(
            collected_data, 
            eligibility_result
        )
        
        # Add recommendations to eligibility result
        if "eligibility_result" not in state:
            state["eligibility_result"] = {}
        state["eligibility_result"]["recommendations"] = recommendations
        
        # Generate response message
        rec_msg = "Here are some recommendations to help improve your situation:\n\n"
        
        if recommendations.get("job_opportunities"):
            rec_msg += "💼 **Job Opportunities:**\n"
            for job in recommendations["job_opportunities"][:3]:
                rec_msg += f"• {job.get('title', 'Job Opportunity')}\n"
            rec_msg += "\n"
        
        if recommendations.get("training_programs"):
            rec_msg += "📚 **Training Programs:**\n"
            for program in recommendations["training_programs"][:3]:
                rec_msg += f"• {program.get('name', 'Training Program')}\n"
            rec_msg += "\n"
        
        rec_msg += "These recommendations are based on your profile and can help improve your long-term financial situation."
        
        state["messages"].append({
            "role": "assistant",
            "content": rec_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        state["processing_status"] = "recommendations_generated"
        
        # Log recommendation generation
        state["workflow_history"].append({
            "step": "generate_recommendations",
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
        
    except Exception as e:
        error_msg = f"Error generating recommendations: {str(e)}"
        state["error_messages"].append(error_msg)
        state["processing_status"] = "recommendation_error"
    
    return state


async def finalize_application(state: ConversationState) -> ConversationState:
    """Finalize the application process"""
    
    try:
        # Prepare final decision
        final_decision = state.get("eligibility_result", {})
        state["final_decision"] = final_decision
        
        # Generate final summary message
        application_id = state.get("application_id", "UNKNOWN")
        
        final_msg = f"🎯 **Application Summary** (ID: {application_id})\n\n"
        
        if final_decision.get("eligible"):
            final_msg += "✅ **Status:** APPROVED\n"
            support_amount = final_decision.get("support_amount", 0)
            final_msg += f"💰 **Monthly Support:** {support_amount:,.0f} AED\n\n"
        else:
            final_msg += "❌ **Status:** Not approved for direct financial support\n\n"
        
        final_msg += "📋 **Next Steps:**\n"
        final_msg += "• You will receive an official notification within 24 hours\n"
        final_msg += "• If approved, support will begin next month\n"
        final_msg += "• You can track your application status online\n\n"
        final_msg += "Thank you for using the Social Support AI system!"
        
        state["messages"].append({
            "role": "assistant",
            "content": final_msg,
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
        
    except Exception as e:
        error_msg = f"Error finalizing application: {str(e)}"
        state["error_messages"].append(error_msg)
        state["processing_status"] = "finalization_error"
    
    return state


# Conditional routing functions
def should_continue_conversation(state: ConversationState) -> str:
    """Determine if conversation should continue"""
    if state.get("processing_status") == "completed":
        return "end"
    return "continue"


def determine_next_action(state: ConversationState) -> str:
    """Determine next action based on conversation state"""
    
    current_step = state.get("current_step", "")
    processing_status = state.get("processing_status", "")
    
    # Check if we have new documents to process
    if (state.get("uploaded_documents") and 
        len(state.get("uploaded_documents", [])) > len(state.get("processed_documents", []))):
        return "process_documents"
    
    # Check if we need to validate information
    if current_step in [ConversationStep.DOCUMENT_COLLECTION, ConversationStep.ELIGIBILITY_PROCESSING]:
        if not state.get("validation_results"):
            return "validate_info"
        elif state.get("validation_results", {}).get("status") == "complete":
            return "assess_eligibility"
    
    # Check if ready for eligibility assessment
    if current_step == ConversationStep.ELIGIBILITY_PROCESSING:
        if not state.get("eligibility_result"):
            return "assess_eligibility"
        else:
            return "finalize"
    
    # Check if application is complete
    if processing_status == "completed" or current_step == ConversationStep.COMPLETION:
        return "finalize"
    
    # Continue conversation by default
    return "continue_conversation"


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
        if not state.get("eligibility_result", {}).get("recommendations"):
            return "generate_recommendations"
        else:
            return "finalize"
    return "finalize"


# Helper functions
def determine_document_type(file_path: str) -> str:
    """Determine document type from file path"""
    filename = file_path.lower()
    
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
    """Perform basic data consistency checks"""
    
    inconsistencies = []
    
    # Check income vs employment status
    employment_status = collected_data.get("employment_status", "")
    monthly_income = collected_data.get("monthly_income", 0)
    
    if employment_status == "unemployed" and monthly_income > 2000:
        inconsistencies.append("High income reported for unemployed status")
    
    if employment_status == "employed" and monthly_income == 0:
        inconsistencies.append("Zero income reported for employed status")
    
    # Check family size reasonableness
    family_size = collected_data.get("family_size", 1)
    if family_size > 15:
        inconsistencies.append("Unusually large family size")
    
    return {
        "inconsistencies": inconsistencies,
        "total_checks": 3,
        "passed_checks": 3 - len(inconsistencies)
    }


def has_minimum_required_data(collected_data: Dict) -> bool:
    """Check if minimum required data is available"""
    required_fields = ["name", "employment_status", "monthly_income", "family_size"]
    return all(field in collected_data for field in required_fields)


def generate_fallback_eligibility_decision(collected_data: Dict) -> Dict:
    """Generate fallback eligibility decision"""
    
    monthly_income = collected_data.get("monthly_income", 0)
    family_size = collected_data.get("family_size", 1)
    
    # Simple threshold-based decision
    income_threshold = 3000 * family_size
    
    if monthly_income < income_threshold:
        support_amount = max(500, (income_threshold - monthly_income) * 0.5)
        return {
            "eligible": True,
            "decision": "approved",
            "support_amount": support_amount,
            "breakdown": {
                "Base Support": 500,
                "Family Size Supplement": (family_size - 1) * 200,
                "Income Gap Support": max(0, support_amount - 500 - ((family_size - 1) * 200))
            },
            "reason": "Approved based on income threshold assessment"
        }
    else:
        return {
            "eligible": False,
            "decision": "declined", 
            "support_amount": 0,
            "reason": "Monthly income exceeds the eligibility threshold"
        }


def generate_economic_enablement_recommendations(collected_data: Dict, eligibility_result: Dict) -> Dict:
    """Generate economic enablement recommendations"""
    
    employment_status = collected_data.get("employment_status", "unemployed")
    monthly_income = collected_data.get("monthly_income", 0)
    
    recommendations = {
        "job_opportunities": [],
        "training_programs": [],
        "business_support": [],
        "financial_literacy": []
    }
    
    # Job opportunities based on employment status
    if employment_status == "unemployed":
        recommendations["job_opportunities"] = [
            {"title": "Customer Service Representative", "company": "Various Companies", "salary_range": "3000-5000 AED"},
            {"title": "Administrative Assistant", "company": "Government Offices", "salary_range": "4000-6000 AED"},
            {"title": "Retail Sales Associate", "company": "Shopping Centers", "salary_range": "2500-4000 AED"}
        ]
        
        recommendations["training_programs"] = [
            {"name": "Digital Literacy Program", "duration": "4 weeks", "cost": "Free"},
            {"name": "English Language Course", "duration": "8 weeks", "cost": "Subsidized"},
            {"name": "Customer Service Skills", "duration": "2 weeks", "cost": "Free"}
        ]
    
    elif employment_status == "employed" and monthly_income < 5000:
        recommendations["training_programs"] = [
            {"name": "Professional Development Course", "duration": "6 weeks", "cost": "Subsidized"},
            {"name": "Management Skills Training", "duration": "4 weeks", "cost": "Free"},
            {"name": "Technical Certification Program", "duration": "12 weeks", "cost": "Subsidized"}
        ]
    
    # Business support for self-employed
    if employment_status == "self_employed":
        recommendations["business_support"] = [
            {"name": "Small Business Development Program", "type": "Mentorship", "duration": "6 months"},
            {"name": "Micro-financing Options", "type": "Financial Support", "amount": "Up to 50,000 AED"},
            {"name": "Business Registration Assistance", "type": "Legal Support", "cost": "Free"}
        ]
    
    # Financial literacy for everyone
    recommendations["financial_literacy"] = [
        {"name": "Budget Management Workshop", "duration": "1 day", "cost": "Free"},
        {"name": "Savings and Investment Basics", "duration": "2 days", "cost": "Free"},
        {"name": "Family Financial Planning", "duration": "3 hours", "cost": "Free"}
    ]
    
    return recommendations 