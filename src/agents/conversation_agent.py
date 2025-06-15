"""
Conversation Agent for Social Support AI Workflow

Manages conversational flow for application submission through natural language interaction.
Orchestrates the collection of applicant information, document processing, and eligibility assessment.
"""
from typing import List, Dict, Optional, Any
import asyncio
import json
from datetime import datetime
import re
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import logging configuration
from src.utils.logging_config import get_logger

# Setup logging
logger = get_logger("conversation_agent")

from .base_agent import BaseAgent
from .data_extraction_agent import DataExtractionAgent
from .eligibility_agent import EligibilityAssessmentAgent


class ConversationStep:
    """Enumeration of conversation steps"""
    GREETING = "greeting"
    NAME_COLLECTION = "name_collection"
    IDENTITY_VERIFICATION = "identity_verification"
    EMPLOYMENT_INQUIRY = "employment_inquiry"
    INCOME_ASSESSMENT = "income_assessment"
    FAMILY_DETAILS = "family_details"
    HOUSING_SITUATION = "housing_situation"
    DOCUMENT_COLLECTION = "document_collection"
    ELIGIBILITY_PROCESSING = "eligibility_processing"
    RECOMMENDATIONS = "recommendations"
    COMPLETION = "completion"


class ConversationAgent(BaseAgent):
    """
    Conversational Agent for Social Support Application Processing
    
    Manages conversation flow through defined steps (name, income, documents, etc.),
    processes user input with local LLM integration, and coordinates document upload
    and eligibility assessment workflows.
    """
    
    def __init__(self):
        super().__init__("ConversationAgent")
        
        # Initialize supporting agents
        self.data_extraction_agent = DataExtractionAgent()
        self.eligibility_agent = EligibilityAssessmentAgent()
        
        # Conversation flow configuration
        self.required_fields = [
            "name", "emirates_id", "employment_status", 
            "monthly_income", "family_size", "housing_status"
        ]
        
        # Step progression mapping
        self.step_progression = {
            ConversationStep.GREETING: ConversationStep.NAME_COLLECTION,
            ConversationStep.NAME_COLLECTION: ConversationStep.IDENTITY_VERIFICATION,
            ConversationStep.IDENTITY_VERIFICATION: ConversationStep.EMPLOYMENT_INQUIRY,
            ConversationStep.EMPLOYMENT_INQUIRY: ConversationStep.INCOME_ASSESSMENT,
            ConversationStep.INCOME_ASSESSMENT: ConversationStep.FAMILY_DETAILS,
            ConversationStep.FAMILY_DETAILS: ConversationStep.HOUSING_SITUATION,
            ConversationStep.HOUSING_SITUATION: ConversationStep.DOCUMENT_COLLECTION,
            ConversationStep.DOCUMENT_COLLECTION: ConversationStep.ELIGIBILITY_PROCESSING,
            ConversationStep.ELIGIBILITY_PROCESSING: ConversationStep.RECOMMENDATIONS,
            ConversationStep.RECOMMENDATIONS: ConversationStep.COMPLETION
        }
    
    async def process_message(
        self, 
        user_message: str, 
        conversation_history: List[Dict],
        conversation_state: Dict
    ) -> Dict[str, Any]:
        """Process user message and return appropriate response"""
        
        current_step = conversation_state.get("current_step", ConversationStep.GREETING)
        collected_data = conversation_state.get("collected_data", {})
        
        logger.debug(f"Processing message: '{user_message}' at step: {current_step}")
        logger.debug(f"Collected data so far: {collected_data}")
        
        # First, check if user wants to make corrections or go back
        correction_response = await self._handle_corrections_and_navigation(
            user_message, conversation_state
        )
        if correction_response:
            return correction_response
        
        # Route to appropriate handler based on current step
        if current_step == ConversationStep.GREETING:
            # For greeting step, directly process as name collection
            logger.debug("Processing GREETING step as name collection")
            return await self._handle_name_collection(user_message, collected_data)
        
        elif current_step == ConversationStep.NAME_COLLECTION:
            logger.debug("Handling NAME_COLLECTION step")
            return await self._handle_name_collection(user_message, collected_data)
        
        elif current_step == ConversationStep.IDENTITY_VERIFICATION:
            logger.debug("Handling IDENTITY_VERIFICATION step")
            return await self._handle_identity_verification(user_message, collected_data)
        
        elif current_step == ConversationStep.EMPLOYMENT_INQUIRY:
            logger.debug("Handling EMPLOYMENT_INQUIRY step")
            return await self._handle_employment_inquiry(user_message, collected_data)
        
        elif current_step == ConversationStep.INCOME_ASSESSMENT:
            logger.debug("Handling INCOME_ASSESSMENT step")
            return await self._handle_income_assessment(user_message, collected_data)
        
        elif current_step == ConversationStep.FAMILY_DETAILS:
            logger.debug("Handling FAMILY_DETAILS step")
            return await self._handle_family_details(user_message, collected_data)
        
        elif current_step == ConversationStep.HOUSING_SITUATION:
            logger.debug("Handling HOUSING_SITUATION step")
            return await self._handle_housing_situation(user_message, collected_data)
        
        elif current_step == ConversationStep.DOCUMENT_COLLECTION:
            logger.debug("Handling DOCUMENT_COLLECTION step")
            return await self._handle_document_collection(user_message, collected_data)
        
        elif current_step == ConversationStep.COMPLETION:
            logger.debug("Handling COMPLETION step")
            return await self._handle_completion_conversation(user_message, conversation_state)
        
        else:
            logger.debug(f"Handling unknown step: {current_step}, falling back to general inquiry")
            return await self._handle_general_inquiry(user_message, current_step, collected_data)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of abstract process method from BaseAgent
        
        Args:
            input_data: Dictionary containing conversation data
                - message: User message
                - conversation_history: List of previous messages
                - conversation_state: Current conversation state
                
        Returns:
            Dictionary containing response and state updates
        """
        try:
            user_message = input_data.get("message", "")
            conversation_history = input_data.get("conversation_history", [])
            conversation_state = input_data.get("conversation_state", {})
            
            # Process the message using the existing method
            result = await self.process_message(user_message, conversation_history, conversation_state)
            
            return {
                "status": "success",
                "agent_name": self.agent_name,
                **result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "agent_name": self.agent_name,
                "error": str(e),
                "message": "I apologize, I encountered an error processing your message. Could you please try again?"
            }
    
    async def process_document_upload(
        self, 
        file_path: str, 
        file_type: str,
        conversation_state: Dict
    ) -> Dict[str, Any]:
        """Process uploaded document in real-time during conversation"""
        
        try:
            # Validate file path exists
            if not file_path or not os.path.exists(file_path):
                return {
                    "message": f"I couldn't find the uploaded file. Please try uploading your {file_type.replace('_', ' ')} again, or provide the information manually.",
                    "state_update": {},
                    "extraction_data": {}
                }
            
            # Extract data from document using specialized agent
            extraction_result = await self.data_extraction_agent.process({
                "documents": [{"file_path": file_path, "document_type": file_type}],
                "extraction_mode": "conversational"
            })
            
            if extraction_result.get("status") == "success" and extraction_result.get("successful_extractions", 0) > 0:
                # Get the extracted data for this document type
                extracted_data = extraction_result["extraction_results"].get(file_type, {})
                
                if extracted_data.get("status") == "success":
                    structured_data = extracted_data.get("structured_data", {})
                    
                    # Update conversation state with extracted data
                    collected_data = conversation_state.get("collected_data", {})
                    
                    # Merge structured data into collected data
                    if isinstance(structured_data, dict):
                        collected_data.update(structured_data)
                    
                    # Generate conversational response based on document type
                    response_message = self._generate_document_response(file_type, structured_data)
                    
                    # Determine if we can skip some conversation steps
                    next_step = self._determine_next_step_after_document(
                        file_type, 
                        structured_data, 
                        conversation_state.get("current_step")
                    )
                    
                    return {
                        "message": response_message,
                        "state_update": {
                            "collected_data": collected_data,
                            "current_step": next_step,
                            "document_processed": True
                        },
                        "extraction_data": structured_data
                    }
                else:
                    # Document processing failed
                    error_msg = extracted_data.get("error", "Unknown processing error")
                    return {
                        "message": f"I had trouble reading your {file_type.replace('_', ' ')}: {error_msg}. Could you try uploading it again or tell me the information manually?",
                        "state_update": {},
                        "extraction_data": {}
                    }
            else:
                # No successful extractions
                return {
                    "message": f"I had trouble processing your {file_type.replace('_', ' ')}. The file might be corrupted or in an unsupported format. Could you try uploading it again or provide the information manually?",
                    "state_update": {},
                    "extraction_data": {}
                }
                
        except Exception as e:
            # Log the error for debugging
            logger.error(f"Document processing error for {file_path}: {str(e)}")
            
            return {
                "message": f"I encountered an error processing your document. Please try uploading it again or provide the information manually. If the problem persists, you can continue without the document.",
                "state_update": {},
                "extraction_data": {}
            }
    
    async def _handle_name_collection(self, user_message: str, collected_data: Dict) -> Dict[str, Any]:
        """Handle name collection step"""
        
        # Extract name from message
        name = self._extract_full_name(user_message)
        
        if name and len(name.split()) >= 2:
            collected_data["name"] = name
            collected_data["first_name"] = name.split()[0]
            collected_data["last_name"] = " ".join(name.split()[1:])
            
            return {
                "message": f"Nice to meet you, {name}! Now I need to verify your identity. Can you please upload your Emirates ID, or tell me your Emirates ID number?",
                "state_update": {
                    "current_step": ConversationStep.IDENTITY_VERIFICATION,
                    "collected_data": collected_data
                }
            }
        else:
            return {
                "message": "Could you please provide your full name (first and last name)? For example: 'Ahmed Al Mansouri'",
                "state_update": {}
            }
    
    async def _handle_identity_verification(self, user_message: str, collected_data: Dict) -> Dict[str, Any]:
        """Handle Emirates ID verification"""
        
        # Extract Emirates ID pattern (XXX-XXXX-XXXXXXX-X)
        emirates_id = self._extract_emirates_id(user_message)
        
        if emirates_id:
            collected_data["emirates_id"] = emirates_id
            # Mock validation (in real implementation, validate against government database)
            collected_data["id_verified"] = True
            
            return {
                "message": "Thank you! Your Emirates ID has been recorded. Now let's talk about your employment situation. Are you currently employed, unemployed, self-employed, or retired?",
                "state_update": {
                    "current_step": ConversationStep.EMPLOYMENT_INQUIRY,
                    "collected_data": collected_data
                }
            }
        else:
            return {
                "message": "I need a valid Emirates ID number (format: XXX-XXXX-XXXXXXX-X). Could you provide it or upload a photo of your Emirates ID?",
                "state_update": {}
            }
    
    async def _handle_employment_inquiry(self, user_message: str, collected_data: Dict) -> Dict[str, Any]:
        """Handle employment status inquiry"""
        
        employment_status = self._extract_employment_status(user_message)
        
        collected_data["employment_status"] = employment_status
        
        # Tailor next question based on employment status
        if employment_status == "employed":
            next_message = "Great! Since you're employed, what is your approximate monthly salary in AED? You can also upload a recent bank statement for more accurate assessment."
        elif employment_status == "self_employed":
            next_message = "I understand you're self-employed. What is your average monthly income from your business in AED? A bank statement would help me assess your financial situation more accurately."
        elif employment_status == "unemployed":
            next_message = "I understand you're currently unemployed. Do you receive any form of income (unemployment benefits, family support, etc.)? Please tell me your approximate monthly income in AED, even if it's zero."
        else:  # retired
            next_message = "I see you're retired. What is your monthly pension or retirement income in AED? You can also upload bank statements if available."
        
        return {
            "message": next_message,
            "state_update": {
                "current_step": ConversationStep.INCOME_ASSESSMENT,
                "collected_data": collected_data
            }
        }
    
    async def _handle_income_assessment(self, user_message: str, collected_data: Dict) -> Dict[str, Any]:
        """Handle income assessment"""
        
        monthly_income = self._extract_income_amount(user_message)
        
        collected_data["monthly_income"] = monthly_income
        
        return {
            "message": f"Thank you. I've noted your monthly income as {monthly_income:,.0f} AED. Now, how many people are in your household (including yourself)? For example, if you live with your spouse and 2 children, that would be 4 people total.",
            "state_update": {
                "current_step": ConversationStep.FAMILY_DETAILS,
                "collected_data": collected_data
            }
        }
    
    async def _handle_family_details(self, user_message: str, collected_data: Dict) -> Dict[str, Any]:
        """Handle family size and details"""
        
        family_size = self._extract_number(user_message)
        
        if family_size and family_size > 0:
            collected_data["family_size"] = family_size
            
            return {
                "message": f"Got it - {family_size} people in your household. What's your current housing situation? Do you own your home, rent, or live with family?",
                "state_update": {
                    "current_step": ConversationStep.HOUSING_SITUATION,
                    "collected_data": collected_data
                }
            }
        else:
            return {
                "message": "Please tell me the number of people in your household. Just say a number like '3' or 'four people'.",
                "state_update": {}
            }
    
    async def _handle_housing_situation(self, user_message: str, collected_data: Dict) -> Dict[str, Any]:
        """Handle housing situation inquiry"""
        
        housing_status = self._extract_housing_status(user_message)
        collected_data["housing_status"] = housing_status
        
        # Check if we have enough basic information to proceed
        if self._has_minimum_required_data(collected_data):
            return {
                "message": "Perfect! I have the basic information I need. You can upload additional documents (bank statements, credit reports, etc.) to improve the accuracy of your assessment, or I can proceed with the eligibility evaluation now. What would you prefer?",
                "state_update": {
                    "current_step": ConversationStep.DOCUMENT_COLLECTION,
                    "collected_data": collected_data
                }
            }
        else:
            missing_fields = self._get_missing_fields(collected_data)
            return {
                "message": f"I still need some information: {', '.join(missing_fields)}. Could you help me with that?",
                "state_update": {
                    "collected_data": collected_data
                }
            }
    
    async def _handle_document_collection(self, user_message: str, collected_data: Dict) -> Dict[str, Any]:
        """Handle document collection phase"""
        
        message_lower = user_message.lower()
        
        # Check for proceed/continue keywords
        proceed_keywords = ["proceed", "continue", "evaluate", "assessment", "skip", "done", "finished", "ready", "go ahead", "next"]
        if any(word in message_lower for word in proceed_keywords):
            # CRITICAL FIX: Don't directly proceed to eligibility assessment
            # Let the LangGraph workflow handle document processing first
            return {
                "message": "Perfect! I'll now process any uploaded documents and then proceed with your eligibility assessment. This may take a moment...",
                "state_update": {
                    "current_step": ConversationStep.ELIGIBILITY_PROCESSING,
                    "ready_for_assessment": True
                }
            }
        
        # Check for upload confirmation keywords (user indicating they've uploaded something)
        upload_confirmation_keywords = ["uploaded", "done uploading", "finished uploading", "sent", "attached", "submitted"]
        if any(phrase in message_lower for phrase in upload_confirmation_keywords):
            return {
                "message": "Great! I can see you've uploaded your documents. Would you like to upload any additional documents, or shall I proceed with the eligibility assessment now? Just say 'proceed' when you're ready.",
                "state_update": {}
            }
        
        # Check for upload intent keywords
        elif any(word in message_lower for word in ["upload", "document", "file", "statement", "add"]):
            return {
                "message": "Perfect! Please use the file upload area on the left to upload your documents. I can process bank statements, Emirates ID, resume, credit reports, and assets/liabilities spreadsheets. After uploading, let me know when you're ready to proceed with the assessment.",
                "state_update": {}
            }
        
        # Default response for unclear input
        else:
            return {
                "message": "You can either:\n• Upload additional documents using the file upload area\n• Say 'proceed' to continue with the eligibility assessment using the information you've already provided\n\nWhat would you like to do?",
                "state_update": {}
            }
    
    async def _proceed_to_eligibility_assessment(self, collected_data: Dict) -> Dict[str, Any]:
        """Proceed to eligibility assessment"""
        
        try:
            # Run eligibility assessment
            eligibility_result = await self.eligibility_agent.process({
                "application_data": collected_data,
                "assessment_mode": "conversational"
            })
            
            status = eligibility_result.get("status")
            if status == "success":
                # EligibilityAssessmentAgent returns the decision inside the key 'assessment_result'.
                # For backward-compatibility we also look for 'eligibility_result'.
                decision = eligibility_result.get("eligibility_result") or eligibility_result.get("assessment_result")
                if not decision:
                    # If the structure is unexpected, fall back to entire payload minus metadata keys
                    decision = {k: v for k, v in eligibility_result.items() if k not in {"status", "agent_name", "application_id", "assessed_at", "assessment_method", "reasoning"}}
                
                # Generate conversational response
                response_message = self._generate_eligibility_response(decision)
                
                return {
                    "message": response_message,
                    "state_update": {
                        "current_step": ConversationStep.COMPLETION,
                        "eligibility_result": decision
                    },
                    "application_complete": True,
                    "final_decision": decision
                }
            else:
                # Fall back to simple rule-based decision instead of looping
                fallback_decision = self._generate_fallback_decision(collected_data)
                response_message = self._generate_eligibility_response(fallback_decision)
                return {
                    "message": response_message + "\n\n(Note: A simplified assessment was used because the advanced evaluation encountered an issue.)",
                    "state_update": {
                        "current_step": ConversationStep.COMPLETION,
                        "eligibility_result": fallback_decision
                    },
                    "application_complete": True,
                    "final_decision": fallback_decision
                }
        
        except Exception as e:
            return {
                "message": f"I encountered an error during the assessment: {str(e)}. Let me try a simplified evaluation.",
                "state_update": {},
                "application_complete": True,
                "final_decision": self._generate_fallback_decision(collected_data)
            }
    
    async def _handle_completion_conversation(self, user_message: str, conversation_state: Dict) -> Dict[str, Any]:
        """Handle conversation after application completion"""
        
        message_lower = user_message.lower()
        collected_data = conversation_state.get("collected_data", {})
        eligibility_result = conversation_state.get("eligibility_result", {})
        
        # Check for restart/new application requests
        if any(phrase in message_lower for phrase in ["start over", "new application", "restart", "begin again"]):
            return {
                "message": "I'd be happy to help you start a new application! Let's begin fresh. What's your full name?",
                "state_update": {
                    "current_step": ConversationStep.NAME_COLLECTION,
                    "collected_data": {},
                    "eligibility_result": None
                }
            }
        
        # Check for economic enablement questions
        if any(word in message_lower for word in ["economic", "enablement", "recommendations", "programs", "training", "job", "career", "skills"]):
            return await self._provide_economic_enablement_details(eligibility_result, collected_data)
        
        # Check for eligibility questions
        if any(word in message_lower for word in ["eligibility", "eligible", "qualify", "decision", "result", "assessment"]):
            return await self._explain_eligibility_decision(eligibility_result, collected_data)
        
        # Check for support amount questions
        if any(word in message_lower for word in ["support", "amount", "money", "financial", "assistance"]):
            return await self._explain_support_details(eligibility_result)
        
        # Check for general help
        if any(word in message_lower for word in ["help", "what can", "options", "next"]):
            return {
                "message": """I can help you with several things:

• **Economic Enablement**: Ask about training programs, job opportunities, or skill development
• **Eligibility Details**: Get more information about your assessment results
• **New Application**: Start a fresh application if your circumstances have changed
• **General Questions**: Ask about the social support system

What would you like to know more about?""",
                "state_update": {}
            }
        
        # Default conversational response
        return {
            "message": "I'm here to help! You can ask me about economic enablement programs, your eligibility results, or start a new application. What would you like to know?",
            "state_update": {}
        }
    
    async def _provide_economic_enablement_details(self, eligibility_result: Dict, collected_data: Dict) -> Dict[str, Any]:
        """Provide detailed economic enablement recommendations"""
        
        employment_status = collected_data.get("employment_status", "unknown")
        monthly_income = collected_data.get("monthly_income", 0)
        
        # Get economic enablement from eligibility result if available
        economic_enablement = eligibility_result.get("economic_enablement", {})
        
        # FIXED: Check if we have LLM-generated recommendations first
        if economic_enablement.get("recommendations_text"):
            # We have LLM-generated recommendations - use them directly
            response = economic_enablement["recommendations_text"]
            logger.info("Using LLM-generated economic enablement recommendations")
        
        elif economic_enablement and (
            economic_enablement.get("recommendations") or 
            economic_enablement.get("training_programs") or 
            economic_enablement.get("job_opportunities") or
            economic_enablement.get("summary")
        ):
            # We have structured economic enablement data from EligibilityAssessmentAgent
            response = "## 🚀 Economic Enablement Recommendations\n\n"
            
            # Add summary if available
            if economic_enablement.get("summary"):
                response += economic_enablement["summary"]
            else:
                response += "Here are personalized recommendations to improve your financial situation:"
            
            # Add key recommendations if available
            if economic_enablement.get("recommendations") and isinstance(economic_enablement["recommendations"], list):
                response += "\n\n**Key Recommendations:**\n"
                for rec in economic_enablement["recommendations"][:5]:  # Show top 5
                    response += f"• {rec}\n"
            
            # Add training programs if available
            if economic_enablement.get("training_programs") and isinstance(economic_enablement["training_programs"], list):
                response += "\n\n**📚 Training Programs Available:**\n"
                for program in economic_enablement["training_programs"][:3]:  # Show top 3
                    if isinstance(program, dict):
                        name = program.get('name', 'Training Program')
                        duration = program.get('duration', 'Various durations')
                        description = program.get('description', '')
                        response += f"• **{name}** - {duration}\n"
                        if description:
                            response += f"  {description}\n"
                    else:
                        response += f"• {program}\n"
            
            # Add job opportunities if available
            if economic_enablement.get("job_opportunities") and isinstance(economic_enablement["job_opportunities"], list):
                response += "\n\n**💼 Job Opportunities:**\n"
                for job in economic_enablement["job_opportunities"][:3]:  # Show top 3
                    if isinstance(job, dict):
                        title = job.get('title', 'Job Opportunity')
                        company = job.get('company', 'Various Companies')
                        salary_range = job.get('salary_range', '')
                        response += f"• **{title}** at {company}\n"
                        if salary_range:
                            response += f"  Salary: {salary_range}\n"
                    else:
                        response += f"• {job}\n"
            
            # Add counseling services if available
            if economic_enablement.get("counseling_services") and isinstance(economic_enablement["counseling_services"], list):
                response += "\n\n**🤝 Support Services:**\n"
                for service in economic_enablement["counseling_services"][:3]:  # Show top 3
                    if isinstance(service, dict):
                        service_name = service.get('service', service.get('name', 'Support Service'))
                        provider = service.get('provider', '')
                        description = service.get('description', '')
                        response += f"• **{service_name}**"
                        if provider:
                            response += f" - {provider}"
                        response += "\n"
                        if description:
                            response += f"  {description}\n"
                    else:
                        response += f"• {service}\n"
            
            # Add financial programs if available
            if economic_enablement.get("financial_programs") and isinstance(economic_enablement["financial_programs"], list):
                response += "\n\n**💰 Financial Programs:**\n"
                for program in economic_enablement["financial_programs"][:3]:  # Show top 3
                    if isinstance(program, dict):
                        program_name = program.get('program', program.get('name', 'Financial Program'))
                        provider = program.get('provider', '')
                        description = program.get('description', '')
                        response += f"• **{program_name}**"
                        if provider:
                            response += f" - {provider}"
                        response += "\n"
                        if description:
                            response += f"  {description}\n"
                    else:
                        response += f"• {program}\n"
            
            logger.info("Using structured economic enablement recommendations from EligibilityAssessmentAgent")
        
        else:
            # Generate fresh LLM-powered personalized recommendations
            try:
                logger.info("No existing recommendations found, generating fresh LLM recommendations")
                llm_recommendations = await self._generate_llm_economic_recommendations(collected_data, eligibility_result)
                if llm_recommendations.get("status") == "success":
                    response = llm_recommendations["response"]
                    logger.info("Successfully generated fresh LLM recommendations")
                else:
                    # Fallback to hardcoded recommendations if LLM fails
                    logger.warning("LLM recommendations failed, using fallback")
                    response = self._generate_fallback_economic_recommendations(employment_status, monthly_income)
            except Exception as e:
                logger.error(f"Error generating LLM recommendations: {str(e)}")
                # Fallback to hardcoded recommendations
                response = self._generate_fallback_economic_recommendations(employment_status, monthly_income)
        
        response += "\n\nWould you like more details about any specific program or opportunity?"
        
        return {
            "message": response,
            "state_update": {}
        }
    
    async def _generate_llm_economic_recommendations(self, collected_data: Dict, eligibility_result: Dict) -> Dict[str, Any]:
        """Generate personalized economic enablement recommendations using LLM"""
        
        # Build concise user profile
        employment_status = collected_data.get("employment_status", "unknown")
        monthly_income = collected_data.get("monthly_income", 0)
        family_size = collected_data.get("family_size", 1)
        
        # SIMPLIFIED system prompt for faster response
        system_prompt = """You are a UAE economic advisor. Give exactly 3 brief recommendations in 30 words total. Be extremely concise. Format: 1. [recommendation] 2. [recommendation] 3. [recommendation]"""

        # SIMPLIFIED user prompt
        user_prompt = f"""Person: {employment_status}, {monthly_income} AED/month, {family_size} people.
        
3 brief income recommendations (30 words max):"""

        try:
            # Call LLM with simplified prompts
            llm_result = await self.invoke_llm(user_prompt, system_prompt)
            
            if llm_result.get("status") == "success" and llm_result.get("response"):
                # Format the LLM response simply
                formatted_response = "## 🚀 Economic Enablement Recommendations\n\n"
                formatted_response += llm_result["response"]
                
                return {
                    "status": "success",
                    "response": formatted_response,
                    "source": "llm_generated"
                }
            else:
                return {
                    "status": "error",
                    "error": llm_result.get("error", "LLM response failed"),
                    "response": ""
                }
                
        except Exception as e:
            logger.error(f"LLM recommendation generation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "response": ""
            }
    
    def _build_user_profile_for_llm(self, collected_data: Dict, eligibility_result: Dict) -> str:
        """Build a comprehensive user profile string for LLM context"""
        
        profile_parts = []
        
        # Basic Demographics
        if collected_data.get("name"):
            profile_parts.append(f"Name: {collected_data['name']}")
        
        # Employment Information
        employment_status = collected_data.get("employment_status", "unknown")
        profile_parts.append(f"Employment Status: {employment_status.replace('_', ' ').title()}")
        
        monthly_income = collected_data.get("monthly_income", 0)
        profile_parts.append(f"Monthly Income: {monthly_income:,.0f} AED")
        
        # Family Situation
        family_size = collected_data.get("family_size", 1)
        profile_parts.append(f"Household Size: {family_size} people")
        
        # Calculate per-person income
        per_person_income = monthly_income / family_size if family_size > 0 else monthly_income
        profile_parts.append(f"Per-Person Income: {per_person_income:,.0f} AED")
        
        # Housing Situation
        housing_status = collected_data.get("housing_status", "unknown")
        profile_parts.append(f"Housing Status: {housing_status.replace('_', ' ').title()}")
        
        # Financial Assessment
        if eligibility_result:
            eligible = eligibility_result.get("eligible", False)
            support_amount = eligibility_result.get("support_amount", 0)
            profile_parts.append(f"Support Eligibility: {'Eligible' if eligible else 'Not Eligible'}")
            if support_amount > 0:
                profile_parts.append(f"Approved Support: {support_amount:,.0f} AED/month")
        
        # Income Analysis
        if monthly_income < 2000:
            profile_parts.append("Income Level: Low income - needs immediate support")
        elif monthly_income < 5000:
            profile_parts.append("Income Level: Moderate income - opportunities for growth")
        else:
            profile_parts.append("Income Level: Stable income - focus on advancement")
        
        # Employment-specific context
        if employment_status == "unemployed":
            profile_parts.append("Priority: Job placement and skills development")
        elif employment_status == "employed" and monthly_income < 3000:
            profile_parts.append("Priority: Career advancement and income increase")
        elif employment_status == "self_employed":
            profile_parts.append("Priority: Business development and growth")
        elif employment_status == "retired":
            profile_parts.append("Priority: Supplemental income and financial security")
        
        return "\n".join(profile_parts)
    
    def _generate_fallback_economic_recommendations(self, employment_status: str, monthly_income: float) -> str:
        """Generate fallback recommendations when LLM is unavailable"""
        
        response = "## 🚀 Economic Enablement Recommendations\n\n"
        
        if employment_status == "unemployed":
            response += "Since you're currently unemployed, here are some opportunities:\n\n"
            response += "**📚 Skill Development:**\n"
            response += "• Digital literacy and computer skills training\n"
            response += "• Language courses (English/Arabic)\n"
            response += "• Vocational training in high-demand sectors\n\n"
            response += "**💼 Job Search Support:**\n"
            response += "• Resume writing and interview preparation\n"
            response += "• Job placement assistance\n"
            response += "• Career counseling services\n"
        
        elif employment_status == "employed" and monthly_income < 3000:
            response += "To increase your income potential:\n\n"
            response += "**📈 Career Advancement:**\n"
            response += "• Professional certification programs\n"
            response += "• Leadership and management training\n"
            response += "• Industry-specific skill upgrades\n\n"
            response += "**💰 Additional Income:**\n"
            response += "• Part-time opportunities\n"
            response += "• Freelancing skill development\n"
            response += "• Small business entrepreneurship training\n"
        
        else:
            response += "**📚 Continuous Learning:**\n"
            response += "• Professional development courses\n"
            response += "• Financial literacy programs\n"
            response += "• Investment and savings planning\n"
        
        response += "\n*Note: These are general recommendations. For personalized advice, please ensure the AI service is running.*"
        
        return response
    
    async def _explain_eligibility_decision(self, eligibility_result: Dict, collected_data: Dict) -> Dict[str, Any]:
        """Explain the eligibility decision in detail"""
        
        eligible = eligibility_result.get("eligible", False)
        support_amount = eligibility_result.get("support_amount", 0)
        reason = eligibility_result.get("reason", "Assessment completed based on provided information.")
        
        if eligible:
            response = f"✅ **Your Application Status: APPROVED**\n\n"
            response += f"You qualify for **{support_amount:,.0f} AED per month** in financial assistance.\n\n"
            
            if "breakdown" in eligibility_result:
                response += "**Support Breakdown:**\n"
                for item, amount in eligibility_result["breakdown"].items():
                    response += f"• {item}: {amount} AED\n"
            
            response += "\n**Next Steps:**\n"
            response += "• You will be contacted for final verification\n"
            response += "• Required documents will be collected\n"
            response += "• Support payments will begin after approval\n"
        
        else:
            response = f"❌ **Your Application Status: NOT APPROVED**\n\n"
            response += f"**Reason:** {reason}\n\n"
            response += "**Don't worry!** This doesn't mean you can't get help. I have economic enablement recommendations that can improve your situation and help you qualify in the future.\n\n"
            response += "**What you can do:**\n"
            response += "• Ask about economic enablement programs\n"
            response += "• Improve your situation with our recommendations\n"
            response += "• Reapply when your circumstances change\n"
        
        return {
            "message": response,
            "state_update": {}
        }
    
    async def _explain_support_details(self, eligibility_result: Dict) -> Dict[str, Any]:
        """Explain support amount and payment details"""
        
        eligible = eligibility_result.get("eligible", False)
        support_amount = eligibility_result.get("support_amount", 0)
        
        if eligible and support_amount > 0:
            response = f"💰 **Your Monthly Support: {support_amount:,.0f} AED**\n\n"
            
            if "breakdown" in eligibility_result:
                response += "**How this amount is calculated:**\n"
                for item, amount in eligibility_result["breakdown"].items():
                    response += f"• {item}: {amount} AED\n"
                response += "\n"
            
            response += "**Payment Information:**\n"
            response += "• Payments are made monthly\n"
            response += "• Direct bank transfer to your account\n"
            response += "• Regular review every 6 months\n"
            response += "• Must report any changes in circumstances\n"
        
        else:
            response = "Currently, you don't qualify for direct financial support. However, you may be eligible for:\n\n"
            response += "• Training program subsidies\n"
            response += "• Job placement assistance\n"
            response += "• Skill development vouchers\n"
            response += "• Career counseling services\n\n"
            response += "These programs can help improve your situation for future applications."
        
        return {
            "message": response,
            "state_update": {}
        }

    async def _handle_general_inquiry(self, user_message: str, current_step: str, collected_data: Dict) -> Dict[str, Any]:
        """Handle general inquiries and questions"""
        
        message_lower = user_message.lower()
        
        # CRITICAL FIX: If we're in completion step, don't handle economic enablement here
        # Let the completion conversation handler deal with it
        if current_step == ConversationStep.COMPLETION:
            # This should not happen - completion questions should be routed to completion handler
            logger.warning(f"General inquiry called during completion step: {user_message}")
            return {
                "message": "I'm processing your question. Please wait a moment...",
                "state_update": {}
            }
        
        # Check for specific question types (only for non-completion steps)
        if any(word in message_lower for word in ["economic", "enablement", "recommendations", "programs", "training", "job"]):
            return {
                "message": "I'd be happy to help with economic enablement recommendations! However, I need to complete your application assessment first to provide personalized recommendations. Let's continue with the application process.",
                "state_update": {}
            }
        
        elif any(word in message_lower for word in ["help", "question", "how", "what", "why"]):
            return {
                "message": "I'm here to help! I'm collecting information to assess your eligibility for social support. Feel free to ask specific questions, or we can continue with the application process.",
                "state_update": {}
            }
        
        else:
            return {
                "message": "I understand you're providing additional information. Let me continue with the next step in your application process.",
                "state_update": {}
            }
    
    # Helper methods for data extraction
    def _extract_full_name(self, text: str) -> str:
        """Extract full name from text"""
        # Simple implementation - in production, use NLP
        words = text.strip().split()
        if len(words) >= 2 and all(word.replace("'", "").isalpha() for word in words[:3]):
            return " ".join(words[:3])  # Take first 3 words max for name
        return ""
    
    def _extract_emirates_id(self, text: str) -> str:
        """Extract Emirates ID from text"""
        # Pattern: XXX-XXXX-XXXXXXX-X
        pattern = r'\d{3}-?\d{4}-?\d{7}-?\d{1}'
        match = re.search(pattern, text.replace(" ", ""))
        if match:
            id_num = match.group()
            # Format consistently
            digits = re.sub(r'[^0-9]', '', id_num)
            if len(digits) == 15:
                return f"{digits[0:3]}-{digits[3:7]}-{digits[7:14]}-{digits[14]}"
        return ""
    
    def _extract_employment_status(self, text: str) -> str:
        """Extract employment status from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["employed", "working", "job", "work"]) and "unemployed" not in text_lower:
            return "employed"
        elif any(word in text_lower for word in ["self", "business", "own", "freelance", "entrepreneur"]):
            return "self_employed"
        elif any(word in text_lower for word in ["retired", "pension", "retire"]):
            return "retired"
        else:
            return "unemployed"
    
    def _extract_income_amount(self, text: str) -> float:
        """Extract income amount from text"""
        # Look for numbers in the text
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d{2})?\b', text.replace(",", ""))
        
        if numbers:
            # Take the largest number found (assuming it's the income)
            amounts = [float(num.replace(",", "")) for num in numbers]
            return max(amounts)
        
        # If no number found, check for zero indicators
        if any(word in text.lower() for word in ["zero", "nothing", "none", "no income"]):
            return 0.0
        
        return 0.0  # Default
    
    def _extract_number(self, text: str) -> Optional[int]:
        """Extract a number from text"""
        # Look for digits
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            return int(numbers[0])
        
        # Look for written numbers
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        
        text_lower = text.lower()
        for word, num in word_to_num.items():
            if word in text_lower:
                return num
        
        return None
    
    def _extract_housing_status(self, text: str) -> str:
        """Extract housing status from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["own", "owner", "bought", "mortgage"]):
            return "owned"
        elif any(word in text_lower for word in ["rent", "renting", "tenant", "lease"]):
            return "rented"
        elif any(word in text_lower for word in ["family", "parents", "relatives", "free"]):
            return "family"
        else:
            return "other"
    
    def _has_minimum_required_data(self, collected_data: Dict) -> bool:
        """Check if we have minimum required data"""
        required = ["name", "employment_status", "monthly_income", "family_size"]
        return all(field in collected_data for field in required)
    
    def _get_missing_fields(self, collected_data: Dict) -> List[str]:
        """Get list of missing required fields"""
        required = ["name", "emirates_id", "employment_status", "monthly_income", "family_size"]
        return [field for field in required if field not in collected_data]
    
    def _generate_document_response(self, file_type: str, extracted_data: Dict) -> str:
        """Generate conversational response for document processing"""
        
        # Check for OCR-related errors
        if extracted_data.get("error") and "ocr" in extracted_data.get("error", "").lower():
            if file_type == "emirates_id":
                return "I can see you uploaded your Emirates ID, but I can't read text from images right now. Could you please tell me your full name and Emirates ID number instead? I'll still be able to help you with your application."
            else:
                return f"I received your {file_type.replace('_', ' ')}, but I can't process images right now. Could you please provide the key information manually? This won't affect your application."
        
        # Check for manual input required
        if extracted_data.get("user_message"):
            return extracted_data["user_message"]
        
        # Generate specific responses based on document type
        if file_type == "emirates_id":
            name = extracted_data.get("name", "")
            age = extracted_data.get("age", "")
            if name and age:
                response = f"Perfect! I've processed your Emirates ID. I can see you're {name}, age {age}. This information matches what you told me earlier."
            else:
                response = "I've processed your Emirates ID. The information looks good!"
        
        elif file_type == "bank_statement":
            income = extracted_data.get("monthly_income", 0)
            if income > 0:
                response = f"Excellent! I've analyzed your bank statement. I can see your average monthly income is approximately {income:,.0f} AED. This helps me make a more accurate assessment."
            else:
                response = "I've processed your bank statement. This will help me better understand your financial situation."
        
        elif file_type == "resume":
            experience = extracted_data.get("experience_years", 0) 
            skills = extracted_data.get("skills", [])
            if experience or skills:
                response = f"Great! I've reviewed your resume. I can see you have {experience} years of experience and valuable skills. This will help with economic enablement recommendations."
            else:
                response = "I've processed your resume. This information will be helpful for career recommendations."
        
        else:
            response = f"Thank you for uploading your {file_type.replace('_', ' ')}. I've processed it and the information will improve the accuracy of your assessment."
        
        # Always add next steps guidance
        response += "\n\nYou can upload more documents or say 'proceed' to continue with the eligibility evaluation."
        
        return response
    
    def _determine_next_step_after_document(self, file_type: str, extracted_data: Dict, current_step: str) -> str:
        """Determine next conversation step after document processing"""
        
        # If Emirates ID processed, we can skip identity verification
        if file_type == "emirates_id" and current_step == ConversationStep.IDENTITY_VERIFICATION:
            return ConversationStep.EMPLOYMENT_INQUIRY
        
        # If bank statement processed during income assessment, we have income data
        elif file_type == "bank_statement" and current_step == ConversationStep.INCOME_ASSESSMENT:
            return ConversationStep.FAMILY_DETAILS
        
        # For all other cases, move to document collection step so user can proceed
        elif current_step != ConversationStep.DOCUMENT_COLLECTION:
            return ConversationStep.DOCUMENT_COLLECTION
        
        # Otherwise, stay in current step
        else:
            return current_step
    
    def _generate_eligibility_response(self, decision: Dict) -> str:
        """Generate conversational response for eligibility decision"""
        
        eligibility = decision.get("eligible", False)
        support_amount = decision.get("support_amount", 0)
        
        if eligibility:
            response = f"🎉 Great news! Based on your information, you are eligible for social support. "
            response += f"You qualify for {support_amount:,.0f} AED per month in financial assistance.\n\n"
            
            # Add breakdown if available
            if "breakdown" in decision:
                response += "Here's how this amount is calculated:\n"
                for item, amount in decision["breakdown"].items():
                    response += f"• {item}: {amount} AED\n"
                response += "\n"
            
            # Add economic enablement recommendations
            if "economic_enablement" in decision:
                enablement = decision["economic_enablement"]
                response += "## 🚀 Economic Enablement Opportunities\n\n"
                response += enablement.get("summary", "")
                
                if "recommendations" in enablement and enablement["recommendations"]:
                    response += "\n\n**Key Recommendations:**\n"
                    for rec in enablement["recommendations"][:3]:  # Show top 3
                        response += f"• {rec}\n"
                
                # Add program counts
                programs_info = []
                if "training_programs" in enablement:
                    programs_info.append(f"**Training Programs:** {len(enablement['training_programs'])} available")
                if "job_opportunities" in enablement:
                    programs_info.append(f"**Job Opportunities:** {len(enablement['job_opportunities'])} types")
                if "counseling_services" in enablement:
                    programs_info.append(f"**Support Services:** {len(enablement['counseling_services'])} services")
                
                if programs_info:
                    response += "\n\n" + " | ".join(programs_info)
            else:
                response += "\n\nI also have economic enablement recommendations to help you build long-term financial independence."
            
        else:
            response = "I've completed the assessment of your application. "
            reason = decision.get("reason", "Based on the current criteria, you don't qualify for direct financial support at this time.")
            response += reason
            
            # Add economic enablement recommendations for declined applications
            if "economic_enablement" in decision:
                enablement = decision["economic_enablement"]
                response += "\n\n## 🚀 Economic Enablement Opportunities\n\n"
                response += enablement.get("summary", "")
                
                if "recommendations" in enablement and enablement["recommendations"]:
                    response += "\n\n**Key Recommendations:**\n"
                    for rec in enablement["recommendations"][:3]:  # Show top 3
                        response += f"• {rec}\n"
                
                # Add program counts
                programs_info = []
                if "training_programs" in enablement:
                    programs_info.append(f"**Training Programs:** {len(enablement['training_programs'])} available")
                if "job_opportunities" in enablement:
                    programs_info.append(f"**Job Opportunities:** {len(enablement['job_opportunities'])} types")
                if "counseling_services" in enablement:
                    programs_info.append(f"**Support Services:** {len(enablement['counseling_services'])} services")
                
                if programs_info:
                    response += "\n\n" + " | ".join(programs_info)
            else:
                response += "\n\nHowever, I have economic enablement recommendations that can help improve your situation."
        
        return response
    
    def _generate_fallback_decision(self, collected_data: Dict) -> Dict:
        """Generate a fallback decision when assessment fails"""
        
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
                "reason": "Approved based on income threshold assessment"
            }
        else:
            return {
                "eligible": False,
                "decision": "declined",
                "support_amount": 0,
                "reason": "Monthly income exceeds the threshold for direct financial support"
            }
    
    async def _handle_corrections_and_navigation(
        self, user_message: str, conversation_state: Dict
    ) -> Optional[Dict[str, Any]]:
        """Handle user corrections, going back, or updating previous information"""
        
        message_lower = user_message.lower()
        collected_data = conversation_state.get("collected_data", {})
        
        # Check for correction keywords
        correction_keywords = [
            "correct", "change", "update", "wrong", "mistake", "fix", 
            "go back", "previous", "earlier", "modify", "edit"
        ]
        
        if any(keyword in message_lower for keyword in correction_keywords):
            return await self._handle_correction_request(user_message, conversation_state)
        
        # Check for specific field updates (e.g., "my name is actually...")
        field_updates = self._detect_field_updates(user_message, collected_data)
        if field_updates:
            return await self._handle_field_updates(field_updates, conversation_state)
        
        # Check for navigation commands
        navigation_response = self._handle_navigation_commands(user_message, conversation_state)
        if navigation_response:
            return navigation_response
        
        return None
    
    async def _handle_correction_request(
        self, user_message: str, conversation_state: Dict
    ) -> Dict[str, Any]:
        """Handle explicit correction requests"""
        
        collected_data = conversation_state.get("collected_data", {})
        
        # Show current information and ask what to correct
        current_info = self._format_collected_data_for_review(collected_data)
        
        correction_message = f"""I understand you want to make a correction. Here's what I have so far:

{current_info}

What would you like to correct? You can say things like:
• "My name is actually [correct name]"
• "Change my income to [amount]"
• "My family size is [number]"
• "Go back to employment status"

Or tell me specifically what needs to be corrected."""
        
        return {
            "message": correction_message,
            "state_update": {
                "awaiting_correction": True
            }
        }
    
    def _detect_field_updates(self, user_message: str, collected_data: Dict) -> Dict[str, Any]:
        """Detect if user is providing updates to specific fields"""
        
        updates = {}
        message_lower = user_message.lower()
        
        # Name updates
        name_patterns = [
            r"my name is (?:actually )?(.+)",
            r"name should be (.+)",
            r"correct name is (.+)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, message_lower)
            if match:
                new_name = match.group(1).strip()
                if len(new_name.split()) >= 2:
                    updates["name"] = new_name
                    updates["first_name"] = new_name.split()[0]
                    updates["last_name"] = " ".join(new_name.split()[1:])
        
        # Emirates ID updates
        emirates_patterns = [
            r"my (?:emirates )?id is (?:actually )?(\d{3}-?\d{4}-?\d{7}-?\d{1})",
            r"correct id is (\d{3}-?\d{4}-?\d{7}-?\d{1})",
            r"id should be (\d{3}-?\d{4}-?\d{7}-?\d{1})"
        ]
        for pattern in emirates_patterns:
            match = re.search(pattern, user_message.replace(" ", ""))
            if match:
                emirates_id = match.group(1)
                # Format consistently
                digits = re.sub(r'[^0-9]', '', emirates_id)
                if len(digits) == 15:
                    updates["emirates_id"] = f"{digits[0:3]}-{digits[3:7]}-{digits[7:14]}-{digits[14]}"
        
        # Income updates
        income_patterns = [
            r"my (?:monthly )?(?:salary|income) is (?:actually )?(\d+(?:,\d{3})*(?:\.\d{2})?)",
            r"income should be (\d+(?:,\d{3})*(?:\.\d{2})?)",
            r"correct income is (\d+(?:,\d{3})*(?:\.\d{2})?)"
        ]
        for pattern in income_patterns:
            match = re.search(pattern, message_lower)
            if match:
                income_str = match.group(1).replace(',', '')
                try:
                    updates["monthly_income"] = float(income_str)
                except ValueError:
                    pass
        
        # Employment status updates
        employment_patterns = [
            r"i am (?:actually )?(?:currently )?(employed|unemployed|self[- ]?employed|retired)",
            r"employment (?:status )?(?:is|should be) (employed|unemployed|self[- ]?employed|retired)"
        ]
        for pattern in employment_patterns:
            match = re.search(pattern, message_lower)
            if match:
                status = match.group(1).replace("-", "_").replace(" ", "_")
                updates["employment_status"] = status
        
        # Family size updates
        family_patterns = [
            r"(?:my )?family size is (?:actually )?(\d+)",
            r"we are (?:actually )?(\d+) people",
            r"family should be (\d+)"
        ]
        for pattern in family_patterns:
            match = re.search(pattern, message_lower)
            if match:
                try:
                    updates["family_size"] = int(match.group(1))
                except ValueError:
                    pass
        
        # Housing status updates
        housing_patterns = [
            r"we (?:actually )?(own|rent|live with family)",
            r"housing (?:is|should be) (own|rent|family)"
        ]
        for pattern in housing_patterns:
            match = re.search(pattern, message_lower)
            if match:
                housing_type = match.group(1)
                if "own" in housing_type:
                    updates["housing_status"] = "owned"
                elif "rent" in housing_type:
                    updates["housing_status"] = "rented"
                elif "family" in housing_type:
                    updates["housing_status"] = "family"
        
        return updates
    
    async def _handle_field_updates(
        self, field_updates: Dict[str, Any], conversation_state: Dict
    ) -> Dict[str, Any]:
        """Handle specific field updates"""
        
        collected_data = conversation_state.get("collected_data", {})
        
        # Apply updates
        old_values = {}
        for field, new_value in field_updates.items():
            old_values[field] = collected_data.get(field, "Not set")
            collected_data[field] = new_value
        
        # Generate confirmation message
        update_messages = []
        for field, new_value in field_updates.items():
            old_value = old_values[field]
            field_name = field.replace("_", " ").title()
            update_messages.append(f"• {field_name}: {old_value} → {new_value}")
        
        confirmation_message = f"""✅ I've updated your information:

{chr(10).join(update_messages)}

Is this correct now? We can continue from where we left off, or you can make more corrections."""
        
        return {
            "message": confirmation_message,
            "state_update": {
                "collected_data": collected_data,
                "awaiting_correction": False
            }
        }
    
    def _handle_navigation_commands(
        self, user_message: str, conversation_state: Dict
    ) -> Optional[Dict[str, Any]]:
        """Handle navigation commands like 'go back', 'start over', etc."""
        
        message_lower = user_message.lower()
        current_step = conversation_state.get("current_step")
        
        # Start over
        if any(phrase in message_lower for phrase in ["start over", "restart", "begin again"]):
            return {
                "message": "Sure! Let's start over. What's your full name?",
                "state_update": {
                    "current_step": ConversationStep.NAME_COLLECTION,
                    "collected_data": {},
                    "awaiting_correction": False
                }
            }
        
        # Go back to specific steps
        step_mappings = {
            "name": ConversationStep.NAME_COLLECTION,
            "identity": ConversationStep.IDENTITY_VERIFICATION,
            "emirates": ConversationStep.IDENTITY_VERIFICATION,
            "employment": ConversationStep.EMPLOYMENT_INQUIRY,
            "job": ConversationStep.EMPLOYMENT_INQUIRY,
            "income": ConversationStep.INCOME_ASSESSMENT,
            "salary": ConversationStep.INCOME_ASSESSMENT,
            "family": ConversationStep.FAMILY_DETAILS,
            "housing": ConversationStep.HOUSING_SITUATION
        }
        
        for keyword, target_step in step_mappings.items():
            if f"go back to {keyword}" in message_lower or f"change {keyword}" in message_lower:
                step_messages = {
                    ConversationStep.NAME_COLLECTION: "Let's update your name. What's your full name?",
                    ConversationStep.IDENTITY_VERIFICATION: "Let's verify your identity again. What's your Emirates ID number?",
                    ConversationStep.EMPLOYMENT_INQUIRY: "Let's update your employment information. Are you currently employed, unemployed, self-employed, or retired?",
                    ConversationStep.INCOME_ASSESSMENT: "Let's update your income information. What's your monthly income in AED?",
                    ConversationStep.FAMILY_DETAILS: "Let's update your family information. How many people are in your household?",
                    ConversationStep.HOUSING_SITUATION: "Let's update your housing information. Do you own your home, rent, or live with family?"
                }
                
                return {
                    "message": step_messages.get(target_step, "Let's update that information."),
                    "state_update": {
                        "current_step": target_step,
                        "awaiting_correction": False
                    }
                }
        
        # Generic go back
        if "go back" in message_lower or "previous step" in message_lower:
            previous_step = self._get_previous_step(current_step)
            if previous_step:
                return {
                    "message": f"Going back to the previous step. {self._get_step_question(previous_step)}",
                    "state_update": {
                        "current_step": previous_step,
                        "awaiting_correction": False
                    }
                }
        
        return None
    
    def _get_previous_step(self, current_step: str) -> Optional[str]:
        """Get the previous step in the conversation flow"""
        
        step_order = [
            ConversationStep.NAME_COLLECTION,
            ConversationStep.IDENTITY_VERIFICATION,
            ConversationStep.EMPLOYMENT_INQUIRY,
            ConversationStep.INCOME_ASSESSMENT,
            ConversationStep.FAMILY_DETAILS,
            ConversationStep.HOUSING_SITUATION,
            ConversationStep.DOCUMENT_COLLECTION
        ]
        
        try:
            current_index = step_order.index(current_step)
            if current_index > 0:
                return step_order[current_index - 1]
        except ValueError:
            pass
        
        return None
    
    def _get_step_question(self, step: str) -> str:
        """Get the question for a specific step"""
        
        questions = {
            ConversationStep.NAME_COLLECTION: "What's your full name?",
            ConversationStep.IDENTITY_VERIFICATION: "What's your Emirates ID number?",
            ConversationStep.EMPLOYMENT_INQUIRY: "What's your employment status?",
            ConversationStep.INCOME_ASSESSMENT: "What's your monthly income in AED?",
            ConversationStep.FAMILY_DETAILS: "How many people are in your household?",
            ConversationStep.HOUSING_SITUATION: "What's your housing situation?"
        }
        
        return questions.get(step, "Please provide the requested information.")
    
    def _format_collected_data_for_review(self, collected_data: Dict) -> str:
        """Format collected data for user review"""
        
        if not collected_data:
            return "No information collected yet."
        
        formatted_lines = []
        
        field_labels = {
            "name": "👤 Name",
            "emirates_id": "🆔 Emirates ID",
            "employment_status": "💼 Employment Status",
            "monthly_income": "💰 Monthly Income",
            "family_size": "👨‍👩‍👧‍👦 Family Size",
            "housing_status": "🏠 Housing Status"
        }
        
        for field, label in field_labels.items():
            if field in collected_data:
                value = collected_data[field]
                if field == "monthly_income":
                    value = f"{value:,.0f} AED"
                elif field == "family_size":
                    value = f"{value} people"
                elif field == "employment_status":
                    value = value.replace("_", " ").title()
                elif field == "housing_status":
                    value = value.replace("_", " ").title()
                
                formatted_lines.append(f"{label}: {value}")
        
        return "\n".join(formatted_lines) if formatted_lines else "No information collected yet." 