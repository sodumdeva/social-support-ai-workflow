# Create new conversational interface
import streamlit as st
import requests
import json
from typing import Dict, List
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import settings

# API base URL
API_BASE = f"http://{settings.api_host}:{settings.api_port}"

def show_chat_interface():
    """Main conversational interface with interactive buttons for social support application"""
    st.title("🤖 Social Support AI Assistant")
    st.markdown("I'll guide you through your social support application step by step.")
    
    # Initialize session state
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = {
            "current_step": "name_collection",
            "collected_data": {},
            "uploaded_documents": [],
            "application_id": None
        }
    
    # Initialize with greeting message only if no messages exist AND we're starting fresh
    if "conversation_messages" not in st.session_state:
        st.session_state.conversation_messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your Social Support AI Assistant. I'll help you apply for financial support through an easy conversation. Let's start - what's your full name?"
            }
        ]
    
    # Display chat messages
    for message in st.session_state.conversation_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Interactive buttons section based on current step
    show_interactive_buttons()
    
    # Document upload section (always visible)
    with st.sidebar:
        st.markdown("### 📄 Upload Documents")
        st.markdown("You can upload documents anytime during our conversation:")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'png', 'jpg', 'jpeg', 'xlsx', 'xls', 'docx'],
            help="Supported: Bank statements, Emirates ID, Resume, Credit reports, Assets/liabilities"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file not in st.session_state.conversation_state["uploaded_documents"]:
                    # Process uploaded document
                    document_response = process_document_upload(uploaded_file)
                    st.session_state.conversation_state["uploaded_documents"].append(uploaded_file)
                    
                    # Add document processing message to chat
                    st.session_state.conversation_messages.append({
                        "role": "assistant",
                        "content": document_response
                    })
                    st.rerun()
        
        # Show current progress
        show_progress_sidebar()
    
    # Chat input
    if prompt := st.chat_input("Type your response here..."):
        handle_user_input(prompt)

def show_interactive_buttons():
    """Show context-aware interactive buttons based on current conversation step"""
    
    # Safety check: ensure conversation_state exists
    if "conversation_state" not in st.session_state or st.session_state.conversation_state is None:
        st.session_state.conversation_state = {
            "current_step": "name_collection",
            "collected_data": {},
            "uploaded_documents": [],
            "application_id": None
        }
    
    current_step = st.session_state.conversation_state.get("current_step", "name_collection")
    collected_data = st.session_state.conversation_state.get("collected_data", {})
    
    # Name Collection Presets (small buttons)
    if current_step == "name_collection":
        st.markdown("##### Quick options:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Ahmed Al Mansouri", key="btn_name_1", help="Test name 1"):
                handle_button_click("Ahmed Al Mansouri")
        
        with col2:
            if st.button("Fatima Al Zahra", key="btn_name_2", help="Test name 2"):
                handle_button_click("Fatima Al Zahra")
    
    # Emirates ID Presets (small buttons)
    elif current_step == "identity_verification":
        st.markdown("##### Quick options:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("784-1990-1234567-1", key="btn_id_1", help="Test Emirates ID 1"):
                handle_button_click("784-1990-1234567-1")
        
        with col2:
            if st.button("784-1985-7654321-2", key="btn_id_2", help="Test Emirates ID 2"):
                handle_button_click("784-1985-7654321-2")
    
    # Employment Status Buttons
    elif current_step == "employment_inquiry":
        st.markdown("### 💼 Select Your Employment Status:")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👔 Employed", use_container_width=True, key="btn_employed"):
                handle_button_click("employed")
        
        with col2:
            if st.button("❌ Unemployed", use_container_width=True, key="btn_unemployed"):
                handle_button_click("unemployed")
        
        with col3:
            if st.button("🏢 Self-Employed", use_container_width=True, key="btn_self_employed"):
                handle_button_click("self-employed")
        
        with col4:
            if st.button("🏖️ Retired", use_container_width=True, key="btn_retired"):
                handle_button_click("retired")
    
    # Family Size Buttons
    elif current_step == "family_details":
        st.markdown("### 👨‍👩‍👧‍👦 Quick Family Size Selection:")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        family_sizes = [1, 2, 3, 4, 5]
        for i, size in enumerate(family_sizes):
            with [col1, col2, col3, col4, col5][i]:
                if st.button(f"{size} {'person' if size == 1 else 'people'}", use_container_width=True, key=f"btn_family_{size}"):
                    handle_button_click(str(size))
        
        # Option for larger families
        if st.button("6+ people (click to specify)", use_container_width=True, key="btn_family_more"):
            st.info("Please type the exact number of people in your household in the chat.")
    
    # Housing Status Buttons
    elif current_step == "housing_situation":
        st.markdown("### 🏠 Select Your Housing Situation:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏡 Own Home", use_container_width=True, key="btn_own"):
                handle_button_click("own")
        
        with col2:
            if st.button("🏠 Rent", use_container_width=True, key="btn_rent"):
                handle_button_click("rent")
        
        with col3:
            if st.button("👨‍👩‍👧‍👦 Live with Family", use_container_width=True, key="btn_family"):
                handle_button_click("live with family")
    
    # Document Collection Buttons
    elif current_step == "document_collection":
        st.markdown("### 📄 Document Upload Options:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Upload More Documents", use_container_width=True, key="btn_upload_more"):
                st.info("👆 Use the file uploader in the sidebar to add more documents")
        
        with col2:
            if st.button("✅ Proceed with Assessment", use_container_width=True, key="btn_proceed"):
                handle_button_click("proceed with assessment")
    
    # Completion Step Buttons
    elif current_step == "completion":
        st.markdown("### 🎯 What would you like to do next?")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Economic Enablement", use_container_width=True, key="btn_economic"):
                handle_button_click("what economic enablement recommendations do you suggest?")
        
        with col2:
            if st.button("📊 Eligibility Details", use_container_width=True, key="btn_eligibility"):
                handle_button_click("explain my eligibility decision")
        
        with col3:
            if st.button("🔄 New Application", use_container_width=True, key="btn_new_app"):
                handle_button_click("start new application")
        
        # Additional help options
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💰 Support Details", use_container_width=True, key="btn_support"):
                handle_button_click("tell me about support amount details")
        
        with col2:
            if st.button("❓ General Help", use_container_width=True, key="btn_help_completion"):
                handle_button_click("what can you help me with?")
    
    # General Action Buttons (always available except completion)
    if current_step not in ["completion"]:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Start Over", use_container_width=True, key="btn_restart"):
                restart_conversation()
        
        with col2:
            if st.button("❓ Need Help", use_container_width=True, key="btn_help"):
                show_help_info()
        
        with col3:
            # Show skip button for optional steps
            if current_step in ["document_collection"]:
                if st.button("⏭️ Skip Documents", use_container_width=True, key="btn_skip"):
                    handle_button_click("skip documents and proceed")

def show_progress_sidebar():
    """Show conversation progress in sidebar"""
    
    st.sidebar.markdown("### 📊 Application Progress")
    
    # Safety check: ensure conversation_state exists
    if "conversation_state" not in st.session_state or st.session_state.conversation_state is None:
        st.session_state.conversation_state = {
            "current_step": "name_collection",
            "collected_data": {},
            "uploaded_documents": [],
            "application_id": None
        }
    
    current_step = st.session_state.conversation_state.get("current_step", "greeting")
    collected_data = st.session_state.conversation_state.get("collected_data", {})
    eligibility_result = st.session_state.conversation_state.get("eligibility_result", {})
    
    # Define steps and their completion status
    steps = [
        ("👤 Name", "name" in collected_data),
        ("🆔 Identity", "emirates_id" in collected_data),
        ("💼 Employment", "employment_status" in collected_data),
        ("💰 Income", "monthly_income" in collected_data),
        ("👨‍👩‍👧‍👦 Family", "family_size" in collected_data),
        ("🏠 Housing", "housing_status" in collected_data),
        ("📄 Documents", len(st.session_state.conversation_state.get("uploaded_documents", [])) > 0),
        ("✅ Assessment", current_step == "completion")
    ]
    
    for step_name, completed in steps:
        if completed:
            st.sidebar.markdown(f"✅ {step_name}")
        else:
            st.sidebar.markdown(f"⏳ {step_name}")
    
    # Show completion status
    if current_step == "completion":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎉 Application Complete!")
        
        if eligibility_result:
            eligible = eligibility_result.get("eligible", False)
            support_amount = eligibility_result.get("support_amount", 0)
            
            if eligible:
                st.sidebar.success(f"✅ **APPROVED**\n\n{support_amount:,.0f} AED/month")
            else:
                st.sidebar.error("❌ **NOT APPROVED**")
                st.sidebar.info("💡 Ask about economic enablement programs!")
    
    # Show collected data summary
    elif collected_data:
        st.sidebar.markdown("### 📋 Information Collected")
        
        if "name" in collected_data:
            st.sidebar.markdown(f"**Name:** {collected_data['name']}")
        
        if "employment_status" in collected_data:
            st.sidebar.markdown(f"**Employment:** {collected_data['employment_status'].replace('_', ' ').title()}")
        
        if "monthly_income" in collected_data:
            st.sidebar.markdown(f"**Income:** {collected_data['monthly_income']:,.0f} AED")
        
        if "family_size" in collected_data:
            st.sidebar.markdown(f"**Family Size:** {collected_data['family_size']} people")
        
        if "housing_status" in collected_data:
            st.sidebar.markdown(f"**Housing:** {collected_data['housing_status'].replace('_', ' ').title()}")

def handle_button_click(button_value: str):
    """Handle button clicks by processing them as user input"""
    
    # Add user message for the button click
    st.session_state.conversation_messages.append({
        "role": "user",
        "content": button_value
    })
    
    # Process the button click as a regular message
    try:
        response = process_chat_message(
            button_value,
            st.session_state.conversation_messages,
            st.session_state.conversation_state
        )
        
        # Ensure response is not None
        if response is None:
            response = {
                "message": "I'm having trouble processing your request right now. Could you please try again?",
                "state_update": {}
            }
        
        # CRITICAL: Check if this is a restart scenario
        state_update = response.get("state_update", {})
        if (state_update.get("current_step") == "name_collection" and 
            state_update.get("collected_data") == {} and
            "start" in button_value.lower() and "new" in button_value.lower()):
            
            # This is a restart - reset frontend state completely
            st.session_state.conversation_state = {
                "current_step": "name_collection",
                "collected_data": {},
                "uploaded_documents": [],
                "application_id": None
            }
            
            # Keep only the restart message and response
            restart_messages = [
                {
                    "role": "user",
                    "content": button_value
                },
                {
                    "role": "assistant", 
                    "content": response.get("message", "I'd be happy to help you start a new application! Let's begin fresh. What's your full name?")
                }
            ]
            st.session_state.conversation_messages = restart_messages
            st.rerun()
            return
        
        # Add assistant response only if there is a message
        if response.get("message"):
            st.session_state.conversation_messages.append({
                "role": "assistant",
                "content": response["message"]
            })
        
        # Update conversation state
        st.session_state.conversation_state.update(response.get("state_update", {}))
        
        # Check if application is complete
        if response.get("application_complete"):
            st.session_state.conversation_state["application_complete"] = True
            show_final_results(response.get("final_decision"))
        
    except Exception as e:
        st.session_state.conversation_messages.append({
            "role": "assistant",
            "content": f"I apologize, I encountered an error: {str(e)}. Let me try to help you in a different way."
        })
    
    st.rerun()

def handle_user_input(prompt: str):
    """Handle user text input"""
    
    # Add user message
    st.session_state.conversation_messages.append({
        "role": "user", 
        "content": prompt
    })
    
    # Process with conversation agent
    try:
        response = process_chat_message(
            prompt, 
            st.session_state.conversation_messages,
            st.session_state.conversation_state
        )
        
        # Ensure response is not None
        if response is None:
            response = {
                "message": "I'm having trouble processing your request right now. Could you please try again?",
                "state_update": {}
            }
        
        # CRITICAL: Check if this is a restart scenario
        state_update = response.get("state_update", {})
        if (state_update.get("current_step") == "name_collection" and 
            state_update.get("collected_data") == {} and
            "start" in prompt.lower() and "new" in prompt.lower()):
            
            # This is a restart - reset frontend state completely
            st.session_state.conversation_state = {
                "current_step": "name_collection",
                "collected_data": {},
                "uploaded_documents": [],
                "application_id": None
            }
            
            # Keep only the restart message and response
            restart_messages = [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant", 
                    "content": response.get("message", "I'd be happy to help you start a new application! Let's begin fresh. What's your full name?")
                }
            ]
            st.session_state.conversation_messages = restart_messages
            st.rerun()
            return
        
        # Add assistant response only if there is a message
        if response.get("message"):
            st.session_state.conversation_messages.append({
                "role": "assistant",
                "content": response["message"]
            })
        
        # Update conversation state
        st.session_state.conversation_state.update(response.get("state_update", {}))
        
        # Check if application is complete
        if response.get("application_complete"):
            st.session_state.conversation_state["application_complete"] = True
            show_final_results(response.get("final_decision"))
            
    except Exception as e:
        st.session_state.conversation_messages.append({
            "role": "assistant",
            "content": f"I apologize, I encountered an error: {str(e)}. Let me try to help you in a different way."
        })
    
    st.rerun()

def restart_conversation():
    """Restart the conversation from the beginning"""
    
    st.session_state.conversation_state = {
        "current_step": "name_collection",
        "collected_data": {},
        "uploaded_documents": [],
        "application_id": None
    }
    
    st.session_state.conversation_messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your Social Support AI Assistant. I'll help you apply for financial support through an easy conversation. Let's start - what's your full name?"
        }
    ]
    
    st.rerun()

def show_help_info():
    """Show help information"""
    
    st.info("""
    **How to use this application:**
    
    1. **Answer Questions**: I'll ask you questions step by step
    2. **Use Buttons**: Click the buttons for quick responses
    3. **Upload Documents**: Use the sidebar to upload supporting documents
    4. **Type Responses**: You can also type your answers in the chat
    
    **Supported Documents:**
    - Bank Statements (PDF)
    - Emirates ID (Image/PDF)
    - Resume/CV (PDF/Word)
    - Credit Reports (PDF)
    - Assets/Liabilities (Excel)
    
    **Need Help?** Just type your question and I'll assist you!
    """)

def process_chat_message(user_message: str, conversation_history: List[Dict], conversation_state: Dict) -> Dict:
    """Process chat message through the API"""
    
    try:
        # CRITICAL FIX: Increase timeout for completion conversations
        # LLM-generated economic enablement recommendations can take 60+ seconds
        current_step = conversation_state.get("current_step", "")
        is_completion = (current_step == "completion" or 
                        conversation_state.get("processing_status") == "completion_chat" or
                        conversation_state.get("eligibility_result") is not None)
        
        # Use longer timeout for completion conversations with LLM
        timeout_seconds = 120 if is_completion else 60  # 2 minutes for completion, 1 minute for others
        
        response = requests.post(
            f"{API_BASE}/conversation/message",
            json={
                "message": user_message,
                "conversation_history": conversation_history,
                "conversation_state": conversation_state
            },
            timeout=timeout_seconds
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {
                "message": "I'm having trouble processing your request right now. Could you please try again?",
                "state_update": {}
            }
            
    except Exception as e:
        # Fallback to rule-based responses if API is down
        return generate_fallback_response(user_message, conversation_state)

def process_document_upload(uploaded_file) -> str:
    """Process uploaded document and return conversational response"""
    
    try:
        # Save file temporarily
        file_path = f"data/uploads/{uploaded_file.name}"
        os.makedirs("data/uploads", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Get current conversation state and make it JSON serializable
        conversation_state = st.session_state.get("conversation_state", {})
        
        # Create a clean, serializable version of conversation state
        serializable_state = {
            "current_step": conversation_state.get("current_step", "document_collection"),
            "collected_data": conversation_state.get("collected_data", {}),
            "uploaded_documents": conversation_state.get("uploaded_documents", [])
        }
        
        # Ensure all values in collected_data are serializable
        clean_collected_data = {}
        for key, value in serializable_state["collected_data"].items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                clean_collected_data[key] = value
            else:
                # Convert non-serializable objects to string
                clean_collected_data[key] = str(value)
        
        serializable_state["collected_data"] = clean_collected_data
        
        # Send to document processing API
        with open(file_path, "rb") as f:
            files = {"file": (uploaded_file.name, f, uploaded_file.type)}
            data = {
                "file_type": get_document_type(uploaded_file.name),
                "conversation_state": json.dumps(serializable_state)
            }
            response = requests.post(
                f"{API_BASE}/conversation/upload-document",
                files=files,
                data=data,
                timeout=60
            )
        
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass
        
        if response.status_code == 200:
            result = response.json()
            
            # Update conversation state if provided
            if "state_update" in result:
                current_state = st.session_state.get("conversation_state", {})
                current_state.update(result["state_update"])
                st.session_state.conversation_state = current_state
            
            return result.get("message", f"Thank you for uploading {uploaded_file.name}. I'm processing it now...")
        else:
            error_detail = ""
            try:
                error_response = response.json()
                error_detail = error_response.get("detail", "")
            except:
                error_detail = response.text
            
            return f"I received your {uploaded_file.name} but had trouble processing it: {error_detail}. Could you try uploading it again or provide the information manually?"
            
    except Exception as e:
        # Clean up temporary file if it exists
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
            
        return f"I had trouble with your {uploaded_file.name} upload: {str(e)}. Please try again or tell me the information manually."

def get_document_type(filename: str) -> str:
    """Determine document type from filename"""
    filename_lower = filename.lower()
    
    if any(word in filename_lower for word in ['bank', 'statement', 'transaction']):
        return "bank_statement"
    elif any(word in filename_lower for word in ['emirates', 'id', 'identity']):
        return "emirates_id" 
    elif any(word in filename_lower for word in ['resume', 'cv', 'curriculum']):
        return "resume"
    elif any(word in filename_lower for word in ['credit', 'report', 'score']):
        return "credit_report"
    elif any(word in filename_lower for word in ['asset', 'liability', 'wealth']):
        return "assets_liabilities"
    else:
        return "other"

def generate_fallback_response(user_message: str, conversation_state: Dict) -> Dict:
    """Generate fallback response when API is unavailable"""
    
    current_step = conversation_state.get("current_step", "name_collection")
    message_lower = user_message.lower()
    collected_data = conversation_state.get("collected_data", {})
    
    if current_step == "name_collection":
        # Extract potential name from message
        words = user_message.split()
        if len(words) >= 2:
            collected_data["name"] = user_message
            return {
                "message": f"Nice to meet you, {user_message}! Now I need to verify your identity. Can you please upload your Emirates ID, or tell me your Emirates ID number?",
                "state_update": {
                    "current_step": "identity_verification",
                    "collected_data": collected_data
                }
            }
        else:
            return {
                "message": "Could you please provide your full name (first and last name)?",
                "state_update": {}
            }
    
    elif current_step == "identity_verification":
        if any(char.isdigit() for char in user_message):
            collected_data["emirates_id"] = user_message
            return {
                "message": "Thank you! Now let's talk about your employment situation. Are you currently employed, unemployed, self-employed, or retired?",
                "state_update": {
                    "current_step": "employment_inquiry",
                    "collected_data": collected_data
                }
            }
        else:
            return {
                "message": "I need your Emirates ID number for verification. Could you provide it or upload a photo of your Emirates ID?",
                "state_update": {}
            }
    
    elif current_step == "employment_inquiry":
        employment_status = "unemployed"  # Default
        if any(word in message_lower for word in ["employed", "working", "job"]) and "unemployed" not in message_lower:
            employment_status = "employed"
        elif any(word in message_lower for word in ["self", "business", "own"]):
            employment_status = "self_employed"
        elif any(word in message_lower for word in ["retired", "pension"]):
            employment_status = "retired"
        
        collected_data["employment_status"] = employment_status
        
        return {
            "message": f"I understand you are {employment_status.replace('_', ' ')}. What is your approximate monthly income in AED? If you have a recent bank statement, you can upload it for more accurate assessment.",
            "state_update": {
                "current_step": "income_assessment",
                "collected_data": collected_data
            }
        }
    
    elif current_step == "income_assessment":
        # Extract income amount from user message
        import re
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d{2})?\b', user_message.replace(",", ""))
        
        if numbers:
            # Take the largest number found (assuming it's the income)
            amounts = [float(num.replace(",", "")) for num in numbers]
            monthly_income = max(amounts)
        elif any(word in message_lower for word in ["zero", "nothing", "none", "no income"]):
            monthly_income = 0.0
        else:
            monthly_income = 0.0  # Default
        
        collected_data["monthly_income"] = monthly_income
        
        return {
            "message": f"Thank you. I've noted your monthly income as {monthly_income:,.0f} AED. Now, how many people are in your household (including yourself)? For example, if you live with your spouse and 2 children, that would be 4 people total.",
            "state_update": {
                "current_step": "family_details",
                "collected_data": collected_data
            }
        }
    
    elif current_step == "family_details":
        # Extract family size from user message
        import re
        numbers = re.findall(r'\b\d+\b', user_message)
        
        if numbers:
            family_size = int(numbers[0])
            collected_data["family_size"] = family_size
            
            return {
                "message": f"Got it - {family_size} people in your household. What's your current housing situation? Do you own your home, rent, or live with family?",
                "state_update": {
                    "current_step": "housing_situation",
                    "collected_data": collected_data
                }
            }
        else:
            return {
                "message": "Please tell me the number of people in your household. Just say a number like '3' or 'four people'.",
                "state_update": {}
            }
    
    elif current_step == "housing_situation":
        housing_status = "other"  # Default
        if any(word in message_lower for word in ["own", "owner", "bought", "mortgage"]):
            housing_status = "owned"
        elif any(word in message_lower for word in ["rent", "renting", "tenant", "lease"]):
            housing_status = "rented"
        elif any(word in message_lower for word in ["family", "parents", "relatives", "free"]):
            housing_status = "family"
        
        collected_data["housing_status"] = housing_status
        
        return {
            "message": "Perfect! I have the basic information I need. You can upload additional documents (bank statements, credit reports, etc.) to improve the accuracy of your assessment, or I can proceed with the eligibility evaluation now. What would you prefer?",
            "state_update": {
                "current_step": "document_collection",
                "collected_data": collected_data
            }
        }
    
    else:
        return {
            "message": "I'm collecting your information to assess your eligibility for social support. Please continue answering my questions, and feel free to upload any relevant documents.",
            "state_update": {}
        }

def show_final_results(final_decision: Dict):
    """Display final application results"""
    
    if not final_decision:
        return
    
    st.markdown("---")
    st.markdown("## 🎉 Application Complete!")
    
    decision = final_decision.get("decision", "pending")
    
    if decision == "approved":
        st.success("✅ Your application has been APPROVED!")
        
        support_amount = final_decision.get("support_amount", 0)
        st.metric("Monthly Support Amount", f"{support_amount:,.0f} AED")
        
        # Show breakdown
        if "breakdown" in final_decision:
            with st.expander("📊 Support Breakdown"):
                for item, amount in final_decision["breakdown"].items():
                    st.write(f"• {item}: {amount} AED")
    
    elif decision == "declined":
        st.error("❌ Your application has been declined.")
        
        if "reason" in final_decision:
            st.write(f"**Reason:** {final_decision['reason']}")
    
    else:
        st.info("⏳ Your application is under review.")
    
    # Show recommendations
    if "recommendations" in final_decision:
        st.markdown("### 🎯 Economic Enablement Recommendations")
        
        recommendations = final_decision["recommendations"]
        
        if "job_opportunities" in recommendations:
            st.markdown("#### 💼 Job Opportunities")
            for job in recommendations["job_opportunities"][:3]:
                st.write(f"• {job.get('title', 'Job Opportunity')} - {job.get('company', 'Various Companies')}")
        
        if "training_programs" in recommendations:
            st.markdown("#### 📚 Recommended Training")
            for program in recommendations["training_programs"][:3]:
                st.write(f"• {program.get('name', 'Training Program')} - {program.get('duration', 'Various Durations')}")

def process_chat_message_from_main(prompt: str, messages: List[Dict]) -> str:
    """Process chat message from main interface (compatibility function)"""
    
    try:
        conversation_state = st.session_state.get("conversation_state", {
            "current_step": "general_inquiry",
            "collected_data": {},
            "uploaded_documents": []
        })
        
        result = process_chat_message(prompt, messages, conversation_state)
        return result.get("message", "I'm here to help with your application questions.")
        
    except Exception as e:
        return generate_fallback_response(prompt, {}).get("message", "How can I help you today?") 