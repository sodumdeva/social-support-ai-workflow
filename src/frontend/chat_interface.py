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
    
    # CRITICAL: Check if application is completed and show status
    application_completed = st.session_state.conversation_state.get("application_complete", False)
    processing_status = st.session_state.conversation_state.get("processing_status", "")
    
    if application_completed or processing_status == "completed":
        st.success("🎉 **Application Completed Successfully!**")
        
        # Show the last completion message if available
        if hasattr(st.session_state, 'last_response') and st.session_state.last_response:
            st.info("📋 **Final Assessment:**")
            st.markdown(st.session_state.last_response)
        
        # Add restart option
        if st.button("🔄 Start New Application", key="restart_after_completion"):
            restart_conversation()
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.conversation_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Interactive buttons section based on current step
    show_interactive_buttons()
    
    # Show current progress and status check in sidebar
    with st.sidebar:
        # Application Status Check section
        st.markdown("### 📋 Check Application Status")
        st.markdown("Use your Emirates ID to find your application:")
        
        emirates_id = st.text_input(
            "Emirates ID:",
            placeholder="784199012345678",
            key="sidebar_emirates_input",
            help="Enter your 15-digit Emirates ID number"
        )
        
        if st.button("🔍 Find Application", key="sidebar_search_btn", use_container_width=True):
            if emirates_id:
                lookup_data = {"emirates_id": emirates_id}
                result = lookup_application_by_method(lookup_data)
                
                if result and result.get("found"):
                    display_application_status_sidebar(result["application"])
                else:
                    st.error("❌ Application not found. Please check your Emirates ID.")
            else:
                st.warning("Please enter your Emirates ID")
        
        st.markdown("---")
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
            if st.button("JOHN CITIZEN", key="btn_name_1", help="Test name 1"):
                handle_button_click("JOHN CITIZEN")
        
        with col2:
            if st.button("Fatima Al Zahra", key="btn_name_2", help="Test name 2"):
                handle_button_click("Fatima Al Zahra")
    
    # Emirates ID Presets (small buttons)
    elif current_step == "identity_verification":
        st.markdown("##### Quick options:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("784-1985-7654321-1", key="btn_id_1", help="Test Emirates ID 1"):
                handle_button_click("784-1985-7654321-1")
        
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
    
    # Income Assessment Buttons
    elif current_step == "income_assessment":
        st.markdown("### 💰 Select Your Monthly Salary Range (AED):")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💵 <1000", use_container_width=True, key="btn_income_low"):
                handle_button_click("700")
        
        with col2:
            if st.button("💰 1,000 - 5,000", use_container_width=True, key="btn_income_mid"):
                handle_button_click("3000")
        
        with col3:
            if st.button("💎 5,000 - 10,000", use_container_width=True, key="btn_income_high"):
                handle_button_click("8000")
        
        with col4:
            if st.button("🏆 10,000+", use_container_width=True, key="btn_income_very_high"):
                handle_button_click("12000")
        
        # Additional quick options
        st.markdown("##### Quick options:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("No Income", key="btn_no_income", help="Currently no income"):
                handle_button_click("0")
        
        with col2:
            if st.button("5,000 AED", key="btn_income_5k", help="Exact amount"):
                handle_button_click("5000")
        
        with col3:
            if st.button("10,000 AED", key="btn_income_10k", help="Exact amount"):
                handle_button_click("10000")
    
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
        
        # Inline file uploader
        st.markdown("**Upload your documents here:**")
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['pdf', 'jpg', 'jpeg', 'png', 'doc', 'docx', 'xls', 'xlsx'],
            key="inline_file_uploader",
            help="Supported formats: PDF, Images (JPG, PNG), Word documents, Excel files"
        )
        
        if uploaded_files:
            st.markdown("**Files ready to process:**")
            for uploaded_file in uploaded_files:
                st.write(f"📎 {uploaded_file.name} ({uploaded_file.size:,} bytes)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Process Documents", key="process_docs_inline_btn", use_container_width=True, type="primary"):
                    with st.spinner("Processing documents..."):
                        for uploaded_file in uploaded_files:
                            try:
                                response_message = process_document_upload(uploaded_file)
                                
                                # Add the document processing response to chat
                                st.session_state.conversation_messages.append({
                                    "role": "assistant", 
                                    "content": response_message
                                })
                                
                                # Update uploaded documents list in conversation state
                                if "uploaded_documents" not in st.session_state.conversation_state:
                                    st.session_state.conversation_state["uploaded_documents"] = []
                                
                                file_path = f"data/uploads/{uploaded_file.name}"
                                if file_path not in st.session_state.conversation_state["uploaded_documents"]:
                                    st.session_state.conversation_state["uploaded_documents"].append(file_path)
                                
                            except Exception as e:
                                error_message = f"Error processing {uploaded_file.name}: {str(e)}"
                                st.session_state.conversation_messages.append({
                                    "role": "assistant",
                                    "content": error_message
                                })
                        
                        st.success("✅ Documents processed!")
                        st.rerun()
            
            with col2:
                if st.button("✅ Proceed with Assessment", use_container_width=True, key="btn_proceed_after_upload"):
                    handle_button_click("proceed with assessment")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📤 Upload Documents Above", use_container_width=True, key="btn_upload_prompt"):
                    st.info("👆 Use the file uploader above to upload your documents")
            
            with col2:
                if st.button("✅ Proceed with Assessment", use_container_width=True, key="btn_proceed"):
                    handle_button_click("proceed with assessment")
        
        # Document type guidance
        with st.expander("📋 What documents should I upload?"):
            st.markdown("""
            **Identity Documents:**
            - Emirates ID (front/back)
            - Passport copy
            
            **Financial Documents:**
            - Bank statements (3-6 months)
            - Salary certificates
            - Employment letters
            
            **Supporting Documents:**
            - Family book/certificates
            - Medical reports (if applicable)
            - Housing contracts
            - Assets/liabilities statements
            """)
    
    # Add file uploader to other relevant steps as well
    elif current_step in ["identity_verification", "income_assessment", "housing_situation"]:
        st.markdown("---")
        st.markdown("### 📎 Optional: Upload Supporting Documents")
        
        uploaded_files = st.file_uploader(
            "Upload documents to support your application",
            accept_multiple_files=True,
            type=['pdf', 'jpg', 'jpeg', 'png', 'doc', 'docx', 'xls', 'xlsx'],
            key=f"optional_uploader_{current_step}",
            help="Upload relevant documents like Emirates ID, bank statements, etc."
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", key=f"process_optional_{current_step}", use_container_width=True):
                with st.spinner("Processing documents..."):
                    for uploaded_file in uploaded_files:
                        try:
                            response_message = process_document_upload(uploaded_file)
                            
                            # Add the document processing response to chat
                            st.session_state.conversation_messages.append({
                                "role": "assistant", 
                                "content": response_message
                            })
                            
                            # Update uploaded documents list in conversation state
                            if "uploaded_documents" not in st.session_state.conversation_state:
                                st.session_state.conversation_state["uploaded_documents"] = []
                            
                            file_path = f"data/uploads/{uploaded_file.name}"
                            if file_path not in st.session_state.conversation_state["uploaded_documents"]:
                                st.session_state.conversation_state["uploaded_documents"].append(file_path)
                            
                        except Exception as e:
                            error_message = f"Error processing {uploaded_file.name}: {str(e)}"
                            st.session_state.conversation_messages.append({
                                "role": "assistant",
                                "content": error_message
                            })
                    
                    st.success("✅ Documents processed!")
                    st.rerun()
    
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
        print(f"🔍 FRONTEND: Starting to process user input: '{prompt}'")
        
        response = process_chat_message(
            prompt, 
            st.session_state.conversation_messages,
            st.session_state.conversation_state
        )
        
        print(f"🔍 FRONTEND: Received response type: {type(response)}")
        print(f"🔍 FRONTEND: Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
        
        # Ensure response is not None
        if response is None:
            print("🔍 FRONTEND: Response is None, using fallback")
            response = {
                "message": "I'm having trouble processing your request right now. Could you please try again?",
                "state_update": {}
            }
        
        # CRITICAL: Check if this is a restart scenario
        state_update = response.get("state_update", {})
        if (state_update.get("current_step") == "name_collection" and 
            state_update.get("collected_data") == {} and
            "start" in prompt.lower() and "new" in prompt.lower()):
            
            print("🔍 FRONTEND: Detected restart scenario")
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
        
        # CRITICAL FIX: Handle application completion properly
        application_complete = response.get("application_complete", False)
        processing_status = state_update.get("processing_status", "")
        current_step = state_update.get("current_step", "")
        
        print(f"🔍 FRONTEND: Application complete: {application_complete}")
        print(f"🔍 FRONTEND: Processing status: {processing_status}")
        print(f"🔍 FRONTEND: Current step: {current_step}")
        
        # ENHANCED: Check for completion in multiple ways
        is_completion = (
            application_complete or 
            processing_status == "completed" or 
            processing_status == "completion_chat" or
            current_step == "completion" or
            ("approved" in response.get("message", "").lower() and "AED" in response.get("message", ""))
        )
        
        if is_completion:
            print("🔍 FRONTEND: Handling application completion")
            
            # Add the completion message FIRST
            completion_message = response.get("message", "")
            if completion_message:
                print(f"🔍 FRONTEND: Adding completion message ({len(completion_message)} chars)")
                st.session_state.conversation_messages.append({
                    "role": "assistant",
                    "content": completion_message
                })
                
                # CRITICAL: Force immediate display by updating session state
                st.session_state.last_response = completion_message
                st.session_state.application_completed = True
                
            else:
                print("🔍 FRONTEND: WARNING - No completion message in response")
            
            # Update conversation state with completion status
            print("🔍 FRONTEND: Updating conversation state")
            st.session_state.conversation_state.update(state_update)
            st.session_state.conversation_state["application_complete"] = True
            st.session_state.conversation_state["processing_status"] = "completed"
            
            # Show success message immediately
            st.success("🎉 Application completed successfully!")
            
            print("🔍 FRONTEND: Calling st.rerun() for completion")
            st.rerun()
            return
        
        # Add assistant response only if there is a message
        if response.get("message"):
            print(f"🔍 FRONTEND: Adding regular assistant message ({len(response['message'])} chars)")
            st.session_state.conversation_messages.append({
                "role": "assistant",
                "content": response["message"]
            })
        else:
            print("🔍 FRONTEND: No message in response")
        
        # Update conversation state
        print("🔍 FRONTEND: Updating conversation state with regular response")
        st.session_state.conversation_state.update(response.get("state_update", {}))
        
        # LEGACY: Check if application is complete (fallback)
        if response.get("application_complete"):
            print("🔍 FRONTEND: Legacy application complete check triggered")
            st.session_state.conversation_state["application_complete"] = True
            try:
                show_final_results(response.get("final_decision"))
            except Exception as e:
                print(f"🔍 FRONTEND: ERROR in legacy show_final_results: {str(e)}")
            
    except Exception as e:
        print(f"🔍 FRONTEND: EXCEPTION in handle_user_input: {str(e)}")
        print(f"🔍 FRONTEND: Exception type: {type(e).__name__}")
        import traceback
        print(f"🔍 FRONTEND: Traceback: {traceback.format_exc()}")
        
        st.session_state.conversation_messages.append({
            "role": "assistant",
            "content": f"I apologize, I encountered an error: {str(e)}. Let me try to help you in a different way."
        })
    
    print("🔍 FRONTEND: Calling final st.rerun()")
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
    3. **Upload Documents**: Use the file uploader that appears during the conversation
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
        
        # ENHANCED: Also check for document collection with "proceed" - this triggers full assessment
        is_document_proceed = (current_step == "document_collection" and 
                              "proceed" in user_message.lower())
        
        
        timeout_seconds = 300   # 3 minutes for regular conversation
        
        # DEBUG: Log the request being sent
        request_data = {
            "message": user_message,
            "conversation_history": conversation_history,
            "conversation_state": conversation_state
        }
        print(f"🔍 FRONTEND DEBUG: Sending request - Step: {current_step}, Message: '{user_message}', Timeout: {timeout_seconds}s")
        
        response = requests.post(
            f"{API_BASE}/conversation/message",
            json=request_data,
            timeout=timeout_seconds
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # DEBUG: Log the response received
            response_message = result.get("message", "")
            state_update = result.get("state_update", {})
            application_complete = result.get("application_complete", False)
            
            print(f"🔍 FRONTEND DEBUG: Received response - Status: {response.status_code}")
            print(f"   Message length: {len(response_message)} chars")
            print(f"   Application complete: {application_complete}")
            print(f"   New step: {state_update.get('current_step', 'unknown')}")
            print(f"   Processing status: {state_update.get('processing_status', 'unknown')}")
            
            return result
        else:
            print(f"🔍 FRONTEND DEBUG: API error - Status: {response.status_code}")
            return {
                "message": "I'm having trouble processing your request right now. Could you please try again?",
                "state_update": {}
            }
            
    except Exception as e:
        print(f"🔍 FRONTEND DEBUG: Exception occurred - {str(e)}")
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
                timeout=300
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

def lookup_application_by_method(lookup_data):
    """Lookup application using the flexible API endpoint"""
    try:
        response = requests.post(f"{API_BASE}/applications/lookup", json=lookup_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None

def display_application_status_sidebar(app_data):
    """Display application status information in sidebar"""
    
    st.success("✅ Application Found!")
    
    # Main info
    st.markdown(f"**👤 {app_data['full_name']}**")
    st.markdown(f"**Reference:** `{app_data['reference_number']}`")
    
    # Status
    status = app_data['status']
    if status == 'completed':
        st.success("✅ COMPLETED")
    elif status == 'under_review':
        st.warning("⏳ UNDER REVIEW")
    else:
        st.info(f"📝 {status.upper()}")
    
    # Support amount
    if app_data.get('is_eligible'):
        st.success(f"💰 {app_data.get('support_amount', 0):.0f} AED/month")
    else:
        st.error("❌ Not Eligible")
    
    # Timeline
    if app_data.get('submitted_at'):
        from datetime import datetime
        submitted_date = datetime.fromisoformat(app_data['submitted_at'].replace('Z', '+00:00'))
        st.write(f"**Submitted:** {submitted_date.strftime('%B %d, %Y')}") 