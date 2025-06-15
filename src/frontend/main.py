"""
Streamlit Frontend for Social Support AI Workflow

Single-page interface with:
- Main chat interface for applications
- Quick access buttons for status lookup and demo scenarios
"""
import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import settings

# Configure Streamlit page
st.set_page_config(
    page_title="Social Support AI Workflow",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API base URL
API_BASE = f"http://{settings.api_host}:{settings.api_port}"

def main():
    """Main Streamlit application - single page design"""
    
    # Header with action buttons
    col1, col2, col3 = st.columns([4, 1, 1])
    
    with col1:
        st.title("🤝 Social Support AI Assistant")
        st.markdown("**Apply for financial support through our AI-powered conversation**")
    
    with col2:
        if st.button("📋 Check Status", use_container_width=True, type="secondary"):
            st.session_state.show_status_modal = True
    
    with col3:
        if st.button("🔄 New Chat", use_container_width=True, type="primary"):
            restart_chat()
    
    # Show status modal if requested
    if st.session_state.get("show_status_modal", False):
        show_status_lookup_modal()
    
    # Main chat interface
    show_main_chat_interface()


def show_main_chat_interface():
    """Display the main chat interface"""
    
    st.markdown("---")
    
    # Import and use the chat interface from the same directory
    try:
        # Import the chat interface function directly
        import sys
        import os
        
        # Add the frontend directory to path
        frontend_dir = os.path.dirname(__file__)
        if frontend_dir not in sys.path:
            sys.path.insert(0, frontend_dir)
        
        from chat_interface import show_chat_interface
        show_chat_interface()
        
    except ImportError as e:
        st.error(f"Chat interface import error: {str(e)}")
        st.markdown("**Troubleshooting:** Please ensure chat_interface.py is in the same directory.")
        show_fallback_chat_interface()
    except Exception as e:
        st.error(f"Chat interface error: {str(e)}")
        show_fallback_chat_interface()


def show_status_lookup_modal():
    """Show application status lookup in a modal-style container"""
    
    with st.container():
        st.markdown("## 📋 Application Status Lookup")
        st.markdown("**Use your 15-digit Emirates ID to find your application**")
        
        emirates_id = st.text_input(
            "Emirates ID:",
            placeholder="784199012345678",
            key="status_emirates_input",
            help="Enter your 15-digit Emirates ID number"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔍 Find Application", key="emirates_search_btn", use_container_width=True):
                if emirates_id:
                    lookup_data = {"emirates_id": emirates_id}
                    result = lookup_application_by_method(lookup_data)
                    
                    if result and result.get("found"):
                        display_application_status(result["application"])
                    else:
                        st.error("❌ Application not found. Please check your Emirates ID.")
                else:
                    st.warning("Please enter your Emirates ID")
        
        with col2:
            if st.button("❌ Close", key="close_status", use_container_width=True):
                st.session_state.show_status_modal = False
                st.rerun()


def restart_chat():
    """Restart the chat conversation"""
    
    # Clear chat-related session state but reinitialize properly
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
    
    # Clear modal states
    st.session_state.show_status_modal = False
    
    st.rerun()


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


def display_application_status(app_data):
    """Display application status information"""
    
    st.success("✅ Application Found!")
    
    # Main status card
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### 👤 {app_data['full_name']}")
            st.markdown(f"**Reference:** `{app_data['reference_number']}`")
        
        with col2:
            status = app_data['status']
            if status == 'completed':
                st.markdown("### ✅ Status")
                st.success("COMPLETED")
            elif status == 'under_review':
                st.markdown("### ⏳ Status")
                st.warning("UNDER REVIEW")
            else:
                st.markdown("### 📝 Status")
                st.info(status.upper())
        
        with col3:
            if app_data.get('is_eligible'):
                st.markdown("### 💰 Support")
                st.success(f"{app_data.get('support_amount', 0):.0f} AED/month")
            else:
                st.markdown("### ❌ Support")
                st.error("Not Eligible")
    
    # Timeline
    if app_data.get('submitted_at'):
        submitted_date = datetime.fromisoformat(app_data['submitted_at'].replace('Z', '+00:00'))
        st.write(f"**Submitted:** {submitted_date.strftime('%B %d, %Y at %I:%M %p')}")


def show_fallback_chat_interface():
    """Fallback chat interface if the main one is not available"""
    
    st.markdown("### 🤖 Simple Chat Interface")
    st.markdown("*Note: This is a simplified version. For full features, please ensure the chat interface is properly installed.*")
    
    # Initialize session state for chat
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm your Social Support AI Assistant. I'll help you apply for financial support. Let's start - what's your full name?"}
        ]
    
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = {
            "current_step": "name_collection",
            "collected_data": {},
            "application_id": None
        }
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_simple_chat_response(prompt, st.session_state.chat_state)
                st.write(response["message"])
                
                # Update state
                if "state_update" in response:
                    st.session_state.chat_state.update(response["state_update"])
                
                # Add assistant message
                st.session_state.chat_messages.append({"role": "assistant", "content": response["message"]})


def generate_simple_chat_response(user_input: str, state: dict) -> dict:
    """Generate a simple chat response (fallback implementation)"""
    
    current_step = state.get("current_step", "name_collection")
    collected_data = state.get("collected_data", {})
    
    # Simple state machine for basic conversation
    if current_step == "name_collection":
        collected_data["name"] = user_input
        return {
            "message": f"Nice to meet you, {user_input}! What's your Emirates ID number?",
            "state_update": {
                "current_step": "emirates_id_collection",
                "collected_data": collected_data
            }
        }
    
    elif current_step == "emirates_id_collection":
        collected_data["emirates_id"] = user_input
        return {
            "message": "Thank you! What's your current employment status? (employed, unemployed, retired, etc.)",
            "state_update": {
                "current_step": "employment_collection",
                "collected_data": collected_data
            }
        }
    
    elif current_step == "employment_collection":
        collected_data["employment_status"] = user_input
        return {
            "message": "Got it! What's your monthly income in AED?",
            "state_update": {
                "current_step": "income_collection",
                "collected_data": collected_data
            }
        }
    
    elif current_step == "income_collection":
        try:
            collected_data["monthly_income"] = float(user_input)
            return {
                "message": "Thank you! How many people are in your family (including yourself)?",
                "state_update": {
                    "current_step": "family_size_collection",
                    "collected_data": collected_data
                }
            }
        except ValueError:
            return {
                "message": "Please enter a valid number for your monthly income.",
                "state_update": state
            }
    
    elif current_step == "family_size_collection":
        try:
            collected_data["family_size"] = int(user_input)
            
            # Simple eligibility check
            monthly_income = collected_data.get("monthly_income", 0)
            family_size = collected_data.get("family_size", 1)
            income_per_person = monthly_income / family_size
            
            if income_per_person < 3000:  # Simple threshold
                support_amount = max(500, (3000 - income_per_person) * family_size * 0.5)
                message = f"Great news! Based on your information, you appear to be eligible for social support of approximately {support_amount:.0f} AED per month. This is a preliminary assessment. For a complete application, please use the full chat interface or contact our office."
            else:
                message = "Based on your information, you may not qualify for direct financial support, but you might be eligible for other assistance programs. Please contact our office for a complete assessment."
            
            return {
                "message": message,
                "state_update": {
                    "current_step": "completed",
                    "collected_data": collected_data
                }
            }
        except ValueError:
            return {
                "message": "Please enter a valid number for family size.",
                "state_update": state
            }
    
    else:
        return {
            "message": "Thank you for using our service! If you'd like to start a new application, please click the 'New Chat' button.",
            "state_update": state
        }


if __name__ == "__main__":
    main() 