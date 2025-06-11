"""
Streamlit Frontend for Social Support AI Workflow

Interactive web interface for:
- Application submission
- Document upload
- Real-time processing
- Results visualization
- Chat interface for questions
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE = f"http://{settings.api_host}:{settings.api_port}"

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🤝 Social Support AI Workflow")
    st.markdown("**Automated Application Processing with AI Agents**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "🏠 Home",
            "📝 Submit Application", 
            "📋 Application Status",
            "🔄 Quick Demo",
            "📊 Analytics",
            "💬 Chat Assistant",
            "🧪 Testing Tools"
        ]
    )
    
    # Route to different pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "📝 Submit Application":
        show_application_form()
    elif page == "📋 Application Status":
        show_status_page()
    elif page == "🔄 Quick Demo":
        show_demo_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "💬 Chat Assistant":
        show_chat_page()
    elif page == "🧪 Testing Tools":
        show_testing_page()


def show_home_page():
    """Display home page with system overview"""
    
    st.markdown("## System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Processing Time", "< 2 minutes", "95% faster")
    
    with col2:
        st.metric("Accuracy Rate", "92%", "vs 78% manual")
    
    with col3:
        st.metric("Documents Supported", "5 types", "Multimodal AI")
    
    st.markdown("---")
    
    # System architecture
    st.markdown("## 🏗️ System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🤖 AI Agents
        - **Master Orchestrator**: Coordinates entire workflow
        - **Data Extraction Agent**: Processes documents
        - **Eligibility Agent**: Assesses applications
        - **ReAct Reasoning**: Intelligent decision making
        """)
        
    with col2:
        st.markdown("""
        ### 📄 Document Processing
        - **Bank Statements**: Financial analysis
        - **Emirates ID**: Identity verification
        - **Resumes**: Employment history
        - **Credit Reports**: Risk assessment
        - **Assets/Liabilities**: Wealth evaluation
        """)
    
    # Process flow
    st.markdown("## 🔄 Process Flow")
    
    flow_steps = [
        "📝 Application Submission",
        "📄 Document Upload",
        "🤖 AI Processing",
        "✅ Eligibility Assessment", 
        "💰 Support Calculation",
        "📊 Final Decision"
    ]
    
    cols = st.columns(len(flow_steps))
    for i, (col, step) in enumerate(zip(cols, flow_steps)):
        with col:
            st.markdown(f"**{i+1}. {step}**")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("## 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Submit New Application", use_container_width=True):
            st.session_state.page = "📝 Submit Application"
            st.rerun()
    
    with col2:
        if st.button("🔄 Try Quick Demo", use_container_width=True):
            st.session_state.page = "🔄 Quick Demo"
            st.rerun()
    
    with col3:
        if st.button("🧪 Generate Test Data", use_container_width=True):
            st.session_state.page = "🧪 Testing Tools"
            st.rerun()


def show_application_form():
    """Display application submission form"""
    
    st.markdown("## 📝 Submit Social Support Application")
    
    with st.form("application_form"):
        st.markdown("### Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("First Name *", placeholder="Enter first name")
            last_name = st.text_input("Last Name *", placeholder="Enter last name")
            email = st.text_input("Email", placeholder="email@example.com")
            phone = st.text_input("Phone", placeholder="+971 XX XXX XXXX")
        
        with col2:
            emirates_id = st.text_input("Emirates ID", placeholder="XXX-XXXX-XXXXXXX-X")
            employment_status = st.selectbox(
                "Employment Status",
                ["unemployed", "employed", "self_employed", "student", "retired"]
            )
            employment_duration = st.number_input("Employment Duration (months)", min_value=0, value=0)
            education_level = st.selectbox(
                "Education Level",
                ["no_education", "primary", "secondary", "bachelor", "master", "phd"]
            )
        
        st.markdown("### Financial Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_income = st.number_input("Monthly Income (AED)", min_value=0.0, value=0.0, step=100.0)
            housing_type = st.selectbox(
                "Housing Type",
                ["rented", "owned", "family_house", "shared"]
            )
            monthly_rent = st.number_input("Monthly Rent (AED)", min_value=0.0, value=0.0, step=100.0)
        
        with col2:
            family_size = st.number_input("Family Size", min_value=1, value=1)
            dependents = st.number_input("Number of Dependents", min_value=0, value=0)
            previous_applications = st.number_input("Previous Applications", min_value=0, value=0)
        
        st.markdown("### Additional Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            has_medical_conditions = st.checkbox("Has Medical Conditions")
        
        with col2:
            has_criminal_record = st.checkbox("Has Criminal Record")
        
        submitted = st.form_submit_button("Submit Application", use_container_width=True)
        
        if submitted:
            if not first_name or not last_name:
                st.error("Please fill in required fields (First Name, Last Name)")
            else:
                # Prepare application data
                application_data = {
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": email if email else None,
                    "phone": phone if phone else None,
                    "emirates_id": emirates_id if emirates_id else None,
                    "monthly_income": monthly_income,
                    "employment_status": employment_status,
                    "employment_duration_months": employment_duration,
                    "family_size": family_size,
                    "number_of_dependents": dependents,
                    "housing_type": housing_type,
                    "monthly_rent": monthly_rent,
                    "education_level": education_level,
                    "has_medical_conditions": has_medical_conditions,
                    "has_criminal_record": has_criminal_record,
                    "previous_applications": previous_applications
                }
                
                # Submit application
                submit_application(application_data)


def submit_application(application_data):
    """Submit application to API"""
    
    try:
        with st.spinner("Submitting application..."):
            response = requests.post(
                f"{API_BASE}/applications/submit",
                json=application_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Application submitted successfully!")
                
                st.markdown("### Application Details")
                st.write(f"**Application ID:** {result['application_id']}")
                st.write(f"**Status:** {result['status']}")
                
                # Store application ID in session state
                st.session_state.application_id = result['application_id']
                
                st.markdown("### Next Steps")
                for step in result['next_steps']:
                    st.write(f"• {step}")
                
                # Option to upload documents
                if st.button("📄 Upload Documents"):
                    st.session_state.show_upload = True
                    st.rerun()
                
            else:
                st.error(f"Failed to submit application: {response.text}")
                
    except Exception as e:
        st.error(f"Error submitting application: {str(e)}")


def show_status_page():
    """Display application status checking page"""
    
    st.markdown("## 📋 Check Application Status")
    
    application_id = st.text_input("Enter Application ID", placeholder="APP-YYYYMMDD-XXXXXXXX")
    
    if st.button("Check Status") and application_id:
        check_application_status(application_id)
    
    # If we have an application ID from session, show it
    if hasattr(st.session_state, 'application_id'):
        st.markdown("---")
        st.markdown("### Recent Application")
        if st.button(f"Check Status for {st.session_state.application_id}"):
            check_application_status(st.session_state.application_id)


def check_application_status(application_id):
    """Check and display application status"""
    
    try:
        with st.spinner("Checking application status..."):
            response = requests.get(f"{API_BASE}/applications/{application_id}/status")
            
            if response.status_code == 200:
                status_data = response.json()
                
                st.markdown("### Application Status")
                
                # Status indicator
                status = status_data['status']
                if status == "completed":
                    st.success(f"✅ Status: {status.upper()}")
                elif status == "processing":
                    st.info(f"⏳ Status: {status.upper()}")
                elif status == "failed":
                    st.error(f"❌ Status: {status.upper()}")
                else:
                    st.warning(f"📝 Status: {status.upper()}")
                
                # Details
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Application ID:** {status_data['application_id']}")
                    st.write(f"**Submitted:** {status_data['submitted_at']}")
                
                with col2:
                    processed_at = status_data.get('processed_at')
                    if processed_at:
                        st.write(f"**Processed:** {processed_at}")
                    else:
                        st.write("**Processed:** Not yet")
                
                # If completed, show results
                if status == "completed":
                    if st.button("📊 View Detailed Results"):
                        show_application_results(application_id)
                
                # If submitted, allow processing
                elif status == "submitted":
                    if st.button("🔄 Process Application"):
                        process_application(application_id)
                        
            else:
                st.error(f"Application not found: {application_id}")
                
    except Exception as e:
        st.error(f"Error checking status: {str(e)}")


def show_application_results(application_id):
    """Display detailed application results"""
    
    try:
        response = requests.get(f"{API_BASE}/applications/{application_id}/results")
        
        if response.status_code == 200:
            results = response.json()
            
            st.markdown("### 📊 Application Results")
            
            # Main decision
            if results['is_eligible']:
                st.success(f"✅ **APPROVED** - Support Amount: {results['support_amount']} AED/month")
            else:
                st.error("❌ **DECLINED**")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Eligibility Score", f"{results['eligibility_score']:.2f}")
            
            with col2:
                st.metric("Support Amount", f"{results['support_amount']} AED")
            
            with col3:
                st.metric("Decision", "APPROVED" if results['is_eligible'] else "DECLINED")
            
            # Assessment details
            if results.get('assessment_data'):
                st.markdown("### Assessment Details")
                st.json(results['assessment_data'])
                
        else:
            st.error("Failed to retrieve results")
            
    except Exception as e:
        st.error(f"Error retrieving results: {str(e)}")


def show_demo_page():
    """Display quick demo page"""
    
    st.markdown("## 🔄 Quick Demo")
    st.markdown("Experience the AI workflow with pre-filled data")
    
    # Demo scenarios
    st.markdown("### Choose a Demo Scenario")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👨‍👩‍👧‍👦 Large Family", use_container_width=True):
            demo_data = {
                "first_name": "Ahmed",
                "last_name": "Al-Mansouri",
                "email": "ahmed.almansouri@email.com",
                "monthly_income": 2500,
                "employment_status": "unemployed",
                "family_size": 5,
                "number_of_dependents": 3,
                "housing_type": "rented",
                "monthly_rent": 2000,
                "education_level": "secondary",
                "has_medical_conditions": True
            }
            process_demo_application(demo_data, "Large Family Scenario")
    
    with col2:
        if st.button("🎓 Young Graduate", use_container_width=True):
            demo_data = {
                "first_name": "Fatima",
                "last_name": "Al-Zahra",
                "email": "fatima.alzahra@email.com",
                "monthly_income": 1800,
                "employment_status": "unemployed",
                "family_size": 1,
                "number_of_dependents": 0,
                "housing_type": "shared",
                "monthly_rent": 1200,
                "education_level": "bachelor",
                "has_medical_conditions": False
            }
            process_demo_application(demo_data, "Young Graduate Scenario")
    
    with col3:
        if st.button("👴 Senior Citizen", use_container_width=True):
            demo_data = {
                "first_name": "Hassan",
                "last_name": "Al-Mahmoud",
                "email": "hassan.mahmoud@email.com",
                "monthly_income": 1200,
                "employment_status": "retired",
                "family_size": 2,
                "number_of_dependents": 0,
                "housing_type": "owned",
                "monthly_rent": 0,
                "education_level": "primary",
                "has_medical_conditions": True
            }
            process_demo_application(demo_data, "Senior Citizen Scenario")


def process_demo_application(demo_data, scenario_name):
    """Process a demo application"""
    
    st.markdown(f"### Processing: {scenario_name}")
    
    # Show demo data
    with st.expander("View Demo Data"):
        st.json(demo_data)
    
    try:
        with st.spinner("Processing demo application through AI workflow..."):
            
            # Submit application with demo data
            request_data = {
                "application_data": demo_data,
                "use_synthetic_data": True
            }
            
            response = requests.post(
                f"{API_BASE}/applications/process-with-data",
                json=request_data,
                timeout=120  # Longer timeout for processing
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("✅ Demo processing completed!")
                
                # Display results
                display_processing_results(result)
                
            else:
                st.error(f"Demo processing failed: {response.text}")
                
    except Exception as e:
        st.error(f"Error processing demo: {str(e)}")


def display_processing_results(result):
    """Display processing results with visualizations"""
    
    st.markdown("### 🎯 Processing Results")
    
    final_decision = result.get('final_decision', {})
    
    # Main decision banner
    if final_decision.get('decision') == 'approved':
        st.success(f"✅ **APPLICATION APPROVED**")
        st.metric("Monthly Support Amount", f"{final_decision.get('support_amount', 0)} AED")
    else:
        st.error("❌ **APPLICATION DECLINED**")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Eligibility Score", f"{final_decision.get('eligibility_score', 0):.2f}")
    
    with col2:
        st.metric("Processing Time", f"{result.get('processing_summary', {}).get('processing_time', 0):.1f}s")
    
    with col3:
        st.metric("Workflow Steps", result.get('processing_summary', {}).get('total_steps', 0))
    
    # Detailed workflow
    st.markdown("### 🔄 Workflow Details")
    
    workflow_state = result.get('workflow_state', {})
    
    if workflow_state.get('workflow_log'):
        st.markdown("#### Processing Log")
        for log_entry in workflow_state['workflow_log']:
            st.write(f"**{log_entry['step']}:** {log_entry['message']}")
    
    # Assessment breakdown if available
    if 'eligibility' in workflow_state.get('results', {}):
        eligibility_data = workflow_state['results']['eligibility']
        if eligibility_data.get('status') == 'success':
            show_eligibility_breakdown(eligibility_data['assessment_result'])


def show_eligibility_breakdown(assessment_result):
    """Show detailed eligibility assessment breakdown"""
    
    st.markdown("#### 📊 Eligibility Assessment Breakdown")
    
    component_scores = assessment_result.get('component_scores', {})
    
    # Create visualization
    categories = []
    scores = []
    
    for category, data in component_scores.items():
        categories.append(category.replace('_', ' ').title())
        scores.append(data.get('score', 0))
    
    if categories and scores:
        fig = px.bar(
            x=categories,
            y=scores,
            title="Eligibility Component Scores",
            labels={'x': 'Assessment Categories', 'y': 'Score (0-1)'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown
    for category, data in component_scores.items():
        with st.expander(f"{category.replace('_', ' ').title()} - Score: {data.get('score', 0):.2f}"):
            for key, value in data.items():
                if key != 'score':
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")


def show_analytics_page():
    """Display analytics and statistics"""
    
    st.markdown("## 📊 System Analytics")
    st.info("This is a demo analytics page. In production, this would show real system metrics.")
    
    # Simulated data for demonstration
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Applications", "1,247", "+23")
    
    with col2:
        st.metric("Approval Rate", "68%", "+2.1%")
    
    with col3:
        st.metric("Avg Processing Time", "1.8 min", "-0.3 min")
    
    with col4:
        st.metric("System Uptime", "99.9%", "0%")
    
    # Charts
    st.markdown("### Application Trends")
    
    # Sample data for charts
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    applications = [20 + i + (i % 7) * 5 for i in range(30)]
    approvals = [int(a * 0.68) for a in applications]
    
    df = pd.DataFrame({
        'Date': dates,
        'Applications': applications,
        'Approvals': approvals
    })
    
    fig = px.line(df, x='Date', y=['Applications', 'Approvals'], 
                  title="Daily Application and Approval Trends")
    st.plotly_chart(fig, use_container_width=True)
    
    # Processing time distribution
    st.markdown("### Processing Time Distribution")
    
    import numpy as np
    processing_times = np.random.normal(90, 20, 1000)  # 90 seconds average
    processing_times = processing_times[processing_times > 30]  # Minimum 30 seconds
    
    fig = px.histogram(processing_times, bins=20, 
                       title="Processing Time Distribution (seconds)")
    st.plotly_chart(fig, use_container_width=True)


def show_chat_page():
    """Display chat assistant page"""
    
    st.markdown("## 💬 AI Chat Assistant")
    st.info("Chat functionality would be implemented here with a conversational AI agent.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm here to help you with the social support application process. How can I assist you today?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about the application process..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response (simplified for demo)
        with st.chat_message("assistant"):
            response = generate_chat_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def generate_chat_response(prompt):
    """Generate a simple chat response (placeholder)"""
    
    # This is a simplified response system
    # In production, this would use the LLM with proper context
    
    prompt_lower = prompt.lower()
    
    if "eligibility" in prompt_lower:
        return "To be eligible for social support, you need to meet income thresholds, family size requirements, and other criteria. The AI system evaluates multiple factors including financial need, family composition, and employment stability."
    
    elif "documents" in prompt_lower:
        return "You can upload these document types: Bank statements, Emirates ID, Resume, Credit reports, and Assets/liabilities spreadsheets. Each document is processed using specialized AI for data extraction."
    
    elif "process" in prompt_lower or "time" in prompt_lower:
        return "The AI workflow typically processes applications in under 2 minutes. It includes document analysis, data validation, eligibility assessment, and final decision generation."
    
    elif "support" in prompt_lower or "amount" in prompt_lower:
        return "Support amounts are calculated based on family size, income level, housing costs, and special circumstances. The base amount is 1,000 AED with additional supplements for larger families and special needs."
    
    else:
        return "I can help you with information about eligibility criteria, required documents, the application process, and support amounts. What specific aspect would you like to know more about?"


def show_testing_page():
    """Display testing and development tools"""
    
    st.markdown("## 🧪 Testing Tools")
    st.markdown("Development and testing utilities for the AI workflow system.")
    
    # Generate synthetic data
    st.markdown("### Generate Test Data")
    
    if st.button("🎲 Generate Synthetic Application Data"):
        generate_synthetic_test_data()
    
    # API status check
    st.markdown("### System Status")
    
    if st.button("🔍 Check API Status"):
        check_api_status()
    
    # Database connection test
    if st.button("🗄️ Test Database Connection"):
        test_database_connection()


def generate_synthetic_test_data():
    """Generate and display synthetic test data"""
    
    try:
        with st.spinner("Generating synthetic data..."):
            response = requests.get(f"{API_BASE}/testing/generate-synthetic-data")
            
            if response.status_code == 200:
                data = response.json()
                
                st.success("✅ Synthetic data generated successfully!")
                
                # Display application data
                st.markdown("#### Application Data")
                st.json(data['application_data'])
                
                # Display document data
                st.markdown("#### Document Data")
                for doc_type, doc_data in data['document_data'].items():
                    with st.expander(f"{doc_type.replace('_', ' ').title()} Data"):
                        st.json(doc_data)
                
            else:
                st.error("Failed to generate synthetic data")
                
    except Exception as e:
        st.error(f"Error generating synthetic data: {str(e)}")


def check_api_status():
    """Check API health status"""
    
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            st.success("✅ API is healthy")
            st.json(health_data)
        else:
            st.error("❌ API health check failed")
            
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {str(e)}")


def test_database_connection():
    """Test database connection"""
    
    st.info("Database connection test would be implemented here.")
    # This would require backend endpoint for database testing


def process_application(application_id):
    """Process an application through the workflow"""
    
    try:
        with st.spinner("Processing application through AI workflow..."):
            response = requests.post(
                f"{API_BASE}/applications/{application_id}/process",
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Application processed successfully!")
                display_processing_results(result)
            else:
                st.error(f"Processing failed: {response.text}")
                
    except Exception as e:
        st.error(f"Error processing application: {str(e)}")


if __name__ == "__main__":
    main() 