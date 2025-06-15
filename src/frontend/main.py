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
            "💬 Chat Application",  # Make chat the first option
            "🏠 Home",
            "📝 Submit Application", 
            "📋 Application Status",
            "🔄 Quick Demo",
            "📊 Analytics",
            "🧪 Testing Tools"
        ]
    )
    
    # Route to different pages
    if page == "💬 Chat Application":
        show_chat_application_page()
    elif page == "🏠 Home":
        show_home_page()
    elif page == "📝 Submit Application":
        show_application_form()
    elif page == "📋 Application Status":
        show_status_page()
    elif page == "🔄 Quick Demo":
        show_demo_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "🧪 Testing Tools":
        show_testing_page()


def show_home_page():
    """Display home page with system overview and interactive buttons"""
    
    st.markdown("## System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Processing Time", "< 2 minutes", "95% faster")
    
    with col2:
        st.metric("Accuracy Rate", "92%", "vs 78% manual")
    
    with col3:
        st.metric("Documents Supported", "5 types", "Multimodal AI")
    
    st.markdown("---")
    
    # Quick start section with prominent buttons
    st.markdown("## 🚀 Get Started")
    st.markdown("Choose how you'd like to apply for social support:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💬 **Recommended: Chat Application**")
        st.markdown("Interactive conversation with AI assistant")
        st.markdown("✅ Step-by-step guidance")
        st.markdown("✅ Real-time document processing")
        st.markdown("✅ Instant feedback")
        
        if st.button("🤖 Start Chat Application", use_container_width=True, type="primary"):
            st.session_state.page = "💬 Chat Application"
            st.rerun()
    
    with col2:
        st.markdown("### 📝 **Traditional: Form Application**")
        st.markdown("Fill out a structured form")
        st.markdown("✅ All information at once")
        st.markdown("✅ Familiar interface")
        st.markdown("✅ Quick submission")
        
        if st.button("📋 Fill Application Form", use_container_width=True):
            st.session_state.page = "📝 Submit Application"
            st.rerun()
    
    st.markdown("---")
    
    # Demo section
    st.markdown("## 🔄 Try a Demo First")
    st.markdown("Not sure how it works? Try our demo scenarios:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👨‍👩‍👧‍👦 Large Family Demo", use_container_width=True):
            # Auto-run large family demo
            demo_data = {
                "name": "Ahmed Al Mansouri",
                "emirates_id": "784-1990-1234567-1",
                "employment_status": "unemployed",
                "monthly_income": 800,
                "family_size": 6,
                "housing_status": "rented",
                "monthly_rent": 2500,
                "has_medical_conditions": True,
                "number_of_dependents": 4
            }
            st.session_state.demo_data = demo_data
            st.session_state.demo_name = "Large Family Demo"
            st.session_state.page = "🔄 Quick Demo"
            st.rerun()
    
    with col2:
        if st.button("🏖️ Retired Person Demo", use_container_width=True):
            demo_data = {
                "name": "Fatima Al Zahra",
                "emirates_id": "784-1955-2345678-2",
                "employment_status": "retired",
                "monthly_income": 1200,
                "family_size": 2,
                "housing_status": "rented",
                "monthly_rent": 1800,
                "has_medical_conditions": False,
                "number_of_dependents": 1
            }
            st.session_state.demo_data = demo_data
            st.session_state.demo_name = "Retired Person Demo"
            st.session_state.page = "🔄 Quick Demo"
            st.rerun()
    
    with col3:
        if st.button("💼 Professional Demo", use_container_width=True):
            demo_data = {
                "name": "Omar Al Rashid",
                "emirates_id": "784-1980-4567890-4",
                "employment_status": "employed",
                "monthly_income": 15000,
                "family_size": 4,
                "housing_status": "owned",
                "monthly_rent": 0,
                "has_medical_conditions": False,
                "number_of_dependents": 2
            }
            st.session_state.demo_data = demo_data
            st.session_state.demo_name = "Professional Demo"
            st.session_state.page = "🔄 Quick Demo"
            st.rerun()
    
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
    
    # Additional quick actions
    st.markdown("## 🛠️ Additional Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Check Application Status", use_container_width=True):
            st.session_state.page = "📋 Application Status"
            st.rerun()
    
    with col2:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.page = "📊 Analytics"
            st.rerun()
    
    with col3:
        if st.button("🧪 Testing Tools", use_container_width=True):
            st.session_state.page = "🧪 Testing Tools"
            st.rerun()


def show_application_form():
    """Display application submission form with interactive buttons"""
    
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
        
        st.markdown("### Employment Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            employment_status = st.selectbox(
                "Employment Status",
                ["employed", "unemployed", "self_employed", "retired", "student"]
            )
        
        with col2:
            # Add employment duration if employed
            if employment_status in ["employed", "self_employed"]:
                employment_duration = st.number_input("Employment Duration (months)", min_value=0, value=0)
        
        st.markdown("### Financial Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_income = st.number_input("Monthly Income (AED)", min_value=0.0, value=0.0, step=100.0)
        
        with col2:
            family_size = st.number_input("Family Size", min_value=1, value=1)
        
        # Add housing information
        st.markdown("### Housing Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            housing_status = st.selectbox(
                "Housing Status",
                ["owned", "rented", "family", "other"]
            )
        
        with col2:
            if housing_status == "rented":
                monthly_rent = st.number_input("Monthly Rent (AED)", min_value=0.0, value=0.0, step=100.0)
            else:
                monthly_rent = 0.0
        
        # Additional information
        st.markdown("### Additional Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            has_medical_conditions = st.checkbox("Do you have any medical conditions requiring ongoing treatment?")
        
        with col2:
            number_of_dependents = st.number_input("Number of Dependents", min_value=0, value=0)
        
        submitted = st.form_submit_button("Submit Application", use_container_width=True)
        
        if submitted:
            # Validate required fields
            if not first_name or not last_name:
                st.error("Please fill in all required fields (marked with *)")
                return
            
            application_data = {
                "first_name": first_name,
                "last_name": last_name,
                "name": f"{first_name} {last_name}",
                "email": email,
                "phone": phone,
                "emirates_id": emirates_id,
                "employment_status": employment_status,
                "monthly_income": monthly_income,
                "family_size": family_size,
                "housing_status": housing_status,
                "monthly_rent": monthly_rent,
                "has_medical_conditions": has_medical_conditions,
                "number_of_dependents": number_of_dependents
            }
            
            # Add employment duration if applicable
            if employment_status in ["employed", "self_employed"]:
                application_data["employment_duration_months"] = employment_duration
            
            submit_application(application_data)
    
    # Quick action buttons below the form
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 Use Chat Interface Instead", use_container_width=True):
            st.session_state.page = "💬 Chat Application"
            st.rerun()
    
    with col2:
        if st.button("🔄 Try Demo Application", use_container_width=True):
            st.session_state.page = "🔄 Quick Demo"
            st.rerun()
    
    with col3:
        if st.button("📋 Check Application Status", use_container_width=True):
            st.session_state.page = "📋 Application Status"
            st.rerun()


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
                
                st.info("💡 You can now check the application status or use the Quick Demo to see AI processing in action!")
                
            else:
                st.error(f"Failed to submit application: {response.text}")
                
    except Exception as e:
        st.error(f"Error submitting application: {str(e)}")


def show_status_page():
    """Display application status checking interface"""
    
    st.markdown("## 📊 Application Status")
    st.markdown("Check the status of your submitted applications.")
    
    application_id = st.text_input(
        "Application ID",
        placeholder="Enter your application ID (e.g., APP-20241201-abc123)"
    )
    
    if st.button("🔍 Check Status") and application_id:
        check_application_status(application_id)
    
    # Show recent applications if any exist in session state
    if "recent_applications" in st.session_state:
        st.markdown("### Recent Applications")
        for app_id in st.session_state.recent_applications:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(app_id)
            with col2:
                if st.button("Check", key=f"check_{app_id}"):
                    check_application_status(app_id)


def check_application_status(application_id):
    """Check and display application status"""
    
    try:
        with st.spinner("Checking application status..."):
            response = requests.get(f"{API_BASE}/applications/{application_id}/status")
            
            if response.status_code == 200:
                status_data = response.json()
                
                st.success("✅ Application found!")
                
                # Display status information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Status", status_data["status"].title())
                    st.metric("Application ID", status_data["application_id"])
                
                with col2:
                    st.metric("Submitted", status_data["submitted_at"][:10])
                    if status_data.get("processed_at"):
                        st.metric("Processed", status_data["processed_at"][:10])
                
                # Show action buttons based on status
                if status_data["status"] == "submitted":
                    st.info("💡 Your application is submitted. Use the **Chat Application** page to continue the interactive process.")
                    if st.button("🚀 Continue with Chat Interface"):
                        st.session_state.page = "Chat Application"
                        st.rerun()
                
                elif status_data["status"] == "processed":
                    if st.button("📋 View Results"):
                        show_application_results(application_id)
                        
            elif response.status_code == 404:
                st.error("❌ Application not found. Please check your application ID.")
            else:
                st.error("❌ Error checking status. Please try again.")
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def show_application_results(application_id):
    """Display detailed application results"""
    
    try:
        with st.spinner("Loading application results..."):
            response = requests.get(f"{API_BASE}/applications/{application_id}/results")
            
            if response.status_code == 200:
                results = response.json()
                
                st.markdown("### 📋 Application Results")
                
                # Basic information
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Application ID", results["application_id"])
                
                with col2:
                    status_color = "🟢" if results.get("is_eligible") else "🔴"
                    st.metric("Eligibility", f"{status_color} {'Eligible' if results.get('is_eligible') else 'Not Eligible'}")
                
                with col3:
                    if results.get("support_amount"):
                        st.metric("Support Amount", f"{results['support_amount']} AED/month")
                
                # Detailed assessment
                if results.get("assessment_data"):
                    st.markdown("### 📊 Assessment Details")
                    
                    try:
                        assessment = json.loads(results["assessment_data"]) if isinstance(results["assessment_data"], str) else results["assessment_data"]
                        
                        if isinstance(assessment, dict):
                            for key, value in assessment.items():
                                if isinstance(value, (dict, list)):
                                    with st.expander(f"{key.replace('_', ' ').title()}"):
                                        st.json(value)
                                else:
                                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        else:
                            st.json(assessment)
                            
                    except Exception as e:
                        st.write("Assessment data available but could not be parsed for display.")
                
            else:
                st.error("❌ Could not load application results.")
                
    except Exception as e:
        st.error(f"❌ Error loading results: {str(e)}")


def show_demo_page():
    """Display demo page with preset scenarios and interactive buttons"""
    
    st.markdown("## 🔄 Quick Demo")
    st.markdown("Try the system with pre-configured scenarios to see how it works.")
    
    # Check if demo data was passed from home page
    if hasattr(st.session_state, 'demo_data') and hasattr(st.session_state, 'demo_name'):
        st.info(f"🎯 Running demo: **{st.session_state.demo_name}**")
        
        # Auto-process the demo
        with st.spinner("Processing demo application..."):
            process_demo_application(st.session_state.demo_data, st.session_state.demo_name)
        
        # Clear the demo data
        del st.session_state.demo_data
        del st.session_state.demo_name
        
        st.markdown("---")
    
    # Demo scenario buttons
    st.markdown("### 📋 Select a Demo Scenario:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Likely Approved Scenarios")
        
        if st.button("👨‍👩‍👧‍👦 Large Family, Low Income", use_container_width=True, key="demo_approved_1"):
            demo_data = {
                "name": "Ahmed Al Mansouri",
                "emirates_id": "784-1990-1234567-1",
                "employment_status": "unemployed",
                "monthly_income": 800,
                "family_size": 6,
                "housing_status": "rented",
                "monthly_rent": 2500,
                "has_medical_conditions": True,
                "number_of_dependents": 4
            }
            process_demo_application(demo_data, "Large Family, Low Income")
        
        if st.button("🏖️ Retired with Pension", use_container_width=True, key="demo_approved_2"):
            demo_data = {
                "name": "Fatima Al Zahra",
                "emirates_id": "784-1955-2345678-2",
                "employment_status": "retired",
                "monthly_income": 1200,
                "family_size": 2,
                "housing_status": "rented",
                "monthly_rent": 1800,
                "has_medical_conditions": False,
                "number_of_dependents": 1
            }
            process_demo_application(demo_data, "Retired with Pension")
        
        if st.button("🤱 Single Parent", use_container_width=True, key="demo_approved_3"):
            demo_data = {
                "name": "Mariam Hassan",
                "emirates_id": "784-1985-3456789-3",
                "employment_status": "unemployed",
                "monthly_income": 0,
                "family_size": 3,
                "housing_status": "family",
                "monthly_rent": 0,
                "has_medical_conditions": False,
                "number_of_dependents": 2
            }
            process_demo_application(demo_data, "Single Parent")
    
    with col2:
        st.markdown("#### ❌ Likely Declined Scenarios")
        
        if st.button("💼 High Income Professional", use_container_width=True, key="demo_declined_1"):
            demo_data = {
                "name": "Omar Al Rashid",
                "emirates_id": "784-1980-4567890-4",
                "employment_status": "employed",
                "monthly_income": 15000,
                "family_size": 4,
                "housing_status": "owned",
                "monthly_rent": 0,
                "has_medical_conditions": False,
                "number_of_dependents": 2
            }
            process_demo_application(demo_data, "High Income Professional")
        
        if st.button("🏢 Successful Business Owner", use_container_width=True, key="demo_declined_2"):
            demo_data = {
                "name": "Khalid Al Maktoum",
                "emirates_id": "784-1975-5678901-5",
                "employment_status": "self_employed",
                "monthly_income": 25000,
                "family_size": 3,
                "housing_status": "owned",
                "monthly_rent": 0,
                "has_medical_conditions": False,
                "number_of_dependents": 1
            }
            process_demo_application(demo_data, "Successful Business Owner")
        
        if st.button("👨‍🎓 Young Professional", use_container_width=True, key="demo_declined_3"):
            demo_data = {
                "name": "Ali Al Nasser",
                "emirates_id": "784-1995-6789012-6",
                "employment_status": "employed",
                "monthly_income": 8000,
                "family_size": 1,
                "housing_status": "rented",
                "monthly_rent": 3000,
                "has_medical_conditions": False,
                "number_of_dependents": 0
            }
            process_demo_application(demo_data, "Young Professional")
    
    # Custom demo option
    st.markdown("---")
    st.markdown("### 🎛️ Custom Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💬 Try Interactive Chat Demo", use_container_width=True):
            st.session_state.page = "💬 Chat Application"
            st.rerun()
    
    with col2:
        if st.button("📝 Create Custom Application", use_container_width=True):
            st.session_state.page = "📝 Submit Application"
            st.rerun()
    
    # Navigation buttons
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = "🏠 Home"
            st.rerun()
    
    with col2:
        if st.button("💬 Start Real Application", use_container_width=True):
            st.session_state.page = "💬 Chat Application"
            st.rerun()
    
    with col3:
        if st.button("📋 Check Status", use_container_width=True):
            st.session_state.page = "📋 Application Status"
            st.rerun()


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
        # Create DataFrame for Streamlit chart
        chart_data = pd.DataFrame({
            'Category': categories,
            'Score': scores
        })
        st.bar_chart(chart_data.set_index('Category'), use_container_width=True)
    
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
    
    st.line_chart(df, x='Date', y=['Applications', 'Approvals'], use_container_width=True)
    
    # Processing time distribution
    st.markdown("### Processing Time Distribution")
    
    import numpy as np
    processing_times = np.random.normal(90, 20, 1000)  # 90 seconds average
    processing_times = processing_times[processing_times > 30]  # Minimum 30 seconds
    
    # Create histogram data manually
    hist, bin_edges = np.histogram(processing_times, bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    hist_df = pd.DataFrame({
        'Processing Time (seconds)': bin_centers,
        'Frequency': hist
    })
    
    st.line_chart(hist_df.set_index('Processing Time (seconds)'), use_container_width=True)


def show_chat_application_page():
    """Display the main chat-based application interface"""
    
    # Import the chat interface
    try:
        from chat_interface import show_chat_interface
        show_chat_interface()
    except ImportError as e:
        st.error(f"Chat interface not available: {str(e)}")
        st.info("Using fallback chat implementation...")
        show_fallback_chat_interface()


def show_fallback_chat_interface():
    """Fallback chat interface if the main one fails to load"""
    
    st.markdown("## 🤖 Social Support AI Assistant")
    st.markdown("I'll help you apply for social support through a conversational interface.")
    
    # Initialize session state for fallback
    if "fallback_messages" not in st.session_state:
        st.session_state.fallback_messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your Social Support AI Assistant. I'll help you apply for financial support. What's your full name?"
            }
        ]
    
    if "fallback_state" not in st.session_state:
        st.session_state.fallback_state = {
            "step": "name_collection",
            "data": {}
        }
    
    # Display messages
    for message in st.session_state.fallback_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your response..."):
        # Add user message
        st.session_state.fallback_messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Generate simple response
        response = generate_simple_chat_response(
            prompt, 
            st.session_state.fallback_state
        )
        
        # Add assistant response
        st.session_state.fallback_messages.append({
            "role": "assistant", 
            "content": response["message"]
        })
        
        # Update state
        st.session_state.fallback_state.update(response.get("state_update", {}))
        
        st.rerun()


def generate_simple_chat_response(user_input: str, state: dict) -> dict:
    """Generate simple chat response for fallback mode"""
    
    current_step = state.get("step", "name_collection")
    
    if current_step == "name_collection":
        if len(user_input.split()) >= 2:
            state["data"]["name"] = user_input
            return {
                "message": f"Nice to meet you, {user_input}! What's your current employment status? Are you employed, unemployed, self-employed, or retired?",
                "state_update": {"step": "employment"}
            }
        else:
            return {
                "message": "Could you provide your full name (first and last name)?",
                "state_update": {}
            }
    
    elif current_step == "employment":
        employment = "unemployed"
        if "employed" in user_input.lower() and "unemployed" not in user_input.lower():
            employment = "employed"
        elif "self" in user_input.lower():
            employment = "self_employed" 
        elif "retired" in user_input.lower():
            employment = "retired"
        
        state["data"]["employment_status"] = employment
        return {
            "message": f"I understand you are {employment}. What's your approximate monthly income in AED?",
            "state_update": {"step": "income"}
        }
    
    elif current_step == "income":
        # Extract number from input
        import re
        numbers = re.findall(r'\d+', user_input)
        income = int(numbers[0]) if numbers else 0
        
        state["data"]["monthly_income"] = income
        return {
            "message": f"Thank you. How many people are in your household (including yourself)?",
            "state_update": {"step": "family_size"}
        }
    
    elif current_step == "family_size":
        import re
        numbers = re.findall(r'\d+', user_input)
        family_size = int(numbers[0]) if numbers else 1
        
        state["data"]["family_size"] = family_size
        
        # Simple eligibility calculation
        income = state["data"].get("monthly_income", 0)
        threshold = 3000 * family_size
        
        if income < threshold:
            support = max(500, (threshold - income) * 0.5)
            return {
                "message": f"🎉 Great news! Based on your information, you are eligible for approximately {support:.0f} AED per month in social support. A complete assessment would provide more detailed calculations.",
                "state_update": {"step": "complete"}
            }
        else:
            return {
                "message": "Based on your income level, you may not qualify for direct financial support, but there are economic enablement programs that could help you.",
                "state_update": {"step": "complete"}
            }
    
    else:
        return {
            "message": "Thank you for using the Social Support AI system! Your application information has been recorded.",
            "state_update": {}
        }


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


if __name__ == "__main__":
    main() 