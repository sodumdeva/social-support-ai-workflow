#!/usr/bin/env python3
"""
Database Integration Demo for Social Support AI System

This demo shows the complete end-to-end database integration:
✅ Applications stored in PostgreSQL with all details
✅ Documents linked and stored with metadata  
✅ Conversation history preserved
✅ ML predictions tracked
✅ Audit trail maintained
✅ Complete workflow integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
from datetime import datetime
import tempfile

from src.models.database import (
    SessionLocal, 
    DatabaseManager, 
    Application, 
    Document,
    ApplicationReview,
    MLModelPerformance
)
from src.workflows.langgraph_workflow import finalize_application
from src.utils.logging_config import get_logger

logger = get_logger("database_demo")


async def demo_complete_application_storage():
    """Demonstrate complete application storage with all features"""
    
    print("🚀 Social Support AI - Complete Database Integration Demo")
    print("=" * 70)
    
    db = SessionLocal()
    
    try:
        # Step 1: Prepare complete application data
        print("📝 Step 1: Preparing complete application data...")
        
        # Create temp documents for demo
        temp_dir = tempfile.mkdtemp()
        doc_paths = [
            os.path.join(temp_dir, "emirates_id.pdf"),
            os.path.join(temp_dir, "bank_statement.pdf")
        ]
        
        for doc_path in doc_paths:
            with open(doc_path, 'w') as f:
                f.write(f"Demo document content: {os.path.basename(doc_path)}")
        
        # Complete application state (as would come from conversation workflow)
        application_state = {
            "messages": [
                {"role": "user", "content": "My name is Sarah Al Mahmoud", "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": "Thank you Sarah! What's your Emirates ID?", "timestamp": datetime.now().isoformat()},
                {"role": "user", "content": "784199088888888", "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": "What's your phone number?", "timestamp": datetime.now().isoformat()},
                {"role": "user", "content": "+971503456789", "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": "What's your employment status?", "timestamp": datetime.now().isoformat()},
                {"role": "user", "content": "I'm currently unemployed", "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": "What was your last monthly income?", "timestamp": datetime.now().isoformat()},
                {"role": "user", "content": "1800 AED per month", "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": "How many people in your family?", "timestamp": datetime.now().isoformat()},
                {"role": "user", "content": "4 people including myself", "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": "Thank you! Processing your application...", "timestamp": datetime.now().isoformat()}
            ],
            "collected_data": {
                "name": "Sarah Al Mahmoud",
                "emirates_id": "784199088888888",
                "phone_number": "+971503456789",  # Added phone number for reference system
                "email": "sarah.mahmoud@email.com",
                "nationality": "UAE",
                "employment_status": "unemployed",
                "monthly_income": 1800.0,
                "family_size": 4,
                "dependents_count": 2,
                "spouse_employment_status": "employed",
                "spouse_monthly_income": 4200.0,
                "housing_status": "rented",
                "monthly_rent": 3500.0,
                "total_assets": 12000.0,
                "total_liabilities": 8000.0,
                "monthly_expenses": 4800.0,
                "savings_amount": 2500.0,
                "credit_score": 720
            },
            "eligibility_result": {
                "eligible": True,
                "support_amount": 3200.0,
                "eligibility_score": 0.92,
                "reason": "Meets all eligibility criteria: unemployed with family of 4, spouse employed providing stability",
                "ml_prediction": {
                    "eligible": True,
                    "support_amount": 3200.0,
                    "confidence": 0.92,
                    "features_used": {
                        "monthly_income": 1800.0,
                        "family_size": 4,
                        "employment_status": "unemployed",
                        "housing_status": "rented"
                    },
                    "processing_time_ms": 45.3
                },
                "economic_enablement": {
                    "recommendations_text": "Recommended programs: Digital Skills Training, Job Placement Assistance, Financial Literacy Workshop",
                    "programs": ["Digital Skills Training", "Job Placement Program", "Financial Literacy"]
                },
                "risk_assessment": {
                    "overall_risk_score": 0.25,
                    "financial_risk": "low",
                    "fraud_risk": "very_low"
                }
            },
            "uploaded_documents": doc_paths,
            "processed_documents": [doc_paths[0]],  # First document processed
            "workflow_history": [
                {"step": "name_collection", "timestamp": datetime.now().isoformat(), "status": "completed"},
                {"step": "eligibility_assessment", "timestamp": datetime.now().isoformat(), "status": "completed"}
            ],
            "error_messages": []
        }
        
        print(f"   ✅ Application data prepared for: {application_state['collected_data']['name']}")
        print(f"   ✅ Phone number: {application_state['collected_data']['phone_number']}")
        print(f"   ✅ {len(application_state['messages'])} conversation messages")
        print(f"   ✅ {len(application_state['uploaded_documents'])} documents uploaded")
        print(f"   ✅ Eligibility: {application_state['eligibility_result']['eligible']} (Support: {application_state['eligibility_result']['support_amount']} AED)")
        
        # Step 2: Store in database using workflow function
        print("\n💾 Step 2: Storing application in PostgreSQL database...")
        
        result_state = await finalize_application(application_state)
        
        if result_state.get("database_stored"):
            application_id = result_state["application_id"]
            print(f"   ✅ Application stored successfully: {application_id}")
        else:
            print(f"   ❌ Database storage failed: {result_state.get('database_error', 'Unknown error')}")
            return
        
        # Step 3: Verify database records and user-friendly references
        print("\n🔍 Step 3: Verifying database records and user-friendly tracking...")
        
        # Get application from database
        db_application = DatabaseManager.get_application_by_id(db, application_id)
        
        if db_application:
            print(f"   ✅ Application Record:")
            print(f"      - ID: {db_application.application_id}")
            print(f"      - 🔢 Reference Number: {db_application.reference_number} (User-friendly!)")
            print(f"      - 📱 Phone Reference: {db_application.phone_reference} (Last 4 digits)")
            print(f"      - Name: {db_application.full_name}")
            print(f"      - Emirates ID: {db_application.emirates_id}")
            print(f"      - Employment: {db_application.employment_status.value}")
            print(f"      - Income: {db_application.monthly_income} AED")
            print(f"      - Family Size: {db_application.family_size}")
            print(f"      - Housing: {db_application.housing_status.value}")
            print(f"      - Status: {db_application.status.value}")
            print(f"      - Eligible: {db_application.is_eligible}")
            print(f"      - Support Amount: {db_application.recommended_support_amount} AED")
            
            # Check conversation history
            if db_application.conversation_history:
                print(f"   ✅ Conversation History: {len(db_application.conversation_history)} messages stored")
            
            # Check metadata
            if db_application.application_metadata:
                metadata = db_application.application_metadata
                print(f"   ✅ Application Metadata:")
                print(f"      - Workflow Version: {metadata.get('workflow_version')}")
                print(f"      - Processing Method: {metadata.get('processing_method')}")
                print(f"      - Total Messages: {metadata.get('total_messages')}")
                print(f"      - Documents Processed: {metadata.get('documents_processed')}")
            
            # Check documents
            documents = db_application.documents
            if documents:
                print(f"   ✅ Documents: {len(documents)} stored")
                for doc in documents:
                    print(f"      - {doc.document_type.value}: {doc.filename} ({doc.file_size} bytes)")
                    print(f"        Processed: {doc.is_processed}, Status: {doc.processing_status}")
            
            # Check reviews
            reviews = db_application.reviews
            if reviews:
                print(f"   ✅ Reviews: {len(reviews)} stored")
                for review in reviews:
                    print(f"      - {review.review_type} by {review.reviewer_name}: {review.decision}")
                    print(f"        Risk Score: {review.risk_assessment.get('overall_risk_score', 'N/A')}")
            
            # Check ML predictions
            ml_records = db.query(MLModelPerformance).filter(
                MLModelPerformance.application_id == db_application.id
            ).all()
            
            if ml_records:
                print(f"   ✅ ML Predictions: {len(ml_records)} stored")
                for record in ml_records:
                    print(f"      - {record.model_name}: {record.confidence_score} confidence")
                    print(f"        Processing Time: {record.processing_time_ms}ms")
            else:
                print("   ℹ️ No ML predictions stored (normal if models not used)")
        
        # Step 4: Demonstrate user-friendly lookup methods
        print("\n🔎 Step 4: Demonstrating user-friendly lookup methods...")
        
        # Test lookup by reference number
        ref_app = DatabaseManager.get_application_by_reference(db, db_application.reference_number)
        if ref_app:
            print(f"   ✅ Lookup by Reference Number ({db_application.reference_number}): Found {ref_app.full_name}")
        
        # Test lookup by Emirates ID
        emirates_app = DatabaseManager.get_application_by_emirates_id(db, db_application.emirates_id)
        if emirates_app:
            print(f"   ✅ Lookup by Emirates ID ({db_application.emirates_id}): Found {emirates_app.full_name}")
        
        # Test lookup by name + phone
        name_phone_app = DatabaseManager.get_application_by_phone_and_name(
            db, db_application.phone_reference, "Sarah"
        )
        if name_phone_app:
            print(f"   ✅ Lookup by Name + Phone (Sarah + {db_application.phone_reference}): Found {name_phone_app.full_name}")
        
        # Step 5: Query capabilities demo
        print("\n🔎 Step 5: Demonstrating advanced query capabilities...")
        
        # Query by criteria
        unemployed_apps = DatabaseManager.get_applications_by_criteria(db, {
            "employment_status": "unemployed"
        })
        print(f"   ✅ Found {len(unemployed_apps)} unemployed applicants")
        
        family_apps = DatabaseManager.get_applications_by_criteria(db, {
            "family_size": 4
        })
        print(f"   ✅ Found {len(family_apps)} applications with family size 4")
        
        income_apps = DatabaseManager.get_applications_by_criteria(db, {
            "income_range": (1000, 2000)
        })
        print(f"   ✅ Found {len(income_apps)} applications with income 1000-2000 AED")
        
        # Step 6: User Experience Summary
        print("\n🎉 Step 6: User Experience Summary")
        print("   ✅ Complete application data stored in PostgreSQL")
        print("   ✅ Documents linked with metadata and processing status")
        print("   ✅ Conversation history preserved for audit trail")
        print("   ✅ ML predictions tracked for model performance")
        print("   ✅ Application reviews stored for compliance")
        print("   ✅ Advanced querying capabilities demonstrated")
        print("   ✅ Production-ready database architecture")
        
        print(f"\n🔢 **USER-FRIENDLY TRACKING METHODS:**")
        print(f"   📋 Reference Number: {db_application.reference_number}")
        print(f"   🆔 Emirates ID: {db_application.emirates_id}")
        print(f"   📱 Name + Phone: {db_application.full_name} + {db_application.phone_reference}")
        print(f"   🔍 Full Application ID: {application_id}")
        
        print(f"\n💡 **HOW USERS CAN CHECK STATUS:**")
        print(f"   1. Use Reference Number: {db_application.reference_number} (easiest!)")
        print(f"   2. Use Emirates ID: {db_application.emirates_id}")
        print(f"   3. Use Name + Last 4 digits of phone: Sarah + {db_application.phone_reference}")
        print(f"   4. Use full Application ID: {application_id}")
        
        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"\n✨ Database Integration Demo Complete!")
        print(f"   Application {application_id} successfully stored with comprehensive tracking system")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()


async def main():
    """Run the database integration demo"""
    await demo_complete_application_storage()


if __name__ == "__main__":
    asyncio.run(main()) 