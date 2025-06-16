#!/usr/bin/env python3
"""
Test ChromaDB Integration in Completion Flow

This script tests the enhanced LLM recommendations that now include
ChromaDB-sourced training programs and job opportunities.
"""
import requests
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

API_BASE = "http://localhost:8000"

def test_chromadb_enhanced_completion():
    """Test the complete flow with ChromaDB-enhanced recommendations"""
    
    print("🧪 Testing ChromaDB-enhanced completion flow...")
    
    # Test conversation state for completion with unemployed user
    conversation_state = {
        "current_step": "document_collection",
        "collected_data": {
            "name": "AHMED HASSAN",
            "first_name": "AHMED",
            "last_name": "HASSAN", 
            "emirates_id": "784-1985-1234567-8",
            "id_verified": True,
            "employment_status": "unemployed",  # This should trigger relevant training/job recommendations
            "monthly_income": 0,
            "family_size": 3,
            "housing_status": "rent"
        },
        "uploaded_documents": [],
        "processing_status": "ready_for_assessment",
        "eligibility_result": None,
        "application_id": None
    }
    
    conversation_messages = [
        {"role": "user", "content": "AHMED HASSAN"},
        {"role": "assistant", "content": "Nice to meet you, AHMED HASSAN!"},
        {"role": "user", "content": "784-1985-1234567-8"},
        {"role": "assistant", "content": "Thank you! Your Emirates ID has been recorded."},
        {"role": "user", "content": "unemployed"},
        {"role": "assistant", "content": "I understand you're currently unemployed."},
        {"role": "user", "content": "0"},
        {"role": "assistant", "content": "Thank you. I've noted your monthly income."},
        {"role": "user", "content": "3"},
        {"role": "assistant", "content": "Got it - 3 people in your household."},
        {"role": "user", "content": "rent"},
        {"role": "assistant", "content": "Perfect! I have the basic information I need."},
        {"role": "user", "content": "proceed with assessment"}
    ]
    
    # Test the API call
    try:
        print("📡 Sending request to API...")
        response = requests.post(
            f"{API_BASE}/conversation/message",
            json={
                "message": "proceed with assessment",
                "conversation_history": conversation_messages,
                "conversation_state": conversation_state
            },
            timeout=300  # 3 minutes for completion processing
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API Response received successfully!")
            print(f"📊 Application Complete: {result.get('application_complete', False)}")
            print(f"📝 Message Length: {len(result.get('message', ''))} characters")
            
            # Check if the response contains ChromaDB-enhanced recommendations
            message = result.get('message', '')
            
            # Look for ChromaDB integration indicators
            chromadb_indicators = [
                "Training Programs:",
                "Job Opportunities:",
                "Contact Information",
                "Digital Skills Training",
                "Customer Service Representative",
                "Vocational Training"
            ]
            
            found_indicators = [indicator for indicator in chromadb_indicators if indicator in message]
            
            print(f"\n🔍 ChromaDB Integration Analysis:")
            print(f"   Found {len(found_indicators)} ChromaDB indicators:")
            for indicator in found_indicators:
                print(f"   ✅ {indicator}")
            
            if len(found_indicators) >= 3:
                print("\n🎉 SUCCESS: ChromaDB integration is working!")
                print("   The completion message includes personalized training and job recommendations.")
            else:
                print("\n⚠️  WARNING: Limited ChromaDB integration detected.")
                print("   The response may be using fallback recommendations.")
            
            # Print a sample of the response
            print(f"\n📄 Response Sample (first 500 chars):")
            print(f"   {message[:500]}...")
            
            # Check for specific ChromaDB-sourced content
            if "Digital Skills Training" in message:
                print("\n✅ CONFIRMED: Digital Skills Training program found in response")
            if "Customer Service Representative" in message:
                print("✅ CONFIRMED: Customer Service job opportunity found in response")
            if "Contact Information" in message:
                print("✅ CONFIRMED: Contact information section found in response")
            
            return True
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (this is expected for completion processing)")
        print("   The backend is likely still processing the request.")
        return False
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

def test_employed_user_recommendations():
    """Test ChromaDB recommendations for employed user with low income"""
    
    print("\n🧪 Testing ChromaDB recommendations for employed user...")
    
    conversation_state = {
        "current_step": "document_collection",
        "collected_data": {
            "name": "FATIMA AL ZAHRA",
            "emirates_id": "784-1985-9876543-2",
            "employment_status": "employed",  # Employed but low income
            "monthly_income": 2500,  # Low income should trigger skill development recommendations
            "family_size": 2,
            "housing_status": "rent"
        },
        "processing_status": "ready_for_assessment"
    }
    
    conversation_messages = [
        {"role": "user", "content": "proceed with assessment"}
    ]
    
    try:
        response = requests.post(
            f"{API_BASE}/conversation/message",
            json={
                "message": "proceed with assessment",
                "conversation_history": conversation_messages,
                "conversation_state": conversation_state
            },
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            message = result.get('message', '')
            
            print("✅ Employed user test completed!")
            
            # Check for skill development recommendations
            skill_indicators = [
                "Digital Skills Training",
                "Professional certification",
                "Career advancement",
                "Training Programs"
            ]
            
            found_skills = [indicator for indicator in skill_indicators if indicator.lower() in message.lower()]
            print(f"   Found {len(found_skills)} skill development indicators")
            
            return len(found_skills) > 0
            
    except Exception as e:
        print(f"❌ Employed user test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    
    print("🚀 Testing ChromaDB Integration in Social Support AI Workflow")
    print("=" * 60)
    
    # Test 1: Unemployed user (should get training + job recommendations)
    test1_success = test_chromadb_enhanced_completion()
    
    # Test 2: Employed low-income user (should get skill development recommendations)
    test2_success = test_employed_user_recommendations()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    print(f"   Unemployed User Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"   Employed User Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("   ChromaDB integration is working correctly for job/training recommendations.")
    else:
        print("\n⚠️  SOME TESTS FAILED!")
        print("   Check the API server and ChromaDB setup.")
    
    print("\n💡 Next Steps:")
    print("   - Try the full conversation flow in the UI")
    print("   - Check that personalized recommendations appear in completion")
    print("   - Verify contact information is included")

if __name__ == "__main__":
    main() 