#!/usr/bin/env python3
"""
Simple test script to verify conversation flow with document upload and proceed functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.agents.conversation_agent import ConversationAgent, ConversationStep

async def test_conversation_flow():
    """Test the conversation flow with document upload scenario"""
    
    print("🧪 Testing Conversation Flow with Document Upload")
    print("=" * 50)
    
    # Initialize conversation agent
    conversation_agent = ConversationAgent()
    
    # Initial conversation state
    conversation_state = {
        "current_step": ConversationStep.DOCUMENT_COLLECTION,
        "collected_data": {
            "name": "dev na",
            "emirates_id": "722-1234-1234567-1",
            "employment_status": "retired",
            "monthly_income": 1000.0,
            "family_size": 3,
            "housing_status": "rented"
        }
    }
    
    print(f"📊 Starting at step: {conversation_state['current_step']}")
    print(f"📋 Collected data: {conversation_state['collected_data']}")
    
    # Test scenarios
    test_scenarios = [
        ("uploaded", "User says 'uploaded' after uploading documents"),
        ("proceed with assessment", "User says 'proceed with assessment'"),
        ("proceed", "User says 'proceed'"),
        ("done", "User says 'done'"),
        ("ready", "User says 'ready'"),
        ("go ahead", "User says 'go ahead'")
    ]
    
    for user_input, description in test_scenarios:
        print(f"\n🧪 Test: {description}")
        print(f"👤 User: {user_input}")
        
        try:
            response = await conversation_agent.process_message(
                user_input, [], conversation_state.copy()
            )
            
            print(f"🤖 Response: {response['message'][:100]}...")
            
            if "state_update" in response:
                new_step = response["state_update"].get("current_step")
                if new_step:
                    print(f"📊 New Step: {new_step}")
                    
                if response.get("application_complete"):
                    print("✅ Application completed!")
                    
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 30)
    
    print("\n🎉 All conversation flow tests completed!")

if __name__ == "__main__":
    asyncio.run(test_conversation_flow()) 