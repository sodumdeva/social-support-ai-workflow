"""
Example API Usage for Social Support AI Conversation System

This script demonstrates how to interact with the conversation API endpoints
to build a complete conversational application flow.
"""
import requests
import json
import time


class ConversationAPIExample:
    """Example class for interacting with the conversation API"""
    
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.conversation_state = {
            "current_step": "name_collection",
            "collected_data": {},
            "uploaded_documents": []
        }
        self.conversation_history = []
    
    def send_message(self, user_message: str):
        """Send a message to the conversation API"""
        
        print(f"👤 User: {user_message}")
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare API request
        payload = {
            "message": user_message,
            "conversation_history": self.conversation_history,
            "conversation_state": self.conversation_state
        }
        
        try:
            # Send request to API
            response = requests.post(
                f"{self.api_base_url}/conversation/message",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract response
                assistant_message = result.get("message", "No response")
                print(f"🤖 Assistant: {assistant_message}")
                
                # Update conversation state
                if "state_update" in result:
                    self.conversation_state.update(result["state_update"])
                
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                # Show current state
                print(f"📊 Current Step: {self.conversation_state.get('current_step', 'unknown')}")
                print(f"📋 Collected Data: {json.dumps(self.conversation_state.get('collected_data', {}), indent=2)}")
                
                # Check if application is complete
                if result.get("application_complete"):
                    print("\n🎉 APPLICATION COMPLETED!")
                    final_decision = result.get("final_decision", {})
                    if final_decision.get("eligible"):
                        print(f"✅ APPROVED - Support Amount: {final_decision.get('support_amount', 0)} AED/month")
                    else:
                        print(f"❌ DECLINED - Reason: {final_decision.get('reason', 'Not specified')}")
                    return True
                
                return False
                
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Request Error: {str(e)}")
            return False
    
    def upload_document(self, file_path: str, document_type: str):
        """Upload a document during conversation"""
        
        print(f"📄 Uploading {document_type}: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                files = {"file": (file_path, f, "application/octet-stream")}
                data = {
                    "file_type": document_type,
                    "conversation_state": json.dumps(self.conversation_state)
                }
                
                response = requests.post(
                    f"{self.api_base_url}/conversation/upload-document",
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"🤖 Document Response: {result.get('message', 'Document processed')}")
                
                # Update state
                if "state_update" in result:
                    self.conversation_state.update(result["state_update"])
                
                return True
            else:
                print(f"❌ Upload Error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Upload Error: {str(e)}")
            return False


def run_complete_conversation_example():
    """Run a complete conversation example using the API"""
    
    print("🚀 Social Support AI - Complete Conversation Example")
    print("=" * 60)
    
    # Initialize API client
    api = ConversationAPIExample()
    
    # Test conversation flow
    conversation_steps = [
        "Ahmed Al Mansouri",
        "784-1990-1234567-1", 
        "I am currently employed",
        "My monthly salary is 4500 AED",
        "We are 5 people in my family",
        "We rent our apartment",
        "I want to proceed with the assessment"
    ]
    
    for i, message in enumerate(conversation_steps, 1):
        print(f"\n--- Step {i} ---")
        
        # Send message
        is_complete = api.send_message(message)
        
        if is_complete:
            break
        
        # Add small delay between messages
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("Conversation Example Complete!")


def run_document_upload_example():
    """Example of document upload during conversation"""
    
    print("\n📄 Document Upload Example")
    print("=" * 40)
    
    api = ConversationAPIExample()
    
    # Start conversation
    api.send_message("Fatima Hassan")
    
    # Create a mock document file
    mock_file_path = "mock_emirates_id.txt"
    with open(mock_file_path, "w") as f:
        f.write("EMIRATES ID\nName: Fatima Hassan\nID: 784-1995-9876543-2\nNationality: UAE\nDOB: 15/03/1995")
    
    # Upload document
    api.upload_document(mock_file_path, "emirates_id")
    
    # Continue conversation
    api.send_message("I am employed")
    
    # Clean up
    import os
    if os.path.exists(mock_file_path):
        os.remove(mock_file_path)


def test_api_health():
    """Test if the API is running"""
    
    print("🔍 Testing API Health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API is healthy!")
            print(f"📊 Status: {health_data.get('status', 'unknown')}")
            print(f"🕐 Timestamp: {health_data.get('timestamp', 'unknown')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to API: {str(e)}")
        print("💡 Make sure to start the API server first with: python3 run_api_simple.py")
        return False


def show_api_endpoints():
    """Show available API endpoints"""
    
    print("\n📋 Available API Endpoints")
    print("=" * 40)
    
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"🔗 API: {data.get('message', 'Social Support AI API')}")
            print(f"📦 Version: {data.get('version', 'unknown')}")
            
            endpoints = data.get('endpoints', {})
            print("\n🛠️ Endpoints:")
            for name, path in endpoints.items():
                print(f"  • {name}: {path}")
            
            features = data.get('features', [])
            print("\n✨ Features:")
            for feature in features:
                print(f"  • {feature}")
                
        else:
            print("❌ Could not retrieve API information")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def main():
    """Main function to run all examples"""
    
    print("🤖 Social Support AI - API Usage Examples")
    print("This demonstrates how to interact with the conversation API")
    print("=" * 60)
    
    # Test API health first
    if not test_api_health():
        print("\n⚠️ API is not running. Please start it first:")
        print("   python3 run_api_simple.py")
        return
    
    # Show available endpoints
    show_api_endpoints()
    
    # Run conversation example
    try:
        run_complete_conversation_example()
        
        # Run document upload example
        run_document_upload_example()
        
        print("\n✅ All API examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")


if __name__ == "__main__":
    main() 