# Social Support AI Conversation System - Examples

This document provides comprehensive examples of how to use the Social Support AI conversation system for processing social support applications through natural language interaction.

## 🚀 Quick Start

### 1. Start the API Server
```bash
python3 run_api_simple.py
```

### 2. Start the Frontend (Optional)
```bash
python3 run_frontend.py
```

### 3. Run Examples
```bash
# Test conversation agent directly
python3 test_conversation.py

# Test API endpoints
python3 example_api_usage.py
```

## 📋 Conversation Flow Examples

### Example 1: Complete Application Process

```python
# User starts conversation
👤 User: "Ahmed Al Mansouri"
🤖 Assistant: "Nice to meet you, Ahmed Al Mansouri! Now I need to verify your identity. Can you please upload your Emirates ID, or tell me your Emirates ID number?"

👤 User: "784-1990-1234567-1"
🤖 Assistant: "Thank you! Your Emirates ID has been recorded. Now let's talk about your employment situation. Are you currently employed, unemployed, self-employed, or retired?"

👤 User: "I am currently employed"
🤖 Assistant: "Great! Since you're employed, what is your approximate monthly salary in AED? You can also upload a recent bank statement for more accurate assessment."

👤 User: "My monthly salary is 4500 AED"
🤖 Assistant: "Thank you. I've noted your monthly income as 4,500 AED. Now, how many people are in your household (including yourself)?"

👤 User: "We are 5 people in my family"
🤖 Assistant: "Got it - 5 people in your household. What's your current housing situation? Do you own your home, rent, or live with family?"

👤 User: "We rent our apartment"
🤖 Assistant: "Perfect! I have the basic information I need. You can upload additional documents or I can proceed with the eligibility evaluation now. What would you prefer?"

👤 User: "I want to proceed with the assessment"
🤖 Assistant: "🎉 Great news! Based on your information, you are eligible for approximately 3,000 AED per month in social support."
```

### Example 2: Document Upload During Conversation

```python
# User uploads Emirates ID during identity verification
📄 Document Upload: emirates_id.jpg
🤖 Assistant: "Perfect! I've processed your Emirates ID. I can see you're Ahmed Al Mansouri, age 34. This information matches what you told me earlier."

# User uploads bank statement during income assessment
📄 Document Upload: bank_statement.pdf
🤖 Assistant: "Excellent! I've analyzed your bank statement. I can see your average monthly income is approximately 4,200 AED. This helps me make a more accurate assessment."
```

## 🎭 Different Scenarios

### Scenario 1: Unemployed Single Person
```json
{
  "name": "Omar Abdullah",
  "employment_status": "unemployed",
  "monthly_income": 0,
  "family_size": 1,
  "result": "✅ APPROVED - 1,500 AED/month"
}
```

### Scenario 2: Large Family with Low Income
```json
{
  "name": "Mariam Al Zahra",
  "employment_status": "employed",
  "monthly_income": 3000,
  "family_size": 6,
  "result": "✅ APPROVED - 6,000 AED/month"
}
```

### Scenario 3: High Income Professional
```json
{
  "name": "Khalid Al Rashid",
  "employment_status": "employed",
  "monthly_income": 15000,
  "family_size": 2,
  "result": "❌ DECLINED - Income exceeds eligibility threshold"
}
```

## 🔧 API Usage Examples

### Send Message to Conversation API

```python
import requests

# Send a message
payload = {
    "message": "Ahmed Al Mansouri",
    "conversation_history": [],
    "conversation_state": {
        "current_step": "name_collection",
        "collected_data": {},
        "uploaded_documents": []
    }
}

response = requests.post(
    "http://localhost:8000/conversation/message",
    json=payload
)

result = response.json()
print(result["message"])  # Assistant's response
```

### Upload Document During Conversation

```python
import requests

# Upload a document
with open("emirates_id.jpg", "rb") as f:
    files = {"file": ("emirates_id.jpg", f, "image/jpeg")}
    data = {
        "file_type": "emirates_id",
        "conversation_state": json.dumps(conversation_state)
    }
    
    response = requests.post(
        "http://localhost:8000/conversation/upload-document",
        files=files,
        data=data
    )

result = response.json()
print(result["message"])  # Document processing response
```

## 📊 Conversation State Management

The conversation system maintains state throughout the interaction:

```python
conversation_state = {
    "current_step": "name_collection",  # Current conversation step
    "collected_data": {                 # Data collected so far
        "name": "Ahmed Al Mansouri",
        "emirates_id": "784-1990-1234567-1",
        "employment_status": "employed",
        "monthly_income": 4500,
        "family_size": 5,
        "housing_status": "rented"
    },
    "uploaded_documents": [             # List of uploaded documents
        "data/uploads/emirates_id.jpg",
        "data/uploads/bank_statement.pdf"
    ]
}
```

## 🔄 Conversation Steps

The system follows these conversation steps:

1. **NAME_COLLECTION** - Collect user's full name
2. **IDENTITY_VERIFICATION** - Verify Emirates ID
3. **EMPLOYMENT_INQUIRY** - Ask about employment status
4. **INCOME_ASSESSMENT** - Collect income information
5. **FAMILY_DETAILS** - Get family size information
6. **HOUSING_SITUATION** - Ask about housing status
7. **DOCUMENT_COLLECTION** - Optional document uploads
8. **ELIGIBILITY_PROCESSING** - Process eligibility assessment
9. **COMPLETION** - Provide final decision and recommendations

## 📄 Supported Document Types

The system can process these document types:

- **Emirates ID** (`emirates_id`) - Identity verification
- **Bank Statement** (`bank_statement`) - Income verification
- **Resume/CV** (`resume`) - Employment history
- **Credit Report** (`credit_report`) - Financial assessment
- **Assets/Liabilities** (`assets_liabilities`) - Wealth evaluation

## ⚠️ Error Handling

The system handles various error scenarios:

```python
# Invalid name (too short)
👤 User: "A"
🤖 Assistant: "Could you please provide your full name (first and last name)? For example: 'Ahmed Al Mansouri'"

# Invalid Emirates ID format
👤 User: "123-invalid-id"
🤖 Assistant: "I need a valid Emirates ID number (format: XXX-XXXX-XXXXXXX-X). Could you provide it or upload a photo of your Emirates ID?"

# Non-numeric income
👤 User: "abc"
🤖 Assistant: "Please tell me your monthly income as a number in AED. For example: '4500' or 'zero' if you have no income."
```

## 🎯 Integration Examples

### Frontend Integration (Streamlit)

```python
import streamlit as st
from src.frontend.chat_interface import show_chat_interface

# Use the chat interface in your Streamlit app
show_chat_interface()
```

### Custom Integration

```python
from src.agents.conversation_agent import ConversationAgent

# Initialize agent
agent = ConversationAgent()

# Process user message
response = await agent.process_message(
    user_message="Ahmed Al Mansouri",
    conversation_history=[],
    conversation_state={
        "current_step": "name_collection",
        "collected_data": {},
        "uploaded_documents": []
    }
)

print(response["message"])  # Assistant's response
```

## 🚀 Running the Examples

1. **Test Conversation Agent Directly:**
   ```bash
   python3 test_conversation.py
   ```
   This runs the conversation agent directly without API calls.

2. **Test API Endpoints:**
   ```bash
   python3 example_api_usage.py
   ```
   This tests the conversation through API endpoints.

3. **Use the Web Interface:**
   ```bash
   python3 run_frontend.py
   ```
   Then navigate to `http://localhost:8501` and use the chat interface.

## 📈 Expected Output

When running the examples, you should see:

- ✅ Successful conversation flow through all steps
- 📊 Real-time state updates showing collected data
- 🎉 Final eligibility decision with support amount
- 📄 Document processing responses (if documents uploaded)
- ⚠️ Proper error handling for invalid inputs

## 🔍 Troubleshooting

**API Connection Issues:**
- Make sure the API server is running: `python3 run_api_simple.py`
- Check if port 8000 is available
- Verify the API base URL in examples

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python path configuration

**Document Processing Issues:**
- Verify document file paths exist
- Check supported file formats (PDF, JPG, PNG, XLSX, DOCX)
- Ensure proper file permissions

## 💡 Tips for Best Results

1. **Provide Complete Information:** Give full names, complete Emirates ID numbers, and specific income amounts
2. **Upload Quality Documents:** Use clear, readable document images for better OCR results
3. **Follow the Flow:** Let the conversation guide you through each step naturally
4. **Be Patient:** Document processing may take a few seconds
5. **Use Fallbacks:** The system provides fallback responses if AI processing fails

This conversation system demonstrates a complete transformation from traditional form-based applications to natural, conversational interfaces that can process applications in real-time while maintaining accuracy and user experience. 