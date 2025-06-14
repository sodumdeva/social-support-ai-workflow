"""
Test suite for AI Agents in Social Support Workflow System
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.agents.master_orchestrator import MasterOrchestrator
from src.agents.conversation_agent import ConversationAgent
from src.agents.data_extraction_agent import DataExtractionAgent
from src.agents.eligibility_agent import EligibilityAgent


class TestMasterOrchestrator:
    """Test Master Orchestrator Agent"""
    
    @pytest.fixture
    def orchestrator(self):
        return MasterOrchestrator()
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly"""
        assert orchestrator is not None
        assert hasattr(orchestrator, 'process')
    
    @pytest.mark.asyncio
    async def test_orchestrator_process_flow(self, orchestrator):
        """Test orchestrator processes workflow correctly"""
        test_input = {
            "user_message": "I need financial support",
            "conversation_state": "initial"
        }
        
        # Mock the process method to avoid actual LLM calls
        with patch.object(orchestrator, 'process', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "status": "success",
                "next_step": "data_collection",
                "response": "I'll help you with your application"
            }
            
            result = await orchestrator.process(test_input)
            assert result["status"] == "success"
            assert "response" in result


class TestConversationAgent:
    """Test Conversation Agent"""
    
    @pytest.fixture
    def conversation_agent(self):
        return ConversationAgent()
    
    @pytest.mark.asyncio
    async def test_conversation_agent_initialization(self, conversation_agent):
        """Test conversation agent initializes correctly"""
        assert conversation_agent is not None
        assert hasattr(conversation_agent, 'process')
    
    @pytest.mark.asyncio
    async def test_intent_analysis(self, conversation_agent):
        """Test intent analysis functionality"""
        test_message = "I want to apply for financial assistance"
        
        with patch.object(conversation_agent, '_analyze_intent', new_callable=AsyncMock) as mock_intent:
            mock_intent.return_value = {
                "intent": "financial_support_application",
                "confidence": 0.95,
                "entities": ["financial_assistance"]
            }
            
            result = await conversation_agent._analyze_intent(test_message, [])
            assert result["intent"] == "financial_support_application"
            assert result["confidence"] > 0.9


class TestDataExtractionAgent:
    """Test Data Extraction Agent"""
    
    @pytest.fixture
    def extraction_agent(self):
        return DataExtractionAgent()
    
    @pytest.mark.asyncio
    async def test_extraction_agent_initialization(self, extraction_agent):
        """Test extraction agent initializes correctly"""
        assert extraction_agent is not None
        assert hasattr(extraction_agent, 'process')
    
    @pytest.mark.asyncio
    async def test_document_processing(self, extraction_agent):
        """Test document processing functionality"""
        test_documents = [
            {"path": "test_emirates_id.jpg", "type": "emirates_id"},
            {"path": "test_bank_statement.pdf", "type": "bank_statement"}
        ]
        
        with patch.object(extraction_agent, 'process', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "status": "success",
                "extracted_data": {
                    "emirates_id": {"name": "John Doe", "id_number": "123456789"},
                    "bank_statement": {"monthly_income": 5000, "account_balance": 15000}
                }
            }
            
            result = await extraction_agent.process({"documents": test_documents})
            assert result["status"] == "success"
            assert "extracted_data" in result


class TestEligibilityAgent:
    """Test Eligibility Agent"""
    
    @pytest.fixture
    def eligibility_agent(self):
        return EligibilityAgent()
    
    @pytest.mark.asyncio
    async def test_eligibility_agent_initialization(self, eligibility_agent):
        """Test eligibility agent initializes correctly"""
        assert eligibility_agent is not None
        assert hasattr(eligibility_agent, 'process')
    
    @pytest.mark.asyncio
    async def test_eligibility_assessment(self, eligibility_agent):
        """Test eligibility assessment functionality"""
        test_application_data = {
            "monthly_income": 3000,
            "family_size": 4,
            "employment_status": "employed",
            "housing_status": "rented",
            "age": 35,
            "has_medical_conditions": False
        }
        
        with patch.object(eligibility_agent, 'process', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "status": "success",
                "eligibility_result": {
                    "eligible": True,
                    "confidence": 0.87,
                    "support_amount": 2500,
                    "risk_level": "low",
                    "recommendations": ["financial_support", "job_training"]
                }
            }
            
            result = await eligibility_agent.process({"application_data": test_application_data})
            assert result["status"] == "success"
            assert result["eligibility_result"]["eligible"] is True
            assert result["eligibility_result"]["confidence"] > 0.8


@pytest.mark.asyncio
async def test_agent_integration():
    """Test integration between agents"""
    orchestrator = MasterOrchestrator()
    
    test_workflow_data = {
        "user_message": "I need help with financial support",
        "conversation_state": "initial",
        "user_data": {}
    }
    
    # Mock the entire workflow
    with patch.object(orchestrator, 'process', new_callable=AsyncMock) as mock_orchestrator:
        mock_orchestrator.return_value = {
            "status": "success",
            "workflow_complete": True,
            "final_decision": {
                "approved": True,
                "support_amount": 2000,
                "next_steps": ["document_verification", "account_setup"]
            }
        }
        
        result = await orchestrator.process(test_workflow_data)
        assert result["status"] == "success"
        assert result["workflow_complete"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 