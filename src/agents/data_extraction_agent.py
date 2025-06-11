"""
Data Extraction Agent for Social Support AI Workflow

Specializes in extracting structured information from various document types:
- Bank statements: income, expenses, financial patterns
- Emirates ID: identity verification, demographic info
- Resumes: employment history, skills, education
- Credit reports: credit history, debt information
- Assets/liabilities: financial position assessment
"""
from typing import Dict, Any, List, Optional
import re
import json
from datetime import datetime

from .base_agent import BaseAgent
from src.data.document_processor import DocumentProcessor


class DataExtractionAgent(BaseAgent):
    """Agent specialized in extracting structured data from documents"""
    
    def __init__(self):
        super().__init__("DataExtractionAgent")
        self.document_processor = DocumentProcessor()
        
        # Define extraction templates for different document types
        self.extraction_templates = {
            "bank_statement": {
                "income_patterns": [
                    "salary", "income", "wage", "payroll", "credit transfer"
                ],
                "expense_patterns": [
                    "debit", "payment", "withdrawal", "purchase", "bill"
                ],
                "required_fields": [
                    "monthly_income", "monthly_expenses", "account_balance", 
                    "transaction_frequency", "income_stability"
                ]
            },
            "emirates_id": {
                "required_fields": [
                    "full_name", "emirates_id_number", "nationality", 
                    "date_of_birth", "address", "emirates"
                ]
            },
            "resume": {
                "required_fields": [
                    "work_experience", "education", "skills", 
                    "employment_duration", "industry_experience"
                ]
            },
            "credit_report": {
                "required_fields": [
                    "credit_score", "total_debt", "payment_history", 
                    "credit_utilization", "number_of_accounts"
                ]
            },
            "assets": {
                "required_fields": [
                    "total_assets", "total_liabilities", "net_worth", 
                    "asset_types", "liquid_assets"
                ]
            }
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from documents
        
        Args:
            input_data: {
                "documents": [{"file_path": str, "document_type": str}],
                "application_id": str
            }
            
        Returns:
            Dictionary with extracted data for each document
        """
        documents = input_data.get("documents", [])
        application_id = input_data.get("application_id", "unknown")
        
        extraction_results = {}
        
        for doc_info in documents:
            file_path = doc_info["file_path"]
            document_type = doc_info["document_type"]
            
            try:
                # First, process document using document processor
                raw_extraction = self.document_processor.process_document(
                    file_path, document_type
                )
                
                if raw_extraction["status"] == "error":
                    extraction_results[document_type] = {
                        "status": "error",
                        "error": raw_extraction["error"]
                    }
                    continue
                
                # Then use LLM for intelligent extraction
                structured_data = await self._extract_structured_data(
                    raw_extraction, document_type
                )
                
                extraction_results[document_type] = {
                    "status": "success",
                    "raw_data": raw_extraction,
                    "structured_data": structured_data,
                    "extracted_at": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                extraction_results[document_type] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return {
            "agent_name": self.agent_name,
            "application_id": application_id,
            "extraction_results": extraction_results,
            "total_documents_processed": len(documents),
            "successful_extractions": len([r for r in extraction_results.values() 
                                         if r.get("status") == "success"])
        }
    
    async def _extract_structured_data(
        self, 
        raw_data: Dict[str, Any], 
        document_type: str
    ) -> Dict[str, Any]:
        """Extract structured data using LLM analysis"""
        
        template = self.extraction_templates.get(document_type, {})
        required_fields = template.get("required_fields", [])
        
        # Create specialized prompts for each document type
        if document_type == "bank_statement":
            return await self._extract_bank_statement_data(raw_data, required_fields)
        elif document_type == "emirates_id":
            return await self._extract_emirates_id_data(raw_data, required_fields)
        elif document_type == "resume":
            return await self._extract_resume_data(raw_data, required_fields)
        elif document_type == "credit_report":
            return await self._extract_credit_report_data(raw_data, required_fields)
        elif document_type == "assets":
            return await self._extract_assets_data(raw_data, required_fields)
        else:
            return await self._extract_generic_data(raw_data, required_fields)
    
    async def _extract_bank_statement_data(
        self, 
        raw_data: Dict[str, Any], 
        required_fields: List[str]
    ) -> Dict[str, Any]:
        """Extract structured data from bank statements"""
        
        # Get text content
        text_content = raw_data.get("raw_data", {}).get("full_text", "")
        processed_data = raw_data.get("processed_data", {})
        
        system_prompt = """You are a financial analyst AI specializing in bank statement analysis. 
        Extract accurate financial information and provide numerical values where possible."""
        
        task = "Analyze this bank statement and extract key financial metrics"
        
        context = {
            "document_text": text_content[:3000],  # Limit text length
            "processed_summary": processed_data,
            "required_fields": required_fields
        }
        
        output_format = """{
    "monthly_income": {
        "amount": float,
        "frequency": "monthly/bi-weekly/weekly",
        "source": "salary/benefits/other",
        "stability": "stable/irregular"
    },
    "monthly_expenses": {
        "total": float,
        "categories": {
            "housing": float,
            "utilities": float,
            "food": float,
            "transportation": float,
            "other": float
        }
    },
    "account_balance": {
        "average": float,
        "minimum": float,
        "maximum": float
    },
    "financial_behavior": {
        "savings_rate": float,
        "transaction_frequency": int,
        "large_transactions": int
    },
    "risk_indicators": {
        "overdrafts": int,
        "returned_payments": int,
        "low_balance_days": int
    }
}"""
        
        prompt = self.create_structured_prompt(task, context, output_format)
        
        llm_response = await self.invoke_llm(prompt, system_prompt)
        
        if llm_response["status"] == "success":
            structured_data = self.extract_json_from_response(llm_response["response"])
            if structured_data:
                return structured_data
        
        # Fallback to basic pattern-based extraction
        return self._fallback_bank_statement_extraction(processed_data)
    
    async def _extract_emirates_id_data(
        self, 
        raw_data: Dict[str, Any], 
        required_fields: List[str]
    ) -> Dict[str, Any]:
        """Extract structured data from Emirates ID"""
        
        text_content = raw_data.get("raw_data", {}).get("text", "")
        
        system_prompt = """You are an identity verification specialist. Extract personal 
        information from Emirates ID documents with high accuracy."""
        
        task = "Extract identity information from this Emirates ID document"
        
        context = {
            "document_text": text_content,
            "ocr_confidence": raw_data.get("raw_data", {}).get("ocr_confidence", 0)
        }
        
        output_format = """{
    "personal_info": {
        "full_name": "string",
        "emirates_id_number": "string (format: XXX-XXXX-XXXXXXX-X)",
        "nationality": "string",
        "date_of_birth": "YYYY-MM-DD",
        "gender": "Male/Female",
        "address": "string",
        "emirate": "string"
    },
    "document_info": {
        "issue_date": "YYYY-MM-DD",
        "expiry_date": "YYYY-MM-DD",
        "document_status": "Valid/Expired/Invalid"
    },
    "extraction_confidence": float
}"""
        
        prompt = self.create_structured_prompt(task, context, output_format)
        
        llm_response = await self.invoke_llm(prompt, system_prompt)
        
        if llm_response["status"] == "success":
            structured_data = self.extract_json_from_response(llm_response["response"])
            if structured_data:
                return structured_data
        
        return self._fallback_emirates_id_extraction(text_content)
    
    async def _extract_resume_data(
        self, 
        raw_data: Dict[str, Any], 
        required_fields: List[str]
    ) -> Dict[str, Any]:
        """Extract structured data from resume"""
        
        text_content = raw_data.get("raw_data", {}).get("full_text", "")
        
        system_prompt = """You are an HR specialist AI. Extract comprehensive career 
        information from resumes to assess employment history and qualifications."""
        
        task = "Extract career information from this resume"
        
        context = {
            "document_text": text_content[:3000],
            "required_fields": required_fields
        }
        
        output_format = """{
    "personal_info": {
        "name": "string",
        "email": "string", 
        "phone": "string"
    },
    "work_experience": [
        {
            "position": "string",
            "company": "string",
            "duration_months": int,
            "start_date": "YYYY-MM",
            "end_date": "YYYY-MM or Present",
            "industry": "string"
        }
    ],
    "education": [
        {
            "degree": "string",
            "institution": "string",
            "graduation_year": int,
            "field_of_study": "string"
        }
    ],
    "skills": ["string"],
    "career_summary": {
        "total_experience_months": int,
        "current_employment_status": "employed/unemployed",
        "highest_education": "string",
        "industry_experience": ["string"]
    }
}"""
        
        prompt = self.create_structured_prompt(task, context, output_format)
        
        llm_response = await self.invoke_llm(prompt, system_prompt)
        
        if llm_response["status"] == "success":
            structured_data = self.extract_json_from_response(llm_response["response"])
            if structured_data:
                return structured_data
        
        return self._fallback_resume_extraction(text_content)
    
    async def _extract_credit_report_data(
        self, 
        raw_data: Dict[str, Any], 
        required_fields: List[str]
    ) -> Dict[str, Any]:
        """Extract structured data from credit report"""
        
        text_content = raw_data.get("raw_data", {}).get("full_text", "")
        
        system_prompt = """You are a credit analysis specialist. Extract accurate credit 
        information for financial risk assessment."""
        
        task = "Extract credit information from this credit report"
        
        context = {
            "document_text": text_content[:3000],
            "required_fields": required_fields
        }
        
        output_format = """{
    "credit_score": {
        "score": int,
        "rating": "Excellent/Good/Fair/Poor",
        "report_date": "YYYY-MM-DD"
    },
    "credit_accounts": [
        {
            "account_type": "string",
            "balance": float,
            "credit_limit": float,
            "payment_history": "Good/Fair/Poor",
            "account_age_months": int
        }
    ],
    "credit_summary": {
        "total_debt": float,
        "total_credit_limit": float,
        "credit_utilization_percent": float,
        "number_of_accounts": int,
        "recent_inquiries": int
    },
    "risk_factors": {
        "late_payments": int,
        "defaults": int,
        "bankruptcies": int
    }
}"""
        
        prompt = self.create_structured_prompt(task, context, output_format)
        
        llm_response = await self.invoke_llm(prompt, system_prompt)
        
        if llm_response["status"] == "success":
            structured_data = self.extract_json_from_response(llm_response["response"])
            if structured_data:
                return structured_data
        
        return self._fallback_credit_report_extraction(text_content)
    
    async def _extract_assets_data(
        self, 
        raw_data: Dict[str, Any], 
        required_fields: List[str]
    ) -> Dict[str, Any]:
        """Extract structured data from assets/liabilities document"""
        
        # For tabular data (Excel/CSV), we have structured data already
        if raw_data.get("raw_data", {}).get("type") in ["excel", "csv"]:
            return self._process_tabular_assets_data(raw_data)
        
        # For text-based documents, use LLM extraction
        text_content = raw_data.get("raw_data", {}).get("full_text", "")
        
        system_prompt = """You are a financial analyst specializing in asset and liability 
        assessment. Extract and categorize financial holdings accurately."""
        
        task = "Extract assets and liabilities information"
        
        context = {
            "document_text": text_content,
            "required_fields": required_fields
        }
        
        output_format = """{
    "assets": {
        "cash_and_savings": float,
        "investments": float,
        "real_estate": float,
        "vehicles": float,
        "other_assets": float,
        "total_assets": float
    },
    "liabilities": {
        "credit_cards": float,
        "loans": float,
        "mortgages": float,
        "other_debts": float,
        "total_liabilities": float
    },
    "net_worth": float,
    "liquidity_ratio": float,
    "debt_to_asset_ratio": float
}"""
        
        prompt = self.create_structured_prompt(task, context, output_format)
        
        llm_response = await self.invoke_llm(prompt, system_prompt)
        
        if llm_response["status"] == "success":
            structured_data = self.extract_json_from_response(llm_response["response"])
            if structured_data:
                return structured_data
        
        return {"error": "Could not extract assets/liabilities data"}
    
    async def _extract_generic_data(
        self, 
        raw_data: Dict[str, Any], 
        required_fields: List[str]
    ) -> Dict[str, Any]:
        """Generic extraction for unspecified document types"""
        
        text_content = raw_data.get("raw_data", {}).get("full_text", "") or raw_data.get("raw_data", {}).get("text", "")
        
        return {
            "document_type": "generic",
            "text_content": text_content[:500],  # First 500 characters
            "extracted_fields": {},
            "note": "Generic extraction performed - specific parser not available"
        }
    
    # Fallback extraction methods using pattern matching
    
    def _fallback_bank_statement_extraction(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic pattern-based bank statement extraction"""
        financial_summary = processed_data.get("financial_summary", {})
        
        amounts = financial_summary.get("detected_amounts", [])
        amounts_float = []
        
        for amount in amounts:
            try:
                # Remove commas and convert to float
                clean_amount = float(amount.replace(",", ""))
                amounts_float.append(clean_amount)
            except (ValueError, AttributeError):
                continue
        
        if amounts_float:
            return {
                "monthly_income": {"amount": max(amounts_float), "confidence": "low"},
                "monthly_expenses": {"total": sum(amounts_float) / len(amounts_float), "confidence": "low"},
                "detected_amounts": amounts_float[:5],
                "extraction_method": "pattern_based"
            }
        
        return {"error": "No financial data detected"}
    
    def _fallback_emirates_id_extraction(self, text_content: str) -> Dict[str, Any]:
        """Basic pattern-based Emirates ID extraction"""
        # Simple regex patterns for Emirates ID
        emirates_id_pattern = r'\b\d{3}-\d{4}-\d{7}-\d{1}\b'
        emirates_ids = re.findall(emirates_id_pattern, text_content)
        
        return {
            "personal_info": {
                "emirates_id_detected": len(emirates_ids) > 0,
                "emirates_id_numbers": emirates_ids,
                "text_length": len(text_content)
            },
            "extraction_method": "pattern_based",
            "confidence": "low"
        }
    
    def _fallback_resume_extraction(self, text_content: str) -> Dict[str, Any]:
        """Basic pattern-based resume extraction"""
        # Simple keyword matching
        experience_keywords = ["experience", "work", "employment", "career"]
        education_keywords = ["education", "degree", "university", "college"]
        
        return {
            "career_summary": {
                "has_experience_keywords": any(kw in text_content.lower() for kw in experience_keywords),
                "has_education_keywords": any(kw in text_content.lower() for kw in education_keywords),
                "text_length": len(text_content)
            },
            "extraction_method": "pattern_based",
            "confidence": "low"
        }
    
    def _fallback_credit_report_extraction(self, text_content: str) -> Dict[str, Any]:
        """Basic pattern-based credit report extraction"""
        # Look for score patterns
        score_pattern = r'\b[3-8]\d{2}\b'  # Credit scores typically 300-850
        potential_scores = re.findall(score_pattern, text_content)
        
        return {
            "credit_info": {
                "potential_scores": potential_scores,
                "has_credit_keywords": any(kw in text_content.lower() 
                                         for kw in ["credit", "score", "debt", "payment"]),
                "text_length": len(text_content)
            },
            "extraction_method": "pattern_based",
            "confidence": "low"
        }
    
    def _process_tabular_assets_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process assets data from tabular format (Excel/CSV)"""
        data = raw_data.get("raw_data", {}).get("data", {})
        
        assets_total = 0
        liabilities_total = 0
        
        # For Excel files, process all sheets
        if raw_data.get("raw_data", {}).get("type") == "excel":
            for sheet_name, sheet_data in data.items():
                rows = sheet_data.get("data", [])
                for row in rows:
                    category = row.get("Category", "").lower()
                    value = row.get("Value", 0)
                    
                    try:
                        value = float(value)
                        if "asset" in category:
                            assets_total += value
                        elif "liability" in category or "debt" in category:
                            liabilities_total += value
                    except (ValueError, TypeError):
                        continue
        
        # For CSV files
        else:
            rows = data.get("data", [])
            for row in rows:
                category = row.get("Category", "").lower()
                value = row.get("Value", 0)
                
                try:
                    value = float(value)
                    if "asset" in category:
                        assets_total += value
                    elif "liability" in category or "debt" in category:
                        liabilities_total += value
                except (ValueError, TypeError):
                    continue
        
        net_worth = assets_total - liabilities_total
        
        return {
            "assets": {
                "total_assets": assets_total
            },
            "liabilities": {
                "total_liabilities": liabilities_total
            },
            "net_worth": net_worth,
            "debt_to_asset_ratio": liabilities_total / max(assets_total, 1),
            "extraction_method": "tabular_processing"
        } 