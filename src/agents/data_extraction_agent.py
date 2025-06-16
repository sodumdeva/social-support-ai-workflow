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
import os
import sys
import asyncio
import base64
import requests
from PIL import Image
import io
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import logging configuration
from src.utils.logging_config import get_logger, WorkflowLogger

# Setup logging
logger = get_logger("data_extraction_agent")
demo_logger = WorkflowLogger("data_extraction")

from .base_agent import BaseAgent
from src.data.document_processor import DocumentProcessor


class DataExtractionAgent(BaseAgent):
    """
    Document Processing Agent with OCR and LLM Analysis
    
    Processes various document types (Emirates ID, bank statements, resumes, etc.)
    using Tesseract OCR and local LLM models for structured data extraction.
    Performs verification against user-provided information.
    """
    
    def __init__(self):
        super().__init__("DataExtractionAgent")
        self.document_processor = DocumentProcessor()
        self.ollama_base_url = "http://localhost:11434"
        self.multimodal_model = "llava:7b"  # Using LLaVA for multimodal processing
        
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
        
        # Document type specific prompts for data extraction and verification
        self.document_prompts = {
            "emirates_id": """
            You are analyzing an Emirates ID document for a social support application. Extract ONLY the information that is clearly visible and readable.

            Extract the following information in JSON format:
            {
                "personal_info": {
                    "full_name": "Complete name as written on ID",
                    "emirates_id_number": "15-digit ID number (format: XXX-XXXX-XXXXXXX-X)",
                    "nationality": "Nationality as shown",
                    "date_of_birth": "Date in DD/MM/YYYY format",
                    "gender": "Male or Female",
                    "place_of_birth": "Birth place if visible"
                },
                "document_verification": {
                    "document_type": "Emirates ID",
                    "issue_date": "Issue date if visible (DD/MM/YYYY)",
                    "expiry_date": "Expiry date if visible (DD/MM/YYYY)",
                    "document_quality": "Clear/Blurry/Partially_visible",
                    "security_features_visible": "Yes/No/Partial"
                },
                "extraction_notes": {
                    "confidence_level": "High/Medium/Low",
                    "unreadable_fields": ["list any fields that couldn't be read"],
                    "verification_concerns": ["any concerns about document authenticity"]
                }
            }

            CRITICAL INSTRUCTIONS:
            - Use "Not visible" for any field that cannot be clearly read
            - Do not guess or infer information
            - Focus on accuracy over completeness
            - Note any signs of tampering or poor image quality
            """,
            
            "bank_statement": """
            You are analyzing a bank statement for income and financial verification. Extract key financial data that can verify the applicant's financial situation.

            Extract the following information in JSON format:
            {
                "account_holder_info": {
                    "account_holder_name": "Name on the account",
                    "account_number_last4": "Last 4 digits of account number only",
                    "bank_name": "Name of the bank",
                    "statement_period": "Statement period (MM/YYYY to MM/YYYY)"
                },
                "income_verification": {
                    "monthly_salary": "Regular salary amount if identifiable",
                    "salary_frequency": "Monthly/Bi-weekly/Weekly",
                    "employer_name": "Employer name from salary deposits",
                    "other_income_sources": ["List other regular income sources"],
                    "total_monthly_income": "Total estimated monthly income",
                    "income_consistency": "Consistent/Variable/Irregular"
                },
                "financial_summary": {
                    "opening_balance": "Opening balance amount",
                    "closing_balance": "Closing balance amount",
                    "average_balance": "Average balance if calculable",
                    "total_credits": "Total money coming in",
                    "total_debits": "Total money going out",
                    "net_cash_flow": "Credits minus debits"
                },
                "spending_patterns": {
                    "housing_payments": "Rent/mortgage payments if visible",
                    "utility_payments": "Utility bills if identifiable",
                    "loan_payments": "Any loan payments",
                    "large_expenses": ["Any unusually large transactions"],
                    "financial_stress_indicators": ["Overdrafts, returned payments, etc."]
                },
                "verification_data": {
                    "statement_authenticity": "Appears_genuine/Suspicious/Cannot_determine",
                    "data_consistency": "Consistent/Inconsistent/Mixed",
                    "extraction_confidence": "High/Medium/Low"
                }
            }

            FOCUS ON:
            - Accurate salary/income identification
            - Employer verification through transaction descriptions
            - Financial stability indicators
            - Any red flags or inconsistencies
            """,
            
            "resume": """
            You are analyzing a resume/CV for employment verification. Extract employment history and professional information.

            Extract the following information in JSON format:
            {
                "personal_contact": {
                    "full_name": "Name from resume",
                    "email_address": "Email if provided",
                    "phone_number": "Phone number if provided",
                    "location": "City/Country if provided"
                },
                "current_employment": {
                    "current_position": "Current job title",
                    "current_employer": "Current company name",
                    "employment_status": "Currently_employed/Recently_unemployed/Self_employed",
                    "start_date": "When current job started (MM/YYYY)",
                    "current_salary_mentioned": "Any salary information mentioned",
                    "job_responsibilities": ["Key responsibilities listed"]
                },
                "employment_history": [
                    {
                        "position": "Job title",
                        "company": "Company name",
                        "duration": "Duration in months",
                        "start_date": "MM/YYYY",
                        "end_date": "MM/YYYY or Present",
                        "industry": "Industry sector"
                    }
                ],
                "professional_summary": {
                    "total_experience_years": "Total years of experience",
                    "total_experience_months": "Total months (years × 12 + additional months)",
                    "career_progression": "Upward/Lateral/Downward/Mixed",
                    "employment_gaps": ["Any significant gaps in employment"],
                    "industry_expertise": ["Main industries worked in"]
                },
                "qualifications": {
                    "highest_education": "Highest degree/qualification",
                    "relevant_certifications": ["Professional certifications"],
                    "key_skills": ["Technical and professional skills"],
                    "languages": ["Languages if mentioned"]
                },
                "verification_notes": {
                    "employment_verifiability": "Easy/Moderate/Difficult",
                    "consistency_check": "Consistent/Some_gaps/Inconsistent",
                    "professional_level": "Entry/Mid/Senior/Executive"
                }
            }

            VERIFICATION FOCUS:
            - Match name with other documents
            - Verify employment duration calculations
            - Check for employment gaps or inconsistencies
            - Assess professional credibility
            """,
            
            "salary_certificate": """
            You are analyzing a salary certificate for income verification. This is a critical document for verifying applicant's income claims.

            Extract the following information in JSON format:
            {
                "employee_info": {
                    "employee_name": "Full name on certificate",
                    "employee_id": "Employee ID if provided",
                    "position": "Job title/position",
                    "department": "Department if mentioned"
                },
                "employer_info": {
                    "company_name": "Official company name",
                    "company_address": "Company address if provided",
                    "hr_contact": "HR contact information if provided",
                    "company_stamp": "Company stamp visible (Yes/No)"
                },
                "salary_details": {
                    "basic_salary": "Basic salary amount",
                    "allowances": {
                        "housing_allowance": "Housing allowance if separate",
                        "transport_allowance": "Transport allowance if separate",
                        "other_allowances": ["Other allowances listed"]
                    },
                    "gross_salary": "Total gross salary",
                    "currency": "Currency (AED/USD/etc.)",
                    "salary_frequency": "Monthly/Annual"
                },
                "employment_details": {
                    "employment_start_date": "Date of joining",
                    "employment_type": "Permanent/Contract/Temporary",
                    "contract_duration": "Contract period if applicable",
                    "probation_status": "Confirmed/On_probation/Not_mentioned"
                },
                "document_verification": {
                    "issue_date": "Certificate issue date",
                    "authorized_signatory": "Signed by (name/title)",
                    "official_letterhead": "On company letterhead (Yes/No)",
                    "document_authenticity": "Appears_genuine/Questionable/Cannot_verify",
                    "verification_possible": "Yes/No - can this be verified with employer"
                }
            }

            CRITICAL FOR VERIFICATION:
            - Exact salary amounts for cross-checking
            - Employer details for verification calls
            - Document authenticity indicators
            - Employment status confirmation
            """,
            
            "other": """
            You are analyzing a document that may contain relevant information for a social support application. Extract any useful personal, financial, or employment information.

            Extract the following information in JSON format:
            {
                "document_identification": {
                    "document_type": "Best guess of document type",
                    "document_purpose": "What this document appears to be for",
                    "issuing_authority": "Who issued this document if identifiable"
                },
                "personal_information": {
                    "names_found": ["Any names found in the document"],
                    "id_numbers": ["Any ID numbers found"],
                    "contact_info": ["Phone numbers, emails, addresses"],
                    "dates": ["Important dates found"]
                },
                "financial_information": {
                    "amounts": ["Any monetary amounts found"],
                    "income_references": ["Any references to income/salary"],
                    "financial_obligations": ["Any debts, loans, payments mentioned"]
                },
                "employment_information": {
                    "employer_names": ["Any company/employer names"],
                    "job_titles": ["Any job positions mentioned"],
                    "employment_dates": ["Any employment-related dates"]
                },
                "verification_relevance": {
                    "relevance_score": "High/Medium/Low",
                    "verification_value": "What can this document help verify",
                    "cross_check_potential": ["What information can be cross-checked"]
                }
            }

            Extract any information that might be useful for verifying an applicant's identity, income, or circumstances.
            """
        }
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process documents using multimodal LLM"""
        
        try:
            logger.info("🔍 Starting enhanced multimodal document processing")
            
            documents = request.get("documents", [])
            if not documents:
                return {
                    "status": "error",
                    "error": "No documents provided for processing",
                    "agent_name": self.agent_name
                }
            
            extraction_results = {}
            
            for doc_info in documents:
                file_path = doc_info.get("file_path")
                doc_type = doc_info.get("document_type", "other")
                
                if not file_path or not os.path.exists(file_path):
                    logger.error(f"❌ Document file not found: {file_path}")
                    continue
                
                logger.info(f"📄 Processing {doc_type}: {os.path.basename(file_path)}")
                
                # Process document with multimodal LLM
                extraction_result = await self._process_document_with_llm(file_path, doc_type)
                extraction_results[doc_type] = extraction_result
                
                # Log extraction results
                if extraction_result.get("status") == "success":
                    logger.info(f"✅ Successfully processed {doc_type}")
                    logger.info(f"📊 Extracted data: {json.dumps(extraction_result.get('structured_data', {}), indent=2)}")
                else:
                    logger.error(f"❌ Failed to process {doc_type}: {extraction_result.get('error', 'Unknown error')}")
            
            return {
                "status": "success",
                "extraction_results": extraction_results,
                "agent_name": self.agent_name,
                "processed_at": datetime.now().isoformat(),
                "total_documents": len(documents),
                "successful_extractions": len([r for r in extraction_results.values() if r.get("status") == "success"])
            }
            
        except Exception as e:
            error_msg = f"Error in document processing: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "agent_name": self.agent_name
            }
    
    async def _process_document_with_llm(self, file_path: str, doc_type: str) -> Dict[str, Any]:
        """Process a single document using hybrid OCR + LLM approach"""
        
        try:
            filename = os.path.basename(file_path)
            demo_logger.log_step("SYSTEM_CHECK", f"🔍 Starting hybrid OCR + LLM processing for {filename}")
            
            # STEP 1: Extract text using Tesseract OCR
            demo_logger.log_step("OCR_PROCESSING", f"📄 Extracting text using Tesseract OCR...")
            ocr_result = await self._extract_text_with_tesseract(file_path)
            
            if ocr_result.get("status") != "success":
                demo_logger.log_step("OCR_ERROR", f"❌ OCR extraction failed: {ocr_result.get('error')}", "ERROR")
                return {
                    "status": "error",
                    "error": f"OCR extraction failed: {ocr_result.get('error')}",
                    "extraction_confidence": 0.0
                }
            
            extracted_text = ocr_result.get("text", "")
            ocr_confidence = ocr_result.get("confidence", 0)
            
            demo_logger.log_step("OCR_SUCCESS", f"✅ OCR extracted {len(extracted_text)} characters (confidence: {ocr_confidence:.1f}%)")
            logger.info(f"📄 OCR extracted text length: {len(extracted_text)} chars, confidence: {ocr_confidence:.1f}%")
            
            # DEBUG: Log the full extracted text for debugging
            demo_logger.log_step("OCR_TEXT_DEBUG", f"📝 Full OCR extracted text:\n{extracted_text}")
            logger.info(f"📝 Full OCR extracted text:\n{extracted_text}")
            
            # STEP 2: Use LLM to structure and verify the extracted text
            demo_logger.log_step("LLM_VERIFICATION", f"🤖 Using LLM to structure and verify OCR data...")
            
            # Check if LLM is available
            if not await self._check_ollama_status():
                demo_logger.log_step("LLM_FALLBACK", "⚠️ LLM not available, using OCR-only parsing", "WARNING")
                # Fallback to OCR-only structured parsing
                structured_data = await self._parse_ocr_text_only(extracted_text, doc_type)
                return {
                    "status": "success",
                    "structured_data": structured_data,
                    "extraction_confidence": ocr_confidence / 100.0,
                    "processing_method": "ocr_only",
                    "ocr_confidence": ocr_confidence,
                    "document_type": doc_type
                }
            
            # Create verification prompt for LLM
            verification_prompt = self._create_verification_prompt(extracted_text, doc_type)
            
            # DEBUG: Log the verification prompt for debugging
            demo_logger.log_step("LLM_PROMPT_DEBUG", f"📝 LLM verification prompt:\n{verification_prompt[:1000]}..." if len(verification_prompt) > 1000 else f"📝 LLM verification prompt:\n{verification_prompt}")
            
            demo_logger.log_step("LLM_CALL", f"🤖 Calling LLM for text verification and structuring...")
            logger.info(f"🤖 Calling LLM for text verification and structuring...")
            
            # Call LLM for verification (using text model, not multimodal)
            llm_result = await self._call_ollama_text(verification_prompt)
            
            if llm_result.get("status") == "success":
                processing_time = llm_result.get("processing_time_ms", 0)
                response_length = len(llm_result.get("content", ""))
                
                demo_logger.log_llm_call(self.model_name, doc_type, processing_time)
                demo_logger.log_step("LLM_SUCCESS", f"✅ LLM verification completed ({response_length} chars, {processing_time:.0f}ms)")
                
                # DEBUG: Log the LLM response for debugging
                llm_content = llm_result.get("content", "")
                demo_logger.log_step("LLM_RESPONSE_DEBUG", f"📝 LLM response:\n{llm_content}")
                logger.info(f"📝 LLM response:\n{llm_content}")
                
                # Parse the LLM response
                demo_logger.log_step("JSON_PARSING", f"📊 Parsing LLM response for structured data...")
                structured_data = await self._parse_llm_response(llm_result["content"], doc_type)
                
                if "parsing_error" not in structured_data:
                    demo_logger.log_step("PARSING_SUCCESS", f"✅ Successfully parsed JSON from LLM response")
                    logger.info(f"✅ Successfully parsed JSON from LLM response")
                else:
                    demo_logger.log_step("PARSING_FALLBACK", f"⚠️ JSON parsing failed, using OCR fallback", "WARNING")
                    # Fallback to OCR-only parsing
                    structured_data = await self._parse_ocr_text_only(extracted_text, doc_type)
                
                # Calculate combined confidence (OCR + LLM verification)
                llm_confidence = self._calculate_confidence(structured_data)
                combined_confidence = (ocr_confidence / 100.0 * 0.6) + (llm_confidence * 0.4)  # Weight OCR 60%, LLM 40%
                
                demo_logger.log_step("CONFIDENCE_CALC", f"🎯 Combined confidence: {combined_confidence:.2f} (OCR: {ocr_confidence:.1f}%, LLM: {llm_confidence:.2f})")
                
                return {
                    "status": "success",
                    "structured_data": structured_data,
                    "extraction_confidence": combined_confidence,
                    "processing_time_ms": processing_time,
                    "processing_method": "hybrid_ocr_llm",
                    "ocr_confidence": ocr_confidence,
                    "llm_confidence": llm_confidence,
                    "extracted_text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,  # Include sample of extracted text
                    "llm_model": self.model_name,
                    "document_type": doc_type
                }
            else:
                error_msg = llm_result.get("error", "Unknown LLM error")
                demo_logger.log_step("LLM_FALLBACK", f"⚠️ LLM failed, using OCR-only: {error_msg}", "WARNING")
                
                # Fallback to OCR-only structured parsing
                structured_data = await self._parse_ocr_text_only(extracted_text, doc_type)
                return {
                    "status": "success",
                    "structured_data": structured_data,
                    "extraction_confidence": ocr_confidence / 100.0,
                    "processing_method": "ocr_fallback",
                    "ocr_confidence": ocr_confidence,
                    "extracted_text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                    "fallback_reason": error_msg,
                    "document_type": doc_type
                }
                
        except Exception as e:
            error_msg = f"Hybrid document processing error: {str(e)}"
            demo_logger.log_step("PROCESSING_ERROR", f"❌ {error_msg}", "ERROR")
            logger.error(f"❌ Error in hybrid document processing: {str(e)}")
            return {
                "status": "error",
                "error": error_msg,
                "extraction_confidence": 0.0
            }
    
    async def _check_ollama_status(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def _check_model_availability(self) -> bool:
        """Check if the multimodal model is available"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model.get("name", "") for model in models]
                return any(self.multimodal_model in model for model in available_models)
            return False
        except:
            return False
    
    async def _image_to_base64(self, file_path: str) -> Optional[str]:
        """Convert image file to base64 string"""
        try:
            # Open and potentially resize image for better processing
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (LLaVA works better with reasonable sizes)
                max_size = 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_bytes = buffer.getvalue()
                
                return base64.b64encode(img_bytes).decode('utf-8')
                
        except Exception as e:
            logger.error(f"❌ Error converting image to base64: {str(e)}")
            return None
    
    async def _call_ollama_multimodal(self, prompt: str, image_base64: str) -> Dict[str, Any]:
        """Call Ollama API with multimodal input"""
        
        try:
            start_time = datetime.now()
            
            payload = {
                "model": self.multimodal_model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent extraction
                    "top_p": 0.9,
                    "num_predict": 1000  # Allow longer responses for detailed extraction
                }
            }
            
            logger.info(f"🔄 Calling Ollama API with {self.multimodal_model}...")
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=300  # Longer timeout for multimodal processing
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                logger.info(f"✅ LLM processing completed in {processing_time:.0f}ms")
                logger.info(f"📝 LLM Response length: {len(content)} characters")
                
                return {
                    "status": "success",
                    "content": content,
                    "processing_time_ms": processing_time
                }
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(f"❌ {error_msg}")
                return {
                    "status": "error",
                    "error": error_msg
                }
                
        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "error": "LLM processing timeout - document may be too complex"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"API call failed: {str(e)}"
            }
    
    async def _parse_llm_response(self, llm_response: str, doc_type: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured data"""
        
        try:
            # Try to extract JSON from the response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                structured_data = json.loads(json_str)
                
                logger.info(f"✅ Successfully parsed JSON from LLM response")
                return structured_data
            else:
                # Fallback: try to extract key information using text parsing
                logger.warning("⚠️ No valid JSON found, attempting text parsing...")
                return self._fallback_text_parsing(llm_response, doc_type)
                
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parsing failed: {str(e)}, attempting text parsing...")
            return self._fallback_text_parsing(llm_response, doc_type)
        except Exception as e:
            logger.error(f"❌ Error parsing LLM response: {str(e)}")
            return {"parsing_error": str(e), "raw_response": llm_response}
    
    def _fallback_text_parsing(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Fallback text parsing when JSON extraction fails"""
        
        extracted_info = {
            "parsing_method": "text_fallback",
            "document_type": doc_type,
            "raw_text": text
        }
        
        # Basic text extraction patterns
        import re
        
        # Extract names (capitalized words)
        names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if names:
            extracted_info["potential_names"] = names
        
        # Extract numbers that could be IDs, amounts, etc.
        numbers = re.findall(r'\b\d{3,}\b', text)
        if numbers:
            extracted_info["potential_numbers"] = numbers
        
        # Extract dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        if dates:
            extracted_info["potential_dates"] = dates
        
        # Extract amounts (with currency symbols)
        amounts = re.findall(r'[A-Z]{3}\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*[A-Z]{3}', text)
        if amounts:
            extracted_info["potential_amounts"] = amounts
        
        return extracted_info
    
    def _calculate_confidence(self, structured_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on extracted data completeness"""
        
        if not structured_data or "parsing_error" in structured_data:
            return 0.0
        
        if structured_data.get("parsing_method") == "text_fallback":
            return 0.3  # Lower confidence for fallback parsing
        
        # Count non-empty fields
        total_fields = 0
        filled_fields = 0
        
        def count_fields(obj, path=""):
            nonlocal total_fields, filled_fields
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        count_fields(value, f"{path}.{key}" if path else key)
                    else:
                        total_fields += 1
                        if value and str(value).strip() and str(value).lower() not in ["not visible", "n/a", "unknown"]:
                            filled_fields += 1
            elif isinstance(obj, list):
                for item in obj:
                    count_fields(item, path)
        
        count_fields(structured_data)
        
        if total_fields == 0:
            return 0.0
        
        confidence = filled_fields / total_fields
        return min(confidence, 1.0)  # Cap at 1.0
    
    async def _extract_text_with_tesseract(self, file_path: str) -> Dict[str, Any]:
        """Extract text from image using Tesseract OCR with multiple preprocessing techniques"""
        
        try:
            # Import our independent extraction script functionality
            import cv2
            import pytesseract
            import numpy as np
            
            # Load and preprocess image
            image = cv2.imread(file_path)
            if image is None:
                return {
                    "status": "error",
                    "error": f"Could not load image from {file_path}"
                }
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply multiple preprocessing techniques and find the best result
            processed_images = []
            
            # 1. Basic thresholding
            _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("basic_threshold", thresh1))
            
            # 2. Adaptive thresholding
            thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            processed_images.append(("adaptive_threshold", thresh2))
            
            # 3. Noise removal + thresholding
            denoised = cv2.medianBlur(gray, 3)
            _, thresh3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("denoised_threshold", thresh3))
            
            # Extract text using different preprocessing methods
            best_result = None
            best_confidence = 0
            
            for method_name, processed_img in processed_images:
                try:
                    # Extract text
                    text = pytesseract.image_to_string(processed_img, config='--psm 6')
                    
                    # Get confidence data
                    ocr_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                    avg_confidence = np.mean(confidences) if confidences else 0
                    
                    # Keep the best result
                    if avg_confidence > best_confidence and len(text.strip()) > 10:
                        best_confidence = avg_confidence
                        best_result = {
                            "text": text,
                            "confidence": avg_confidence,
                            "method": method_name,
                            "text_length": len(text)
                        }
                        
                except Exception as e:
                    logger.warning(f"OCR method {method_name} failed: {str(e)}")
                    continue
            
            if best_result:
                return {
                    "status": "success",
                    **best_result
                }
            else:
                return {
                    "status": "error",
                    "error": "All OCR preprocessing methods failed"
                }
                
        except ImportError as e:
            return {
                "status": "error",
                "error": f"OCR dependencies not available: {str(e)}. Please install: pip install opencv-python pytesseract"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"OCR processing failed: {str(e)}"
            }
    
    def _create_verification_prompt(self, extracted_text: str, doc_type: str) -> str:
        """Create a prompt for LLM to verify and structure OCR-extracted text"""
        
        base_prompt = f"""
You are an expert document analyst. I have extracted text from a {doc_type} document using OCR. 
Please analyze this text and extract structured information in JSON format.

EXTRACTED TEXT:
{extracted_text}

Please provide a JSON response with the following structure based on the document type:
"""
        
        if doc_type == "emirates_id":
            structure_prompt = """
{
    "personal_info": {
        "full_name": "extracted name or null",
        "emirates_id_number": "extracted ID number or null", 
        "nationality": "extracted nationality or null",
        "date_of_birth": "extracted DOB or null",
        "gender": "extracted gender or null",
        "place_of_birth": "extracted place or null"
    },
    "document_verification": {
        "document_type": "Emirates ID",
        "issue_date": "extracted issue date or null",
        "expiry_date": "extracted expiry date or null",
        "document_quality": "assessment based on text clarity",
        "security_features_visible": "Yes/No based on text content"
    },
    "extraction_notes": {
        "confidence_level": "High/Medium/Low based on text clarity",
        "unreadable_fields": ["list of fields that couldn't be read"],
        "verification_concerns": ["any concerns about the document"]
    }
}
"""
        elif doc_type == "bank_statement":
            structure_prompt = """
{
    "account_info": {
        "account_holder_name": "extracted name or null",
        "account_number": "extracted account number or null",
        "bank_name": "extracted bank name or null",
        "statement_period": "extracted period or null"
    },
    "financial_data": {
        "opening_balance": "extracted opening balance or null",
        "closing_balance": "extracted closing balance or null",
        "total_credits": "sum of credit transactions or null",
        "total_debits": "sum of debit transactions or null",
        "salary_credits": ["list of salary-related credits"],
        "major_transactions": ["list of significant transactions"]
    },
    "verification_notes": {
        "confidence_level": "High/Medium/Low",
        "data_quality": "assessment of statement completeness",
        "concerns": ["any verification concerns"]
    }
}
"""
        else:
            structure_prompt = """
{
    "extracted_data": {
        "key_information": "main information found in the document",
        "names": ["any names found"],
        "numbers": ["any important numbers found"],
        "dates": ["any dates found"],
        "amounts": ["any monetary amounts found"]
    },
    "document_assessment": {
        "document_type": "best guess of document type",
        "confidence_level": "High/Medium/Low",
        "completeness": "assessment of information completeness"
    }
}
"""
        
        return base_prompt + structure_prompt + """

IMPORTANT INSTRUCTIONS:
1. Extract information ONLY from the provided text
2. Use "null" for fields that cannot be determined from the text
3. Be conservative - only extract information you are confident about
4. Provide valid JSON format
5. If text is unclear or garbled, note this in the confidence assessment
6. Focus on accuracy over completeness
"""
    
    async def _call_ollama_text(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama with text-only prompt (not multimodal)"""
        
        try:
            start_time = time.time()
            
            payload = {
                "model": self.model_name,  # Use regular text model, not multimodal
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent extraction
                    "top_p": 0.9,
                    "num_predict": 2000
                }
            }
            
            demo_logger.log_step("LLM_REQUEST", f"🔄 Calling Ollama API with {self.model_name}...")
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                demo_logger.log_step("LLM_SUCCESS", f"✅ LLM processing completed in {processing_time:.0f}ms")
                logger.info(f"✅ LLM processing completed in {processing_time:.0f}ms")
                logger.info(f"📝 LLM Response length: {len(content)} characters")
                
                return {
                    "status": "success",
                    "content": content,
                    "processing_time_ms": processing_time,
                    "model": self.model_name
                }
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                demo_logger.log_step("LLM_ERROR", f"❌ {error_msg}", "ERROR")
                return {
                    "status": "error",
                    "error": error_msg
                }
                
        except requests.exceptions.Timeout:
            error_msg = "LLM request timed out"
            demo_logger.log_step("LLM_TIMEOUT", f"⏰ {error_msg}", "ERROR")
            return {
                "status": "error",
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"LLM call failed: {str(e)}"
            demo_logger.log_step("LLM_ERROR", f"❌ {error_msg}", "ERROR")
            return {
                "status": "error",
                "error": error_msg
            }
    
    async def _parse_ocr_text_only(self, extracted_text: str, doc_type: str) -> Dict[str, Any]:
        """Parse OCR text using pattern matching when LLM is not available"""
        
        try:
            # Import our extraction script functionality
            from extract_name_salary import ImageDataExtractor
            
            # Create a temporary extractor instance
            extractor = ImageDataExtractor(debug=False)
            
            # Extract names and other data using pattern matching
            names = extractor.extract_names(extracted_text)
            salaries = extractor.extract_salaries(extracted_text)
            emirates_id = extractor.extract_emirates_id(extracted_text)
            
            # Structure the data based on document type
            if doc_type == "emirates_id":
                return {
                    "personal_info": {
                        "full_name": names[0]['name'] if names else None,
                        "emirates_id_number": emirates_id,
                        "nationality": "UAE" if "uae" in extracted_text.lower() or "emirates" in extracted_text.lower() else None,
                        "date_of_birth": None,  # Would need more sophisticated parsing
                        "gender": None,
                        "place_of_birth": None
                    },
                    "document_verification": {
                        "document_type": "Emirates ID",
                        "issue_date": None,
                        "expiry_date": None,
                        "document_quality": "Good" if len(extracted_text) > 100 else "Poor",
                        "security_features_visible": "Unknown"
                    },
                    "extraction_notes": {
                        "confidence_level": "Medium",
                        "unreadable_fields": ["date_of_birth", "issue_date", "expiry_date"],
                        "verification_concerns": ["OCR-only parsing - limited field extraction"]
                    }
                }
            elif doc_type == "bank_statement":
                return {
                    "account_info": {
                        "account_holder_name": names[0]['name'] if names else None,
                        "account_number": None,  # Would need pattern matching
                        "bank_name": "Emirates NBD" if "emirates nbd" in extracted_text.lower() else None,
                        "statement_period": None
                    },
                    "financial_data": {
                        "opening_balance": None,
                        "closing_balance": None,
                        "total_credits": None,
                        "total_debits": None,
                        "salary_credits": [s['formatted'] for s in salaries if s['confidence'] == 'high'],
                        "major_transactions": [s['formatted'] for s in salaries[:5]]  # Top 5 amounts
                    },
                    "verification_notes": {
                        "confidence_level": "Medium",
                        "data_quality": "Partial - OCR extraction only",
                        "concerns": ["Limited parsing without LLM verification"]
                    }
                }
            else:
                return {
                    "extracted_data": {
                        "key_information": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
                        "names": [n['name'] for n in names[:3]],  # Top 3 names
                        "numbers": [s['original_text'] for s in salaries[:5]],  # Top 5 numbers
                        "dates": [],  # Would need date pattern matching
                        "amounts": [s['formatted'] for s in salaries[:5]]
                    },
                    "document_assessment": {
                        "document_type": doc_type,
                        "confidence_level": "Medium",
                        "completeness": "Partial - pattern matching only"
                    }
                }
                
        except Exception as e:
            logger.warning(f"OCR-only parsing failed: {str(e)}")
            # Return minimal structure
            return {
                "extracted_data": {
                    "raw_text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                    "parsing_error": str(e)
                },
                "document_assessment": {
                    "document_type": doc_type,
                    "confidence_level": "Low",
                    "completeness": "Failed - fallback to raw text"
                }
            } 