"""
Real-time Document Processor for Social Support AI Workflow

Processes documents in real-time during conversation, extracting key information
using OCR, NLP, and structured data parsing.
"""
from typing import Dict, Any, List, Optional
import asyncio
import os
import sys
from datetime import datetime
import json
import re
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Document processing imports
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    import pytesseract
    import pandas as pd
    from pypdf2 import PdfReader
    import docx
except ImportError as e:
    print(f"Warning: Some document processing libraries not available: {e}")


class RealtimeDocumentProcessor:
    """Process documents in real-time during conversation"""
    
    def __init__(self):
        self.supported_formats = {
            'pdf': self.process_pdf,
            'png': self.process_image,
            'jpg': self.process_image,
            'jpeg': self.process_image,
            'xlsx': self.process_excel,
            'xls': self.process_excel,
            'docx': self.process_docx
        }
    
    async def process_emirates_id(self, file_path: str) -> Dict[str, Any]:
        """Extract data from Emirates ID using OCR and validation"""
        
        try:
            # Enhance image for better OCR
            enhanced_image = self.enhance_image_for_ocr(file_path)
            
            # OCR processing
            extracted_text = await self.extract_text_from_image(enhanced_image)
            
            # Parse Emirates ID format
            parsed_data = self.parse_emirates_id(extracted_text)
            
            # Validate format
            validation_result = self.validate_emirates_id(parsed_data)
            
            return {
                "status": "success",
                "data": parsed_data,
                "validation": validation_result,
                "confidence": validation_result.get("confidence", 0.8),
                "raw_text": extracted_text
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": {},
                "confidence": 0.0
            }
    
    async def process_bank_statement(self, file_path: str) -> Dict[str, Any]:
        """Extract financial data from bank statement"""
        
        try:
            file_ext = file_path.split('.')[-1].lower()
            
            if file_ext == 'pdf':
                text_content = await self.extract_text_from_pdf(file_path)
            else:
                text_content = await self.extract_text_from_image(file_path)
            
            # Parse financial data
            financial_data = self.parse_bank_statement(text_content)
            
            # Calculate income patterns
            income_analysis = self.analyze_income_patterns(financial_data)
            
            return {
                "status": "success", 
                "data": {
                    "monthly_income": income_analysis.get("average_monthly", 0),
                    "income_stability": income_analysis.get("stability_score", 0),
                    "account_balance": financial_data.get("balance", 0),
                    "transactions": financial_data.get("transactions", [])[:10],
                    "analysis": income_analysis
                },
                "confidence": income_analysis.get("confidence", 0.7)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": {},
                "confidence": 0.0
            }
    
    async def process_resume(self, file_path: str) -> Dict[str, Any]:
        """Extract career information from resume"""
        
        try:
            file_ext = file_path.split('.')[-1].lower()
            
            if file_ext == 'pdf':
                text_content = await self.extract_text_from_pdf(file_path)
            elif file_ext == 'docx':
                text_content = await self.extract_text_from_docx(file_path)
            else:
                text_content = await self.extract_text_from_image(file_path)
            
            # Parse resume content
            resume_data = self.parse_resume(text_content)
            
            return {
                "status": "success",
                "data": resume_data,
                "confidence": 0.8
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": {},
                "confidence": 0.0
            }
    
    async def process_credit_report(self, file_path: str) -> Dict[str, Any]:
        """Extract credit information from credit report"""
        
        try:
            file_ext = file_path.split('.')[-1].lower()
            
            if file_ext == 'pdf':
                text_content = await self.extract_text_from_pdf(file_path)
            else:
                text_content = await self.extract_text_from_image(file_path)
            
            # Parse credit data
            credit_data = self.parse_credit_report(text_content)
            
            return {
                "status": "success",
                "data": credit_data,
                "confidence": 0.7
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": {},
                "confidence": 0.0
            }
    
    async def process_assets_liabilities(self, file_path: str) -> Dict[str, Any]:
        """Extract assets and liabilities from spreadsheet"""
        
        try:
            # Process Excel file
            df = pd.read_excel(file_path)
            
            # Parse assets and liabilities
            assets_liabilities_data = self.parse_assets_liabilities_excel(df)
            
            return {
                "status": "success",
                "data": assets_liabilities_data,
                "confidence": 0.9  # High confidence for structured data
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": {},
                "confidence": 0.0
            }
    
    # Helper methods for document processing
    def enhance_image_for_ocr(self, image_path: str) -> str:
        """Enhance image quality for better OCR"""
        
        try:
            # Load image
            image = cv2.imread(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Enhance contrast
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(denoised)
            
            # Save enhanced image
            enhanced_path = image_path.replace('.', '_enhanced.')
            cv2.imwrite(enhanced_path, enhanced)
            
            return enhanced_path
            
        except Exception:
            # Return original path if enhancement fails
            return image_path
    
    async def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        
        try:
            # Use pytesseract for OCR
            text = pytesseract.image_to_string(Image.open(image_path))
            return text.strip()
            
        except Exception as e:
            raise Exception(f"OCR extraction failed: {str(e)}")
    
    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        
        try:
            text_content = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
            
            return text_content.strip()
            
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    async def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        
        try:
            doc = docx.Document(docx_path)
            text_content = ""
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            
            return text_content.strip()
            
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {str(e)}")
    
    def parse_emirates_id(self, text: str) -> Dict[str, Any]:
        """Parse Emirates ID information from extracted text"""
        
        data = {}
        
        # Extract ID number (XXX-XXXX-XXXXXXX-X)
        id_pattern = r'(\d{3}-?\d{4}-?\d{7}-?\d{1})'
        id_match = re.search(id_pattern, text.replace(" ", ""))
        if id_match:
            data["emirates_id"] = id_match.group(1)
        
        # Extract name (usually appears after "Name" or similar)
        name_patterns = [
            r'Name[:\s]+([A-Za-z\s]+)',
            r'الاسم[:\s]+([A-Za-z\s]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in name_patterns:
            name_match = re.search(pattern, text)
            if name_match:
                name = name_match.group(1).strip()
                if len(name.split()) >= 2:
                    data["name"] = name
                    break
        
        # Extract nationality
        nationality_patterns = [
            r'Nationality[:\s]+([A-Za-z\s]+)',
            r'الجنسية[:\s]+([A-Za-z\s]+)'
        ]
        
        for pattern in nationality_patterns:
            nat_match = re.search(pattern, text)
            if nat_match:
                data["nationality"] = nat_match.group(1).strip()
                break
        
        # Extract date of birth
        dob_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{1,2}-\d{1,2}-\d{4})',
            r'(\d{4}/\d{1,2}/\d{1,2})'
        ]
        
        for pattern in dob_patterns:
            dob_match = re.search(pattern, text)
            if dob_match:
                data["date_of_birth"] = dob_match.group(1)
                # Calculate age
                try:
                    from datetime import datetime
                    dob_str = dob_match.group(1)
                    # Handle different date formats
                    for date_format in ['%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d']:
                        try:
                            dob = datetime.strptime(dob_str, date_format)
                            age = datetime.now().year - dob.year
                            data["age"] = age
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass
                break
        
        return data
    
    def validate_emirates_id(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Emirates ID data"""
        
        validation = {
            "is_valid": True,
            "confidence": 1.0,
            "issues": []
        }
        
        # Check if ID number is present and valid format
        emirates_id = parsed_data.get("emirates_id", "")
        if not emirates_id:
            validation["issues"].append("Emirates ID number not found")
            validation["confidence"] -= 0.5
        elif not re.match(r'\d{3}-?\d{4}-?\d{7}-?\d{1}', emirates_id):
            validation["issues"].append("Invalid Emirates ID format")
            validation["confidence"] -= 0.3
        
        # Check if name is present
        if not parsed_data.get("name"):
            validation["issues"].append("Name not found")
            validation["confidence"] -= 0.3
        
        # Check age reasonableness
        age = parsed_data.get("age", 0)
        if age and (age < 0 or age > 120):
            validation["issues"].append("Unreasonable age detected")
            validation["confidence"] -= 0.2
        
        validation["is_valid"] = validation["confidence"] > 0.5
        return validation
    
    def parse_bank_statement(self, text: str) -> Dict[str, Any]:
        """Parse bank statement information"""
        
        data = {
            "transactions": [],
            "balance": 0,
            "account_info": {}
        }
        
        # Extract account balance
        balance_patterns = [
            r'Balance[:\s]+(\d+[,.]?\d*)',
            r'Current Balance[:\s]+(\d+[,.]?\d*)',
            r'(\d+[,.]?\d*)\s+AED'
        ]
        
        for pattern in balance_patterns:
            balance_match = re.search(pattern, text, re.IGNORECASE)
            if balance_match:
                balance_str = balance_match.group(1).replace(',', '')
                try:
                    data["balance"] = float(balance_str)
                    break
                except ValueError:
                    continue
        
        # Extract transactions (simplified)
        transaction_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})\s+([A-Za-z\s]+)\s+(\d+[,.]?\d*)',
            r'(\d{1,2}-\d{1,2}-\d{4})\s+([A-Za-z\s]+)\s+(\d+[,.]?\d*)'
        ]
        
        for pattern in transaction_patterns:
            transactions = re.findall(pattern, text)
            for trans in transactions[:20]:  # Limit to 20 recent transactions
                try:
                    data["transactions"].append({
                        "date": trans[0],
                        "description": trans[1].strip(),
                        "amount": float(trans[2].replace(',', ''))
                    })
                except (ValueError, IndexError):
                    continue
        
        return data
    
    def analyze_income_patterns(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze income patterns from financial data"""
        
        transactions = financial_data.get("transactions", [])
        
        if not transactions:
            return {
                "average_monthly": 0,
                "stability_score": 0,
                "confidence": 0.1
            }
        
        # Identify salary/income transactions
        income_transactions = []
        income_keywords = ['salary', 'wage', 'payment', 'transfer', 'deposit']
        
        for trans in transactions:
            description = trans.get("description", "").lower()
            amount = trans.get("amount", 0)
            
            if amount > 500 and any(keyword in description for keyword in income_keywords):
                income_transactions.append(trans)
        
        if not income_transactions:
            # Use all positive transactions as potential income
            income_transactions = [t for t in transactions if t.get("amount", 0) > 0]
        
        # Calculate monthly average
        total_income = sum(t.get("amount", 0) for t in income_transactions)
        monthly_average = total_income / max(1, len(income_transactions)) if income_transactions else 0
        
        # Calculate stability score
        if len(income_transactions) >= 3:
            amounts = [t.get("amount", 0) for t in income_transactions]
            avg_amount = sum(amounts) / len(amounts)
            variance = sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)
            stability_score = max(0, 1 - (variance / (avg_amount ** 2)) if avg_amount > 0 else 0)
        else:
            stability_score = 0.5  # Neutral for insufficient data
        
        return {
            "average_monthly": monthly_average,
            "stability_score": stability_score,
            "income_transactions_count": len(income_transactions),
            "confidence": min(1.0, len(income_transactions) / 5)  # Higher confidence with more transactions
        }
    
    def parse_resume(self, text: str) -> Dict[str, Any]:
        """Parse resume information"""
        
        data = {
            "skills": [],
            "experience_years": 0,
            "education": [],
            "employment_history": []
        }
        
        # Extract skills
        skills_section = re.search(r'Skills[:\s]+(.*?)(?=\n[A-Z]|\n\n|$)', text, re.IGNORECASE | re.DOTALL)
        if skills_section:
            skills_text = skills_section.group(1)
            # Extract individual skills
            skills = re.findall(r'([A-Za-z\s]{2,20})', skills_text)
            data["skills"] = [skill.strip() for skill in skills[:10]]  # Limit to 10 skills
        
        # Extract experience years
        experience_patterns = [
            r'(\d+)\s+years?\s+(?:of\s+)?experience',
            r'Experience[:\s]+(\d+)\s+years?'
        ]
        
        for pattern in experience_patterns:
            exp_match = re.search(pattern, text, re.IGNORECASE)
            if exp_match:
                try:
                    data["experience_years"] = int(exp_match.group(1))
                    break
                except ValueError:
                    continue
        
        # Extract education
        education_keywords = ['bachelor', 'master', 'phd', 'diploma', 'certificate', 'degree']
        for keyword in education_keywords:
            if keyword in text.lower():
                data["education"].append(keyword.title())
        
        return data
    
    def parse_credit_report(self, text: str) -> Dict[str, Any]:
        """Parse credit report information"""
        
        data = {
            "credit_score": 0,
            "payment_history": "good",
            "total_debt": 0,
            "credit_utilization": 0
        }
        
        # Extract credit score
        score_patterns = [
            r'Credit Score[:\s]+(\d{3})',
            r'Score[:\s]+(\d{3})',
            r'(\d{3})\s+Credit Score'
        ]
        
        for pattern in score_patterns:
            score_match = re.search(pattern, text, re.IGNORECASE)
            if score_match:
                try:
                    data["credit_score"] = int(score_match.group(1))
                    break
                except ValueError:
                    continue
        
        # Extract total debt
        debt_patterns = [
            r'Total Debt[:\s]+(\d+[,.]?\d*)',
            r'Outstanding[:\s]+(\d+[,.]?\d*)',
            r'(\d+[,.]?\d*)\s+AED\s+(?:debt|outstanding)'
        ]
        
        for pattern in debt_patterns:
            debt_match = re.search(pattern, text, re.IGNORECASE)
            if debt_match:
                try:
                    debt_str = debt_match.group(1).replace(',', '')
                    data["total_debt"] = float(debt_str)
                    break
                except ValueError:
                    continue
        
        return data
    
    def parse_assets_liabilities_excel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse assets and liabilities from Excel file"""
        
        data = {
            "total_assets": 0,
            "total_liabilities": 0,
            "net_worth": 0,
            "assets": [],
            "liabilities": []
        }
        
        try:
            # Look for assets and liabilities columns
            assets_cols = [col for col in df.columns if 'asset' in col.lower()]
            liabilities_cols = [col for col in df.columns if 'liabilit' in col.lower() or 'debt' in col.lower()]
            
            # Calculate total assets
            for col in assets_cols:
                if df[col].dtype in ['int64', 'float64']:
                    asset_total = df[col].sum()
                    data["total_assets"] += asset_total
                    data["assets"].append({
                        "type": col,
                        "amount": asset_total
                    })
            
            # Calculate total liabilities
            for col in liabilities_cols:
                if df[col].dtype in ['int64', 'float64']:
                    liability_total = df[col].sum()
                    data["total_liabilities"] += liability_total
                    data["liabilities"].append({
                        "type": col,
                        "amount": liability_total
                    })
            
            # Calculate net worth
            data["net_worth"] = data["total_assets"] - data["total_liabilities"]
            
        except Exception as e:
            # Fallback: try to find any numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data["total_assets"] = df[numeric_cols].sum().sum()
        
        return data

    # Main processing method
    async def process_document(self, file_path: str, document_type: str) -> Dict[str, Any]:
        """Main method to process any document type"""
        
        try:
            if document_type == "emirates_id":
                return await self.process_emirates_id(file_path)
            elif document_type == "bank_statement":
                return await self.process_bank_statement(file_path)
            elif document_type == "resume":
                return await self.process_resume(file_path)
            elif document_type == "credit_report":
                return await self.process_credit_report(file_path)
            elif document_type == "assets_liabilities":
                return await self.process_assets_liabilities(file_path)
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported document type: {document_type}",
                    "data": {},
                    "confidence": 0.0
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": {},
                "confidence": 0.0
            } 