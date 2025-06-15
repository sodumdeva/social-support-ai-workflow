"""
Eligibility Assessment Agent for Social Support AI Workflow

Evaluates social support applications using:
- ML-based eligibility classification (Random Forest)
- Risk assessment (Gradient Boosting)
- Support amount prediction (Multi-class Classification)
- Fraud detection (Isolation Forest + SVM)
- Rule-based validation as fallback
"""
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from .base_agent import BaseAgent

# Import logging
from src.utils.logging_config import get_logger
logger = get_logger("eligibility_agent")

# Import ML models
try:
    import sys
    import os
    # Add the src directory to path
    current_dir = os.path.dirname(__file__)  # agents directory
    src_dir = os.path.dirname(current_dir)   # src directory
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    from models.ml_models import SocialSupportMLModels
    ML_MODELS_AVAILABLE = True
except ImportError as e:
    ML_MODELS_AVAILABLE = False
    print(f"Warning: ML models not available ({str(e)}), falling back to rule-based assessment")


class EligibilityAssessmentAgent(BaseAgent):
    """Agent specialized in assessing eligibility using ML models + rule-based fallback"""
    
    def __init__(self):
        super().__init__("EligibilityAssessmentAgent")
        
        # Initialize ML models if available (singleton pattern - models load only once)
        if ML_MODELS_AVAILABLE:
            self.ml_models = SocialSupportMLModels()  # Singleton - only loads once
            self.use_ml_models = True
        else:
            self.ml_models = None
            self.use_ml_models = False
        
        # Define eligibility criteria weights (for fallback)
        self.eligibility_weights = {
            "financial_need": 0.35,      # 35% - Primary factor
            "family_composition": 0.25,   # 25% - Family size, dependents
            "employment_stability": 0.20, # 20% - Employment history, income stability
            "housing_situation": 0.10,    # 10% - Housing costs, type
            "demographics": 0.10          # 10% - Age, medical conditions, etc.
        }
        
        # Eligibility thresholds
        self.thresholds = {
            "minimum_score": 0.6,         # 60% minimum eligibility score
            "maximum_income": 8000,       # AED per month
            "minimum_family_size": 2,     # At least 2 family members for family support
            "maximum_assets": 100000,     # AED total assets
            "credit_score_minimum": 400   # Minimum credit score (if available)
        }
        
        # Support amount calculation parameters
        self.support_calculation = {
            "base_amount": 1000,          # Base monthly support
            "per_dependent": 300,         # Additional per dependent
            "housing_adjustment": 0.3,    # 30% adjustment for housing costs
            "maximum_support": 5000       # Maximum monthly support
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process eligibility assessment using ML models with rule-based fallback
        
        Args:
            input_data: {
                "application_data": Dict with applicant information,
                "extracted_documents": Dict with document extraction results,
                "application_id": str
            }
            
        Returns:
            Complete eligibility assessment with ML predictions
        """
        application_data = input_data.get("application_data", {})
        extracted_documents = input_data.get("extracted_documents", {})
        application_id = input_data.get("application_id", "unknown")
        
        try:
            # Primary assessment using ML models
            if self.use_ml_models:
                assessment_result = await self._run_ml_assessment(
                    application_data, extracted_documents
                )
            else:
                # Fallback to rule-based assessment
                assessment_result = await self._perform_rule_based_eligibility_assessment(
                    application_data, extracted_documents
                )
            
            # Perform data validation and inconsistency detection
            validation_result = await self._perform_data_validation(
                application_data, extracted_documents
            )
            
            # Generate detailed reasoning
            reasoning = await self._generate_assessment_reasoning(
                application_data, extracted_documents, assessment_result
            )
            
            # Generate economic enablement recommendations
            enablement_recommendations = await self._generate_economic_enablement_recommendations(
                application_data, extracted_documents, assessment_result
            )
            
            return {
                "agent_name": self.agent_name,
                "application_id": application_id,
                "assessment_result": assessment_result,
                "validation_result": validation_result,
                "reasoning": reasoning,
                "economic_enablement": enablement_recommendations,
                "assessed_at": datetime.utcnow().isoformat(),
                "assessment_method": "ml_based" if self.use_ml_models else "rule_based",
                "status": "success"
            }
            
        except Exception as e:
            return {
                "agent_name": self.agent_name,
                "application_id": application_id,
                "status": "error",
                "error": str(e),
                "assessed_at": datetime.utcnow().isoformat()
            }
    
    async def _run_ml_assessment(
        self, 
        application_data: Dict[str, Any], 
        extracted_documents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run ML-based assessment using simplified models"""
        
        try:
            # Initialize ML models (singleton)
            ml_models = SocialSupportMLModels()
            
            # Run only the essential predictions
            results = {}
            
            # 1. Eligibility Prediction (Essential)
            logger.info("🤖 Running eligibility prediction...")
            eligibility_result = ml_models.predict_eligibility(application_data, extracted_documents)
            results["eligibility"] = eligibility_result
            
            # 2. Support Amount Prediction (Essential)
            logger.info("💰 Running support amount prediction...")
            support_result = ml_models.predict_support_amount(application_data, extracted_documents)
            results["support_amount"] = support_result
            
            # Create final assessment combining both results
            final_assessment = {
                "eligible": eligibility_result.get("eligible", False),
                "confidence": eligibility_result.get("confidence", 0.5),
                "eligibility_reasoning": eligibility_result.get("reasoning", "ML-based assessment"),
                "support_amount": support_result.get("predicted_amount", 0),
                "support_range": support_result.get("amount_range", "Basic Support"),
                "support_reasoning": support_result.get("reasoning", "Amount calculated based on profile"),
                "assessment_method": "simplified_ml",
                "models_used": ["eligibility_classifier", "support_amount_predictor"]
            }
            
            logger.info(f"✅ ML Assessment completed - Eligible: {final_assessment['eligible']}, Amount: {final_assessment['support_amount']:.0f} AED")
            
            return final_assessment
            
        except Exception as e:
            logger.error(f"Error in ML assessment: {str(e)}")
            # Fallback to rule-based assessment
            logger.info("🔄 Falling back to rule-based assessment...")
            return await self._perform_rule_based_eligibility_assessment(application_data, extracted_documents)
    
    async def _perform_rule_based_eligibility_assessment(
        self, 
        application_data: Dict[str, Any], 
        extracted_documents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback rule-based eligibility assessment (original implementation)"""
        
        # Calculate individual assessment components
        financial_assessment = self._assess_financial_need(application_data, extracted_documents)
        family_assessment = self._assess_family_composition(application_data)
        employment_assessment = self._assess_employment_stability(application_data, extracted_documents)
        housing_assessment = self._assess_housing_situation(application_data)
        demographics_assessment = self._assess_demographics(application_data)
        
        # Calculate weighted eligibility score
        total_score = (
            financial_assessment["score"] * self.eligibility_weights["financial_need"] +
            family_assessment["score"] * self.eligibility_weights["family_composition"] +
            employment_assessment["score"] * self.eligibility_weights["employment_stability"] +
            housing_assessment["score"] * self.eligibility_weights["housing_situation"] +
            demographics_assessment["score"] * self.eligibility_weights["demographics"]
        )
        
        # Determine eligibility
        is_eligible = total_score >= self.thresholds["minimum_score"]
        
        # Check hard constraints
        hard_constraints = self._check_hard_constraints(application_data, extracted_documents)
        
        # Override eligibility if hard constraints failed
        if not hard_constraints["passed"]:
            is_eligible = False
        
        return {
            "eligible": is_eligible,
            "total_score": round(total_score, 3),
            "minimum_required_score": self.thresholds["minimum_score"],
            "component_scores": {
                "financial_need": financial_assessment,
                "family_composition": family_assessment,
                "employment_stability": employment_assessment,
                "housing_situation": housing_assessment,
                "demographics": demographics_assessment
            },
            "hard_constraints": hard_constraints,
            "confidence": min(0.95, max(0.6, total_score)),  # Confidence based on score
            "assessment_method": "rule_based"
        }
    
    def _assess_financial_need(
        self, 
        application_data: Dict[str, Any], 
        extracted_documents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess financial need based on income, expenses, and assets"""
        
        monthly_income = application_data.get("monthly_income", 0)
        family_size = application_data.get("family_size", 1)
        
        # Get financial data from bank statements if available
        bank_data = extracted_documents.get("bank_statement", {}).get("structured_data", {})
        if bank_data and "monthly_income" in bank_data:
            extracted_income = bank_data["monthly_income"].get("amount", 0)
            if extracted_income > 0:
                monthly_income = max(monthly_income, extracted_income)
        
        # Get assets data
        assets_data = extracted_documents.get("assets", {}).get("structured_data", {})
        total_assets = 0
        if assets_data and "assets" in assets_data:
            total_assets = assets_data["assets"].get("total_assets", 0)
        
        # Calculate per capita income
        per_capita_income = monthly_income / max(family_size, 1)
        
        # Financial need scoring (inverse relationship with income)
        if per_capita_income <= 1000:
            income_score = 1.0
        elif per_capita_income <= 2000:
            income_score = 0.8
        elif per_capita_income <= 3000:
            income_score = 0.6
        elif per_capita_income <= 4000:
            income_score = 0.4
        else:
            income_score = 0.2
        
        # Assets penalty
        assets_penalty = min(0.3, total_assets / 300000)  # Reduce score based on high assets
        final_score = max(0, income_score - assets_penalty)
        
        return {
            "score": round(final_score, 3),
            "monthly_income": monthly_income,
            "per_capita_income": per_capita_income,
            "total_assets": total_assets,
            "assessment": "high_need" if final_score >= 0.8 else "moderate_need" if final_score >= 0.6 else "low_need"
        }
    
    def _assess_family_composition(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess family composition factors"""
        
        family_size = application_data.get("family_size", 1)
        dependents = application_data.get("number_of_dependents", 0)
        
        # Higher scores for larger families with more dependents
        if family_size >= 6:
            size_score = 1.0
        elif family_size >= 4:
            size_score = 0.8
        elif family_size >= 2:
            size_score = 0.6
        else:
            size_score = 0.3  # Single person gets lower priority
        
        # Bonus for dependents
        dependent_bonus = min(0.2, dependents * 0.1)
        final_score = min(1.0, size_score + dependent_bonus)
        
        return {
            "score": round(final_score, 3),
            "family_size": family_size,
            "dependents": dependents,
            "assessment": "high_priority" if final_score >= 0.8 else "moderate_priority" if final_score >= 0.6 else "low_priority"
        }
    
    def _assess_employment_stability(
        self, 
        application_data: Dict[str, Any], 
        extracted_documents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess employment stability and history"""
        
        employment_status = application_data.get("employment_status", "unemployed")
        employment_duration = application_data.get("employment_duration_months", 0)
        
        # Get resume data if available
        resume_data = extracted_documents.get("resume", {}).get("structured_data", {})
        if resume_data and "career_summary" in resume_data:
            total_experience = resume_data["career_summary"].get("total_experience_months", 0)
            employment_duration = max(employment_duration, total_experience)
        
        # Employment status scoring
        if employment_status == "unemployed":
            status_score = 1.0  # Higher support need
        elif employment_status == "self_employed":
            status_score = 0.7  # Irregular income
        elif employment_status == "employed":
            status_score = 0.5  # Stable but may need support
        elif employment_status == "student":
            status_score = 0.6  # Student support
        else:
            status_score = 0.4
        
        # Employment duration factor (longer unemployment = higher need)
        if employment_status == "unemployed":
            duration_factor = 1.0  # Maximum for unemployed
        elif employment_duration < 6:
            duration_factor = 0.8  # Recent employment
        elif employment_duration < 24:
            duration_factor = 0.6  # Moderate stability
        else:
            duration_factor = 0.4  # Stable employment
        
        final_score = (status_score + duration_factor) / 2
        
        return {
            "score": round(final_score, 3),
            "employment_status": employment_status,
            "employment_duration_months": employment_duration,
            "assessment": "unstable" if final_score >= 0.7 else "moderate" if final_score >= 0.5 else "stable"
        }
    
    def _assess_housing_situation(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess housing situation and costs"""
        
        housing_type = application_data.get("housing_type", "unknown")
        monthly_rent = application_data.get("monthly_rent", 0)
        monthly_income = application_data.get("monthly_income", 1)
        
        # Housing type scoring
        housing_scores = {
            "rented": 0.8,      # High housing costs
            "shared": 0.9,      # Precarious housing
            "family_house": 0.6, # Some stability
            "owned": 0.3        # Lower housing need
        }
        
        housing_score = housing_scores.get(housing_type, 0.5)
        
        # Rent-to-income ratio
        if monthly_income > 0:
            rent_ratio = monthly_rent / monthly_income
            if rent_ratio > 0.5:  # More than 50% of income on rent
                ratio_penalty = 0.2
            elif rent_ratio > 0.3:  # More than 30% of income on rent
                ratio_penalty = 0.1
            else:
                ratio_penalty = 0
        else:
            ratio_penalty = 0
        
        final_score = min(1.0, housing_score + ratio_penalty)
        
        return {
            "score": round(final_score, 3),
            "housing_type": housing_type,
            "monthly_rent": monthly_rent,
            "rent_to_income_ratio": round(monthly_rent / max(monthly_income, 1), 3),
            "assessment": "high_cost" if final_score >= 0.8 else "moderate_cost" if final_score >= 0.6 else "low_cost"
        }
    
    def _assess_demographics(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess demographic factors"""
        
        has_medical_conditions = application_data.get("has_medical_conditions", False)
        previous_applications = application_data.get("previous_applications", 0)
        
        score = 0.5  # Base score
        
        # Medical conditions increase need
        if has_medical_conditions:
            score += 0.3
        
        # Previous applications might indicate ongoing need but also potential abuse
        if previous_applications == 0:
            score += 0.1  # First-time applicant
        elif previous_applications <= 2:
            score += 0.0  # Normal
        else:
            score -= 0.1  # Frequent applicant - needs review
        
        final_score = max(0, min(1.0, score))
        
        return {
            "score": round(final_score, 3),
            "has_medical_conditions": has_medical_conditions,
            "previous_applications": previous_applications,
            "assessment": "high_need" if final_score >= 0.7 else "standard" if final_score >= 0.5 else "review_required"
        }
    
    def _check_hard_constraints(
        self, 
        application_data: Dict[str, Any], 
        extracted_documents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check hard eligibility constraints"""
        
        violations = []
        
        # Income constraint
        monthly_income = application_data.get("monthly_income", 0)
        if monthly_income > self.thresholds["maximum_income"]:
            violations.append(f"Income {monthly_income} exceeds maximum {self.thresholds['maximum_income']}")
        
        # Assets constraint
        assets_data = extracted_documents.get("assets", {}).get("structured_data", {})
        if assets_data and "assets" in assets_data:
            total_assets = assets_data["assets"].get("total_assets", 0)
            if total_assets > self.thresholds["maximum_assets"]:
                violations.append(f"Assets {total_assets} exceed maximum {self.thresholds['maximum_assets']}")
        
        # Criminal record check (if applicable)
        has_criminal_record = application_data.get("has_criminal_record", False)
        if has_criminal_record:
            violations.append("Criminal record requires manual review")
        
        return {
            "passed": len(violations) == 0,
            "violations": violations
        }
    
    async def _calculate_support_amount_rules(
        self, 
        application_data: Dict[str, Any], 
        extracted_documents: Dict[str, Any],
        assessment_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate appropriate support amount using rule-based logic"""
        
        base_amount = self.support_calculation["base_amount"]
        family_size = application_data.get("family_size", 1)
        monthly_rent = application_data.get("monthly_rent", 0)
        monthly_income = application_data.get("monthly_income", 1)
        has_medical_conditions = application_data.get("has_medical_conditions", False)
        employment_status = application_data.get("employment_status", "unknown")
        
        # Base calculation - ensure minimum meaningful amount
        family_supplement = (family_size - 1) * self.support_calculation["per_dependent"]
        
        # Housing supplement for high rent burden
        housing_supplement = 0
        if monthly_rent > 0 and monthly_rent / monthly_income > 0.4:  # High rent burden
            housing_supplement = min(800, monthly_rent * 0.3)  # Cap housing supplement
        
        # Medical supplement - FIXED: Use proper medical adjustment
        medical_supplement = 0
        if has_medical_conditions:
            medical_supplement = 500  # Fixed medical supplement amount
        
        # Employment status supplement
        employment_supplement = 0
        if employment_status == "unemployed":
            employment_supplement = 600
        elif employment_status == "self_employed":
            employment_supplement = 300
        
        # Income-based supplement (inverse relationship)
        income_per_person = monthly_income / family_size
        if income_per_person < 1000:
            income_supplement = 800
        elif income_per_person < 2000:
            income_supplement = 500
        elif income_per_person < 3000:
            income_supplement = 300
        else:
            income_supplement = 0
        
        # Calculate total
        total_support = (base_amount + family_supplement + housing_supplement + 
                        medical_supplement + employment_supplement + income_supplement)
        
        # Ensure minimum support for eligible applicants
        if assessment_result.get("eligible", False):
            total_support = max(total_support, 800)  # Minimum 800 AED for eligible applicants
        
        # Cap at maximum
        final_support = min(total_support, self.support_calculation["maximum_support"])
        
        return {
            "monthly_support_amount": round(final_support, 2),
            "calculation_breakdown": {
                "base_amount": base_amount,
                "family_supplement": family_supplement,
                "housing_supplement": housing_supplement,
                "medical_supplement": medical_supplement,
                "employment_supplement": employment_supplement,
                "income_supplement": income_supplement,
                "subtotal": total_support,
                "final_amount": round(final_support, 2)
            },
            "support_duration_months": 6,  # Standard 6-month support period
            "review_required_after_months": 3  # Review after 3 months
        }
    
    async def _generate_assessment_reasoning(
        self, 
        application_data: Dict[str, Any], 
        extracted_documents: Dict[str, Any],
        assessment_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed reasoning for the assessment decision"""
        
        system_prompt = """You are a social services specialist AI. Provide clear, empathetic, 
        and detailed reasoning for eligibility decisions. Focus on helping the applicant understand 
        the decision and potential next steps."""
        
        task = "Generate a comprehensive assessment reasoning"
        
        # Safely extract component scores
        component_scores = assessment_result.get("component_scores", {})
        if not component_scores:
            # If no component scores, create a simplified structure
            component_scores = {
                "overall_assessment": {
                    "score": assessment_result.get("total_score", 0),
                    "assessment": "automated_evaluation"
                }
            }
        
        context = {
            "application_summary": {
                "monthly_income": application_data.get("monthly_income", 0),
                "family_size": application_data.get("family_size", 1),
                "employment_status": application_data.get("employment_status", "unknown"),
                "housing_type": application_data.get("housing_status", "unknown")
            },
            "assessment_result": {
                "eligible": assessment_result.get("eligible", False),
                "total_score": assessment_result.get("total_score", 0),
                "component_scores": component_scores
            },
            "support_amount": assessment_result.get("support_calculation", {}).get("monthly_support_amount", 0)
        }
        
        output_format = """{
    "decision": "approved/declined/requires_review",
    "key_factors": [
        "Primary factors that influenced the decision"
    ],
    "strengths": [
        "Positive aspects of the application"
    ],
    "areas_of_concern": [
        "Issues that affected the assessment"
    ],
    "recommendations": [
        "Specific recommendations for the applicant"
    ],
    "next_steps": [
        "What the applicant should do next"
    ],
    "summary": "A clear, empathetic summary of the decision"
}"""
        
        prompt = self.create_structured_prompt(task, context, output_format)
        
        llm_response = await self.invoke_llm(prompt, system_prompt)
        
        if llm_response["status"] == "success":
            reasoning = self.extract_json_from_response(llm_response["response"])
            if reasoning:
                return reasoning
        
        # Fallback reasoning
        return self._generate_fallback_reasoning(assessment_result)
    
    def _generate_fallback_reasoning(self, assessment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic reasoning if LLM fails"""
        
        if assessment_result["eligible"]:
            decision = "approved"
            summary = f"Application approved with eligibility score of {assessment_result['total_score']:.2f}."
        else:
            decision = "declined"
            summary = f"Application declined. Eligibility score {assessment_result['total_score']:.2f} below required {assessment_result['minimum_required_score']}."
        
        return {
            "decision": decision,
            "key_factors": ["Automated assessment based on eligibility criteria"],
            "summary": summary,
            "generated_by": "fallback_system"
        }

    async def _generate_economic_enablement_recommendations(
        self, 
        application_data: Dict[str, Any], 
        extracted_documents: Dict[str, Any],
        assessment_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive economic enablement recommendations based on assessment"""
        
        recommendations = []
        
        # Job Training and Upskilling Recommendations
        training_programs = [
           {
               "name": "Digital Skills Training Program",
               "duration": "3 months",
               "description": "Learn essential computer skills, Microsoft Office, and basic digital literacy",
               "provider": "UAE Digital Skills Academy",
               "cost": "Free for eligible applicants",
               "contact": "800-SKILLS"
           },
           {
               "name": "Vocational Training Certificate",
               "duration": "6 months", 
               "description": "Hands-on training in trades like plumbing, electrical work, or automotive repair",
               "provider": "Technical Education Institute",
               "cost": "Subsidized for low-income families",
               "contact": "04-123-4567"
           },
           {
               "name": "English Language Course",
               "duration": "4 months",
               "description": "Improve English communication skills for better job opportunities",
               "provider": "Community Learning Center",
               "cost": "Free",
               "contact": "02-987-6543"
           }
        ]
        
        # Job Matching Opportunities
        job_opportunities = [
           {
               "title": "Customer Service Representative",
               "company": "Various Companies",
               "salary_range": "3000-4500 AED",
               "requirements": "Basic English, computer skills",
               "contact": "UAE Employment Center - 800-JOBS"
           },
           {
               "title": "Retail Sales Associate", 
               "company": "Shopping Centers",
               "salary_range": "2500-3500 AED",
               "requirements": "Customer service skills, flexible schedule",
               "contact": "Retail Jobs Portal - jobs.uae.gov"
           },
           {
               "title": "Food Service Worker",
               "company": "Restaurants & Hotels",
               "salary_range": "2200-3200 AED",
               "requirements": "Food safety certificate (provided)",
               "contact": "Hospitality Jobs Center - 04-555-0123"
           }
        ]
        
        # Career Counseling Services
        counseling_services = [
           {
               "service": "Career Assessment & Planning",
               "provider": "UAE Career Development Center",
               "description": "Professional assessment to identify strengths and career paths",
               "cost": "Free consultation",
               "contact": "800-CAREER"
           },
           {
               "service": "Resume Writing Workshop",
               "provider": "Employment Support Services",
               "description": "Learn to create professional resumes and cover letters",
               "cost": "Free",
               "contact": "career.support@uae.gov"
           },
           {
               "service": "Interview Skills Training",
               "provider": "Professional Development Institute",
               "description": "Practice interview techniques and build confidence",
               "cost": "Free for social support recipients",
               "contact": "04-321-9876"
           }
        ]
        
        # Financial Literacy Programs
        financial_programs = [
           {
               "program": "Personal Finance Management",
               "provider": "UAE Financial Literacy Center",
               "description": "Learn budgeting, saving, and financial planning skills",
               "duration": "2 months",
               "cost": "Free",
               "contact": "finance.education@uae.gov"
           },
           {
               "program": "Small Business Development",
               "provider": "Entrepreneurship Support Center",
               "description": "Training for starting and managing small businesses",
               "duration": "3 months",
               "cost": "Subsidized",
               "contact": "800-BUSINESS"
           }
        ]
        
        # Generate personalized recommendations based on assessment
        if assessment_result["eligible"]:
            recommendations.extend([
               "✅ **Immediate Support**: You qualify for monthly financial assistance. Use this time to focus on skill development.",
               "📚 **Skill Development**: Enroll in digital skills or vocational training to improve job prospects.",
               "💼 **Job Search**: Register with UAE Employment Center for job matching services.",
               "💰 **Financial Planning**: Attend financial literacy workshops to maximize your support benefits.",
               "🎯 **Career Counseling**: Get professional guidance to identify the best career path for your situation."
            ])
            
            summary = f"""🎉 **Good news!** You qualify for social support. Here's your path to economic independence:

**Immediate Next Steps:**
1. **Financial Support**: You'll receive monthly assistance to cover basic needs
2. **Skill Building**: Use this stability to invest in your future through training programs
3. **Job Preparation**: Work with career counselors to prepare for employment
4. **Long-term Planning**: Develop financial literacy skills for sustainable independence

**Available Programs:**
- **Training**: {len(training_programs)} programs available
- **Job Opportunities**: {len(job_opportunities)} types of positions
- **Support Services**: {len(counseling_services)} counseling services
- **Financial Education**: {len(financial_programs)} programs

Remember: This support is designed to help you become self-sufficient. Take advantage of all available resources!"""
        
        else:
            recommendations.extend([
               "📚 **Skill Enhancement**: Focus on developing marketable skills through free training programs.",
               "💼 **Job Search Assistance**: Use employment services to find better-paying opportunities.", 
               "🎯 **Career Counseling**: Get professional guidance to improve your employment prospects.",
               "💰 **Financial Planning**: Learn budgeting and financial management skills.",
               "🔄 **Reapply Later**: Consider reapplying after improving your employment situation."
            ])
            
            summary = f"""While you don't qualify for direct financial support at this time, there are many resources to help improve your situation:

**Focus Areas:**
1. **Skill Development**: Enhance your qualifications through free training programs
2. **Job Search**: Access employment services for better opportunities
3. **Financial Management**: Learn to optimize your current income
4. **Future Planning**: Work toward qualifying for support in the future

**Available Resources:**
- **Free Training**: {len(training_programs)} programs to boost your skills
- **Job Assistance**: {len(job_opportunities)} types of opportunities available
- **Career Support**: {len(counseling_services)} professional services
- **Financial Education**: {len(financial_programs)} programs for money management

Don't give up! These resources can help you build a stronger financial foundation."""
        
        return {
            "recommendations": recommendations,
            "training_programs": training_programs,
            "job_opportunities": job_opportunities, 
            "counseling_services": counseling_services,
            "financial_programs": financial_programs,
            "summary": summary,
            "generated_by": "comprehensive_system"
        }
    
    async def _perform_data_validation(
        self, 
        application_data: Dict[str, Any], 
        extracted_documents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data validation and inconsistency detection
        Addresses the problem statement requirements for:
        - Manual Data Gathering errors
        - Semi-Automated Data Validations
        - Inconsistent Information detection
        """
        
        validation_issues = []
        inconsistencies = []
        confidence_score = 1.0
        
        # 1. Basic Field Validation
        required_fields = ["name", "emirates_id", "monthly_income", "family_size"]
        for field in required_fields:
            if not application_data.get(field):
                validation_issues.append(f"Missing required field: {field}")
                confidence_score -= 0.1
        
        # 2. Emirates ID Format Validation
        emirates_id = application_data.get("emirates_id", "")
        if emirates_id and not self._validate_emirates_id_format(emirates_id):
            validation_issues.append("Invalid Emirates ID format")
            confidence_score -= 0.15
        
        # 3. Income Validation and Cross-checking
        stated_income = application_data.get("monthly_income", 0)
        bank_statement_data = extracted_documents.get("bank_statement", {})
        
        if bank_statement_data and "monthly_income" in bank_statement_data:
            bank_income = bank_statement_data["monthly_income"]
            income_difference = abs(stated_income - bank_income) / max(stated_income, bank_income, 1)
            
            if income_difference > 0.2:  # More than 20% difference
                inconsistencies.append({
                    "type": "income_mismatch",
                    "description": f"Stated income ({stated_income:,.0f} AED) differs significantly from bank statement ({bank_income:,.0f} AED)",
                    "severity": "high" if income_difference > 0.5 else "medium",
                    "difference_percentage": income_difference * 100
                })
                confidence_score -= 0.2 if income_difference > 0.5 else 0.1
        
        # 4. Employment Status Cross-validation
        stated_employment = application_data.get("employment_status", "")
        resume_data = extracted_documents.get("resume", {})
        
        if resume_data and "current_employment_status" in resume_data:
            resume_employment = resume_data["current_employment_status"]
            if stated_employment != resume_employment:
                inconsistencies.append({
                    "type": "employment_mismatch",
                    "description": f"Stated employment status ({stated_employment}) differs from resume ({resume_employment})",
                    "severity": "medium"
                })
                confidence_score -= 0.1
        
        # 5. Address Consistency Check
        stated_address = application_data.get("address", "")
        emirates_id_data = extracted_documents.get("emirates_id", {})
        credit_report_data = extracted_documents.get("credit_report", {})
        
        addresses_to_check = []
        if emirates_id_data.get("address"):
            addresses_to_check.append(("Emirates ID", emirates_id_data["address"]))
        if credit_report_data.get("address"):
            addresses_to_check.append(("Credit Report", credit_report_data["address"]))
        
        for source, address in addresses_to_check:
            if stated_address and address and not self._addresses_match(stated_address, address):
                inconsistencies.append({
                    "type": "address_mismatch",
                    "description": f"Address from {source} doesn't match stated address",
                    "severity": "medium"
                })
                confidence_score -= 0.1
        
        # 6. Family Size Validation
        stated_family_size = application_data.get("family_size", 0)
        if stated_family_size < 1 or stated_family_size > 15:
            validation_issues.append(f"Unusual family size: {stated_family_size}")
            confidence_score -= 0.1
        
        # 7. Income vs Family Size Reasonableness Check
        if stated_income > 0 and stated_family_size > 0:
            per_person_income = stated_income / stated_family_size
            if per_person_income > 10000:  # Very high per-person income
                validation_issues.append("Income appears unusually high for family size")
                confidence_score -= 0.05
            elif per_person_income < 100:  # Very low per-person income
                validation_issues.append("Income appears unusually low - may need verification")
                confidence_score -= 0.05
        
        # 8. Document Quality Assessment
        document_quality_issues = []
        for doc_type, doc_data in extracted_documents.items():
            if doc_data.get("extraction_confidence", 1.0) < 0.7:
                document_quality_issues.append(f"Low confidence in {doc_type} extraction")
                confidence_score -= 0.05
        
        # 9. Fraud Risk Indicators
        fraud_indicators = []
        
        # Check for duplicate applications (simplified)
        if self._check_potential_duplicate(application_data):
            fraud_indicators.append("Potential duplicate application detected")
            confidence_score -= 0.3
        
        # Check for suspicious patterns
        if self._check_suspicious_patterns(application_data, extracted_documents):
            fraud_indicators.append("Suspicious data patterns detected")
            confidence_score -= 0.2
        
        # Calculate overall validation status
        total_issues = len(validation_issues) + len(inconsistencies) + len(fraud_indicators)
        
        if total_issues == 0:
            validation_status = "passed"
        elif total_issues <= 2 and not fraud_indicators:
            validation_status = "passed_with_warnings"
        elif total_issues <= 4 and confidence_score > 0.6:
            validation_status = "requires_review"
        else:
            validation_status = "failed"
        
        return {
            "validation_status": validation_status,
            "confidence_score": max(0.0, confidence_score),
            "validation_issues": validation_issues,
            "inconsistencies": inconsistencies,
            "document_quality_issues": document_quality_issues,
            "fraud_indicators": fraud_indicators,
            "total_issues": total_issues,
            "recommendations": self._generate_validation_recommendations(
                validation_status, validation_issues, inconsistencies, fraud_indicators
            )
        }
    
    def _validate_emirates_id_format(self, emirates_id: str) -> bool:
        """Validate Emirates ID format (XXX-XXXX-XXXXXXX-X)"""
        import re
        pattern = r'^\d{3}-\d{4}-\d{7}-\d{1}$'
        return bool(re.match(pattern, emirates_id))
    
    def _addresses_match(self, addr1: str, addr2: str) -> bool:
        """Check if two addresses are similar (simplified matching)"""
        if not addr1 or not addr2:
            return False
        
        # Normalize addresses for comparison
        addr1_norm = addr1.lower().replace(",", "").replace(".", "").strip()
        addr2_norm = addr2.lower().replace(",", "").replace(".", "").strip()
        
        # Simple similarity check - in production, use more sophisticated matching
        common_words = set(addr1_norm.split()) & set(addr2_norm.split())
        total_words = set(addr1_norm.split()) | set(addr2_norm.split())
        
        if len(total_words) == 0:
            return False
        
        similarity = len(common_words) / len(total_words)
        return similarity > 0.6  # 60% similarity threshold
    
    def _check_potential_duplicate(self, application_data: Dict[str, Any]) -> bool:
        """Check for potential duplicate applications (simplified)"""
        # In production, this would check against a database of previous applications
        # For now, return False (no duplicates detected)
        return False
    
    def _check_suspicious_patterns(self, application_data: Dict[str, Any], extracted_documents: Dict[str, Any]) -> bool:
        """Check for suspicious data patterns that might indicate fraud"""
        
        suspicious_indicators = 0
        
        # Check for round numbers (might indicate fabricated data)
        income = application_data.get("monthly_income", 0)
        if income > 0 and income % 1000 == 0 and income > 2000:
            suspicious_indicators += 1
        
        # Check for inconsistent document timestamps (if available)
        doc_dates = []
        for doc_data in extracted_documents.values():
            if "document_date" in doc_data:
                doc_dates.append(doc_data["document_date"])
        
        # If documents are from very different time periods, it might be suspicious
        if len(doc_dates) > 1:
            # Simplified check - in production, use proper date parsing
            pass
        
        # Check for unusual combinations
        employment_status = application_data.get("employment_status", "")
        if employment_status == "unemployed" and income > 5000:
            suspicious_indicators += 1
        
        return suspicious_indicators >= 2
    
    def _generate_validation_recommendations(
        self, 
        validation_status: str, 
        validation_issues: List[str], 
        inconsistencies: List[Dict], 
        fraud_indicators: List[str]
    ) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        if validation_status == "failed":
            recommendations.append("Application requires manual review before processing")
            recommendations.append("Verify all submitted documents and information")
        
        elif validation_status == "requires_review":
            recommendations.append("Recommend manual verification of flagged inconsistencies")
            recommendations.append("Consider requesting additional documentation")
        
        elif validation_status == "passed_with_warnings":
            recommendations.append("Minor issues detected - proceed with standard processing")
            recommendations.append("Monitor for any additional red flags")
        
        else:  # passed
            recommendations.append("All validations passed - proceed with automated processing")
        
        # Specific recommendations based on issues
        if any("income" in issue.lower() for issue in validation_issues):
            recommendations.append("Request additional income verification documents")
        
        if inconsistencies:
            recommendations.append("Clarify inconsistencies with applicant before final decision")
        
        if fraud_indicators:
            recommendations.append("Escalate to fraud investigation team")
            recommendations.append("Do not process until fraud concerns are resolved")
        
        return recommendations 