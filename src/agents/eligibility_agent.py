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

# Import ML models
try:
    from ..models.ml_models import SocialSupportMLModels
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False
    print("Warning: ML models not available, falling back to rule-based assessment")


class EligibilityAssessmentAgent(BaseAgent):
    """Agent specialized in assessing eligibility using ML models + rule-based fallback"""
    
    def __init__(self):
        super().__init__("EligibilityAssessmentAgent")
        
        # Initialize ML models if available
        if ML_MODELS_AVAILABLE:
            self.ml_models = SocialSupportMLModels()
            self.use_ml_models = True
            # Try to load pre-trained models
            try:
                self.ml_models.load_models()
            except Exception as e:
                print(f"Could not load pre-trained models: {e}")
                self.use_ml_models = False
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
                assessment_result = await self._perform_ml_eligibility_assessment(
                    application_data, extracted_documents
                )
            else:
                # Fallback to rule-based assessment
                assessment_result = await self._perform_rule_based_eligibility_assessment(
                    application_data, extracted_documents
                )
            
            # Calculate support amount if eligible
            if assessment_result["eligible"]:
                if self.use_ml_models:
                    # Use ML model for support amount prediction
                    ml_support = self.ml_models.predict_support_amount(
                        application_data, extracted_documents
                    )
                    if "error" not in ml_support:
                        assessment_result["support_calculation"] = {
                            "monthly_support_amount": ml_support["estimated_amount"],
                            "support_bracket": ml_support["support_bracket"],
                            "confidence": ml_support["confidence"],
                            "method": "ml_prediction"
                        }
                    else:
                        # Fallback to rule-based calculation
                        support_calculation = await self._calculate_support_amount_rules(
                            application_data, extracted_documents, assessment_result
                        )
                        assessment_result["support_calculation"] = support_calculation
                else:
                    # Rule-based support calculation
                    support_calculation = await self._calculate_support_amount_rules(
                        application_data, extracted_documents, assessment_result
                    )
                    assessment_result["support_calculation"] = support_calculation
            
            # Generate detailed reasoning
            reasoning = await self._generate_assessment_reasoning(
                application_data, extracted_documents, assessment_result
            )
            
            return {
                "agent_name": self.agent_name,
                "application_id": application_id,
                "assessment_result": assessment_result,
                "reasoning": reasoning,
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
    
    async def _perform_ml_eligibility_assessment(
        self, 
        application_data: Dict[str, Any], 
        extracted_documents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform ML-based eligibility assessment using multiple models
        
        Returns:
            Comprehensive assessment results from ML models
        """
        
        # 1. Eligibility Classification
        eligibility_prediction = self.ml_models.predict_eligibility(
            application_data, extracted_documents
        )
        
        # 2. Risk Assessment
        risk_prediction = self.ml_models.predict_risk_level(
            application_data, extracted_documents
        )
        
        # 3. Fraud Detection
        fraud_detection = self.ml_models.detect_fraud(
            application_data, extracted_documents
        )
        
        # 4. Economic Program Matching
        program_matching = self.ml_models.match_economic_programs(
            application_data, extracted_documents
        )
        
        # Combine ML predictions for final eligibility decision
        if "error" in eligibility_prediction:
            # Fallback to rule-based if ML fails
            return await self._perform_rule_based_eligibility_assessment(
                application_data, extracted_documents
            )
        
        # Check for fraud flags
        is_high_fraud_risk = (
            fraud_detection.get("risk_level") == "high" or
            fraud_detection.get("fraud_probability", 0) > 0.8
        )
        
        # Override eligibility if fraud detected
        final_eligible = (
            eligibility_prediction["eligible"] and 
            not is_high_fraud_risk and
            risk_prediction.get("risk_level") != "high"
        )
        
        # Calculate confidence score
        confidence_factors = [
            eligibility_prediction.get("confidence", 0.5),
            1.0 - fraud_detection.get("fraud_probability", 0.5),
            0.8 if risk_prediction.get("risk_level") == "low" else 0.5
        ]
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        
        return {
            "eligible": final_eligible,
            "total_score": eligibility_prediction.get("probability_eligible", 0.5),
            "confidence": overall_confidence,
            "ml_predictions": {
                "eligibility": eligibility_prediction,
                "risk": risk_prediction,
                "fraud": fraud_detection,
                "programs": program_matching
            },
            "assessment_method": "ml_ensemble",
            "feature_importance": eligibility_prediction.get("feature_importance", {}),
            "override_reasons": []
        }
    
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
        has_medical_conditions = application_data.get("has_medical_conditions", False)
        
        # Base calculation
        family_supplement = (family_size - 1) * self.support_calculation["per_dependent"]
        
        # Housing supplement for high rent
        housing_supplement = 0
        monthly_income = application_data.get("monthly_income", 1)
        if monthly_rent / monthly_income > 0.4:  # High rent burden
            housing_supplement = self.support_calculation["housing_adjustment"] * monthly_rent
        
        # Medical supplement
        medical_supplement = 0
        if has_medical_conditions:
            medical_supplement = self.support_calculation["housing_adjustment"] * monthly_rent
        
        # Calculate total
        total_support = base_amount + family_supplement + housing_supplement + medical_supplement
        
        # Cap at maximum
        final_support = min(total_support, self.support_calculation["maximum_support"])
        
        return {
            "monthly_support_amount": round(final_support, 2),
            "calculation_breakdown": {
                "base_amount": base_amount,
                "family_supplement": family_supplement,
                "housing_supplement": housing_supplement,
                "medical_supplement": medical_supplement,
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
        
        context = {
            "application_summary": {
                "monthly_income": application_data.get("monthly_income", 0),
                "family_size": application_data.get("family_size", 1),
                "employment_status": application_data.get("employment_status", "unknown"),
                "housing_type": application_data.get("housing_type", "unknown")
            },
            "assessment_result": {
                "eligible": assessment_result["eligible"],
                "total_score": assessment_result["total_score"],
                "component_scores": assessment_result["component_scores"]
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