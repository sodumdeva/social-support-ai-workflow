"""
Master Orchestrator Agent for Social Support AI Workflow

Coordinates the entire application processing workflow:
1. Document processing and data extraction
2. Data validation and consistency checks
3. Eligibility assessment
4. Decision recommendation
5. Economic enablement suggestions
6. Final report generation

Uses ReAct reasoning framework for decision making.
"""
from typing import Dict, Any, List, Optional
import asyncio
import json
from datetime import datetime
from enum import Enum

from .base_agent import BaseAgent
from .data_extraction_agent import DataExtractionAgent
from .eligibility_agent import EligibilityAssessmentAgent


class WorkflowStatus(Enum):
    INITIATED = "initiated"
    PROCESSING_DOCUMENTS = "processing_documents"
    VALIDATING_DATA = "validating_data"
    ASSESSING_ELIGIBILITY = "assessing_eligibility"
    GENERATING_RECOMMENDATIONS = "generating_recommendations"
    COMPLETED = "completed"
    FAILED = "failed"


class MasterOrchestrator(BaseAgent):
    """Master orchestrator agent that manages the entire workflow"""
    
    def __init__(self):
        super().__init__("MasterOrchestrator")
        
        # Initialize sub-agents
        self.data_extraction_agent = DataExtractionAgent()
        self.eligibility_agent = EligibilityAssessmentAgent()
        
        # Workflow state tracking
        self.workflow_history = []
        
        # Economic enablement programs
        self.enablement_programs = {
            "upskilling": {
                "programs": ["Digital Skills", "English Language", "Professional Certification"],
                "duration": "3-6 months",
                "eligibility": "unemployed or underemployed"
            },
            "job_matching": {
                "services": ["Career Counseling", "CV Preparation", "Interview Training"],
                "success_rate": "70%",
                "eligibility": "all applicants"
            },
            "entrepreneurship": {
                "programs": ["Business Planning", "Micro-financing", "Mentorship"],
                "funding": "up to 50,000 AED",
                "eligibility": "business plan required"
            },
            "education_support": {
                "programs": ["Child Education Support", "Adult Literacy", "Vocational Training"],
                "coverage": "tuition and materials",
                "eligibility": "families with children or adult learners"
            }
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate the complete social support application processing workflow
        
        Args:
            input_data: {
                "application_data": Dict with basic applicant information,
                "documents": List of document file paths and types,
                "application_id": str,
                "workflow_config": Optional workflow configuration
            }
            
        Returns:
            Complete workflow results with final decision and recommendations
        """
        application_id = input_data.get("application_id", f"APP-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        workflow_config = input_data.get("workflow_config", {})
        
        # Initialize workflow tracking
        workflow_state = {
            "application_id": application_id,
            "status": WorkflowStatus.INITIATED,
            "started_at": datetime.utcnow().isoformat(),
            "steps_completed": [],
            "current_step": None,
            "errors": [],
            "results": {}
        }
        
        try:
            # Step 1: Process Documents and Extract Data
            workflow_state["status"] = WorkflowStatus.PROCESSING_DOCUMENTS
            workflow_state["current_step"] = "document_processing"
            
            extraction_result = await self._step_extract_data(input_data, workflow_state)
            workflow_state["results"]["extraction"] = extraction_result
            workflow_state["steps_completed"].append("document_processing")
            
            if extraction_result["status"] == "error":
                raise Exception(f"Document processing failed: {extraction_result.get('error')}")
            
            # Step 2: Validate Data Consistency
            workflow_state["status"] = WorkflowStatus.VALIDATING_DATA
            workflow_state["current_step"] = "data_validation"
            
            validation_result = await self._step_validate_data(
                input_data["application_data"], 
                extraction_result["extraction_results"],
                workflow_state
            )
            workflow_state["results"]["validation"] = validation_result
            workflow_state["steps_completed"].append("data_validation")
            
            # Step 3: Assess Eligibility
            workflow_state["status"] = WorkflowStatus.ASSESSING_ELIGIBILITY
            workflow_state["current_step"] = "eligibility_assessment"
            
            eligibility_result = await self._step_assess_eligibility(
                input_data["application_data"],
                extraction_result["extraction_results"],
                workflow_state
            )
            workflow_state["results"]["eligibility"] = eligibility_result
            workflow_state["steps_completed"].append("eligibility_assessment")
            
            # Step 4: Generate Economic Enablement Recommendations
            workflow_state["status"] = WorkflowStatus.GENERATING_RECOMMENDATIONS
            workflow_state["current_step"] = "recommendations"
            
            recommendations_result = await self._step_generate_recommendations(
                input_data["application_data"],
                extraction_result["extraction_results"],
                eligibility_result,
                workflow_state
            )
            workflow_state["results"]["recommendations"] = recommendations_result
            workflow_state["steps_completed"].append("recommendations")
            
            # Step 5: Final Decision and Report
            workflow_state["status"] = WorkflowStatus.COMPLETED
            workflow_state["current_step"] = "final_report"
            
            final_report = await self._step_generate_final_report(workflow_state)
            workflow_state["results"]["final_report"] = final_report
            workflow_state["steps_completed"].append("final_report")
            
            workflow_state["completed_at"] = datetime.utcnow().isoformat()
            
            return {
                "agent_name": self.agent_name,
                "workflow_state": workflow_state,
                "final_decision": self._extract_final_decision(workflow_state),
                "processing_summary": self._generate_processing_summary(workflow_state),
                "status": "success"
            }
            
        except Exception as e:
            workflow_state["status"] = WorkflowStatus.FAILED
            workflow_state["errors"].append({
                "step": workflow_state["current_step"],
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            workflow_state["failed_at"] = datetime.utcnow().isoformat()
            
            return {
                "agent_name": self.agent_name,
                "workflow_state": workflow_state,
                "status": "error",
                "error": str(e)
            }
    
    async def _step_extract_data(
        self, 
        input_data: Dict[str, Any], 
        workflow_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Step 1: Extract data from documents using specialized agents"""
        
        self._log_workflow_step(workflow_state, "Starting document processing and data extraction")
        
        # Prepare documents for extraction
        documents = input_data.get("documents", [])
        
        if not documents:
            # If no documents provided, create synthetic data for demo
            self._log_workflow_step(workflow_state, "No documents provided, using application data only")
            return {
                "status": "success", 
                "extraction_results": {},
                "message": "No documents to process"
            }
        
        # Process documents using data extraction agent
        extraction_input = {
            "documents": documents,
            "application_id": workflow_state["application_id"]
        }
        
        extraction_result = await self.data_extraction_agent.process(extraction_input)
        
        self._log_workflow_step(
            workflow_state, 
            f"Processed {len(documents)} documents, {extraction_result.get('successful_extractions', 0)} successful"
        )
        
        return extraction_result
    
    async def _step_validate_data(
        self, 
        application_data: Dict[str, Any], 
        extraction_results: Dict[str, Any],
        workflow_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Step 2: Validate data consistency using ReAct reasoning"""
        
        self._log_workflow_step(workflow_state, "Starting data validation and consistency checks")
        
        # Use ReAct framework for validation reasoning
        validation_prompt = self._create_react_validation_prompt(application_data, extraction_results)
        
        system_prompt = """You are a data validation specialist using ReAct reasoning framework. 
        Follow the pattern: Thought -> Action -> Observation -> Thought -> Action -> Observation -> Final Answer.
        Validate data consistency across all sources and identify discrepancies."""
        
        llm_response = await self.invoke_llm(validation_prompt, system_prompt)
        
        if llm_response["status"] == "success":
            validation_analysis = self.extract_json_from_response(llm_response["response"])
            if validation_analysis:
                self._log_workflow_step(workflow_state, "Data validation completed using LLM analysis")
                return {
                    "status": "success",
                    "validation_result": validation_analysis,
                    "method": "llm_react_reasoning"
                }
        
        # Fallback to rule-based validation
        fallback_validation = self._perform_rule_based_validation(application_data, extraction_results)
        self._log_workflow_step(workflow_state, "Using fallback rule-based validation")
        
        return {
            "status": "success",
            "validation_result": fallback_validation,
            "method": "rule_based_fallback"
        }
    
    async def _step_assess_eligibility(
        self, 
        application_data: Dict[str, Any], 
        extraction_results: Dict[str, Any],
        workflow_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Step 3: Assess eligibility using eligibility agent"""
        
        self._log_workflow_step(workflow_state, "Starting eligibility assessment")
        
        eligibility_input = {
            "application_data": application_data,
            "extracted_documents": extraction_results,
            "application_id": workflow_state["application_id"]
        }
        
        eligibility_result = await self.eligibility_agent.process(eligibility_input)
        
        if eligibility_result["status"] == "success":
            eligible = eligibility_result["assessment_result"]["eligible"]
            score = eligibility_result["assessment_result"]["total_score"]
            self._log_workflow_step(
                workflow_state, 
                f"Eligibility assessment completed: {'ELIGIBLE' if eligible else 'NOT ELIGIBLE'} (Score: {score:.2f})"
            )
        else:
            self._log_workflow_step(workflow_state, "Eligibility assessment failed")
        
        return eligibility_result
    
    async def _step_generate_recommendations(
        self, 
        application_data: Dict[str, Any], 
        extraction_results: Dict[str, Any],
        eligibility_result: Dict[str, Any],
        workflow_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Step 4: Generate economic enablement recommendations"""
        
        self._log_workflow_step(workflow_state, "Generating economic enablement recommendations")
        
        # Extract relevant information for recommendations
        employment_status = application_data.get("employment_status", "unknown")
        education_level = application_data.get("education_level", "unknown")
        family_size = application_data.get("family_size", 1)
        monthly_income = application_data.get("monthly_income", 0)
        
        # Get career information from resume if available
        resume_data = extraction_results.get("resume", {}).get("structured_data", {})
        
        # Generate personalized recommendations using LLM
        recommendations_prompt = self._create_recommendations_prompt(
            application_data, extraction_results, eligibility_result, resume_data
        )
        
        system_prompt = """You are an economic enablement specialist. Generate personalized 
        recommendations for social support applicants to improve their economic situation 
        through upskilling, job matching, entrepreneurship, and education support."""
        
        llm_response = await self.invoke_llm(recommendations_prompt, system_prompt)
        
        if llm_response["status"] == "success":
            recommendations = self.extract_json_from_response(llm_response["response"])
            if recommendations:
                self._log_workflow_step(workflow_state, "Generated personalized recommendations using LLM")
                return {
                    "status": "success",
                    "recommendations": recommendations,
                    "method": "llm_generated"
                }
        
        # Fallback to template-based recommendations
        fallback_recommendations = self._generate_template_recommendations(
            employment_status, education_level, family_size, monthly_income
        )
        
        self._log_workflow_step(workflow_state, "Generated template-based recommendations")
        
        return {
            "status": "success",
            "recommendations": fallback_recommendations,
            "method": "template_based"
        }
    
    async def _step_generate_final_report(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Generate comprehensive final report"""
        
        self._log_workflow_step(workflow_state, "Generating final comprehensive report")
        
        # Extract key information from all workflow steps
        results = workflow_state["results"]
        
        final_report = {
            "executive_summary": self._create_executive_summary(workflow_state),
            "application_details": self._extract_application_summary(workflow_state),
            "processing_results": {
                "documents_processed": len(results.get("extraction", {}).get("extraction_results", {})),
                "data_validation_status": results.get("validation", {}).get("status", "unknown"),
                "eligibility_decision": self._extract_eligibility_decision(results),
                "support_amount": self._extract_support_amount(results)
            },
            "recommendations": results.get("recommendations", {}).get("recommendations", {}),
            "next_steps": self._generate_next_steps(workflow_state),
            "workflow_metrics": self._calculate_workflow_metrics(workflow_state)
        }
        
        return {
            "status": "success",
            "final_report": final_report
        }
    
    def _create_react_validation_prompt(
        self, 
        application_data: Dict[str, Any], 
        extraction_results: Dict[str, Any]
    ) -> str:
        """Create ReAct framework prompt for data validation"""
        
        context = {
            "application_income": application_data.get("monthly_income", 0),
            "application_employment": application_data.get("employment_status", "unknown"),
            "application_family_size": application_data.get("family_size", 1),
            "extracted_data_summary": {k: v.get("status", "unknown") for k, v in extraction_results.items()}
        }
        
        prompt = f"""
Use ReAct reasoning to validate data consistency across application and extracted documents.

APPLICATION DATA:
{json.dumps(context, indent=2)}

EXTRACTED DOCUMENTS:
{json.dumps({k: v.get("structured_data", {}) for k, v in extraction_results.items()}, indent=2)[:1000]}

Follow this ReAct pattern:

Thought: What inconsistencies should I look for?
Action: Compare income values across sources
Observation: [Your finding]
Thought: What about employment status consistency?
Action: Check employment information alignment
Observation: [Your finding]
Thought: Are there any red flags or missing data?
Action: Identify data quality issues
Observation: [Your finding]

Final Answer: {{
    "consistency_score": float (0-1),
    "major_discrepancies": ["list of major issues"],
    "minor_discrepancies": ["list of minor issues"],
    "missing_data": ["list of missing critical data"],
    "data_quality": "high/medium/low",
    "validation_passed": boolean,
    "recommendations": ["recommendations for data collection"]
}}
"""
        return prompt
    
    def _create_recommendations_prompt(
        self,
        application_data: Dict[str, Any],
        extraction_results: Dict[str, Any],
        eligibility_result: Dict[str, Any],
        resume_data: Dict[str, Any]
    ) -> str:
        """Create prompt for generating economic enablement recommendations"""
        
        context = {
            "applicant_profile": {
                "employment_status": application_data.get("employment_status", "unknown"),
                "education_level": application_data.get("education_level", "unknown"),
                "monthly_income": application_data.get("monthly_income", 0),
                "family_size": application_data.get("family_size", 1),
                "has_experience": bool(resume_data.get("work_experience", []))
            },
            "eligibility_status": eligibility_result.get("assessment_result", {}).get("eligible", False),
            "available_programs": self.enablement_programs
        }
        
        prompt = f"""
Generate personalized economic enablement recommendations for this social support applicant.

APPLICANT CONTEXT:
{json.dumps(context, indent=2)}

Generate recommendations in this format:
{{
    "priority_recommendations": [
        {{
            "program_type": "upskilling/job_matching/entrepreneurship/education_support",
            "specific_program": "program name",
            "rationale": "why this program fits the applicant",
            "expected_outcome": "what improvement is expected",
            "timeline": "estimated duration",
            "priority": "high/medium/low"
        }}
    ],
    "immediate_actions": [
        "actionable steps the applicant can take immediately"
    ],
    "long_term_goals": [
        "strategic goals for economic improvement"
    ],
    "success_factors": [
        "key factors that will determine success"
    ],
    "support_needed": [
        "additional support or resources needed"
    ]
}}
"""
        return prompt
    
    def _perform_rule_based_validation(
        self, 
        application_data: Dict[str, Any], 
        extraction_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback rule-based validation"""
        
        discrepancies = []
        
        # Income consistency check
        app_income = application_data.get("monthly_income", 0)
        bank_data = extraction_results.get("bank_statement", {}).get("structured_data", {})
        
        if bank_data and "monthly_income" in bank_data:
            extracted_income = bank_data["monthly_income"].get("amount", 0)
            if abs(app_income - extracted_income) / max(app_income, 1) > 0.2:  # 20% difference
                discrepancies.append("Income mismatch between application and bank statement")
        
        # Employment status check
        app_employment = application_data.get("employment_status", "")
        resume_data = extraction_results.get("resume", {}).get("structured_data", {})
        
        if resume_data and "career_summary" in resume_data:
            resume_employment = resume_data["career_summary"].get("current_employment_status", "")
            if app_employment != resume_employment and resume_employment:
                discrepancies.append("Employment status mismatch between application and resume")
        
        return {
            "consistency_score": max(0, 1 - len(discrepancies) * 0.2),
            "major_discrepancies": discrepancies,
            "minor_discrepancies": [],
            "missing_data": [],
            "data_quality": "medium" if len(discrepancies) <= 1 else "low",
            "validation_passed": len(discrepancies) <= 2
        }
    
    def _generate_template_recommendations(
        self, 
        employment_status: str, 
        education_level: str, 
        family_size: int, 
        monthly_income: int
    ) -> Dict[str, Any]:
        """Generate template-based recommendations"""
        
        recommendations = []
        
        # Unemployment recommendations
        if employment_status == "unemployed":
            recommendations.append({
                "program_type": "job_matching",
                "specific_program": "Career Counseling and Job Placement",
                "rationale": "Immediate need for employment assistance",
                "priority": "high"
            })
            
            if education_level in ["primary", "secondary", "no_education"]:
                recommendations.append({
                    "program_type": "upskilling",
                    "specific_program": "Digital Skills Training",
                    "rationale": "Improve employability with digital skills",
                    "priority": "high"
                })
        
        # Low income recommendations
        if monthly_income < 3000:
            recommendations.append({
                "program_type": "upskilling",
                "specific_program": "Professional Certification Program",
                "rationale": "Increase earning potential through skill development",
                "priority": "medium"
            })
        
        # Family support recommendations
        if family_size >= 3:
            recommendations.append({
                "program_type": "education_support",
                "specific_program": "Child Education Support",
                "rationale": "Support children's education for long-term family welfare",
                "priority": "medium"
            })
        
        return {
            "priority_recommendations": recommendations,
            "immediate_actions": ["Apply for recommended programs", "Update skills assessment"],
            "long_term_goals": ["Achieve financial independence", "Improve family welfare"],
            "success_factors": ["Program completion", "Regular progress monitoring"],
            "support_needed": ["Transportation support", "Childcare during training"]
        }
    
    def _log_workflow_step(self, workflow_state: Dict[str, Any], message: str):
        """Log workflow step for tracking"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": workflow_state.get("current_step", "unknown"),
            "message": message
        }
        
        if "workflow_log" not in workflow_state:
            workflow_state["workflow_log"] = []
        
        workflow_state["workflow_log"].append(log_entry)
    
    def _extract_final_decision(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract final decision from workflow results"""
        
        eligibility_results = workflow_state["results"].get("eligibility", {})
        
        if eligibility_results.get("status") == "success":
            assessment = eligibility_results.get("assessment_result", {})
            return {
                "decision": "approved" if assessment.get("eligible", False) else "declined",
                "eligibility_score": assessment.get("total_score", 0),
                "support_amount": assessment.get("support_calculation", {}).get("monthly_support_amount", 0),
                "reasoning": eligibility_results.get("reasoning", {})
            }
        
        return {
            "decision": "error",
            "error": "Could not complete eligibility assessment"
        }
    
    def _generate_processing_summary(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate processing summary"""
        
        return {
            "total_steps": len(workflow_state["steps_completed"]),
            "processing_time": self._calculate_processing_time(workflow_state),
            "status": workflow_state["status"].value,
            "errors_encountered": len(workflow_state.get("errors", [])),
            "workflow_efficiency": len(workflow_state["steps_completed"]) / 5  # 5 total steps
        }
    
    def _calculate_processing_time(self, workflow_state: Dict[str, Any]) -> float:
        """Calculate total processing time in seconds"""
        
        start_time = datetime.fromisoformat(workflow_state["started_at"])
        end_time = datetime.fromisoformat(
            workflow_state.get("completed_at") or workflow_state.get("failed_at") or datetime.utcnow().isoformat()
        )
        
        return (end_time - start_time).total_seconds()
    
    def _create_executive_summary(self, workflow_state: Dict[str, Any]) -> str:
        """Create executive summary of the assessment"""
        
        final_decision = self._extract_final_decision(workflow_state)
        
        if final_decision["decision"] == "approved":
            return f"Application APPROVED with eligibility score {final_decision['eligibility_score']:.2f}. Monthly support amount: {final_decision['support_amount']} AED."
        elif final_decision["decision"] == "declined":
            return f"Application DECLINED. Eligibility score {final_decision['eligibility_score']:.2f} below minimum requirements."
        else:
            return "Application processing encountered errors and requires manual review."
    
    def _extract_application_summary(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract application summary"""
        return {
            "application_id": workflow_state["application_id"],
            "processed_at": workflow_state.get("completed_at"),
            "processing_status": workflow_state["status"].value
        }
    
    def _extract_eligibility_decision(self, results: Dict[str, Any]) -> str:
        """Extract eligibility decision"""
        eligibility = results.get("eligibility", {})
        if eligibility.get("status") == "success":
            return "ELIGIBLE" if eligibility.get("assessment_result", {}).get("eligible", False) else "NOT ELIGIBLE"
        return "ERROR"
    
    def _extract_support_amount(self, results: Dict[str, Any]) -> float:
        """Extract support amount"""
        eligibility = results.get("eligibility", {})
        if eligibility.get("status") == "success":
            return eligibility.get("assessment_result", {}).get("support_calculation", {}).get("monthly_support_amount", 0)
        return 0
    
    def _generate_next_steps(self, workflow_state: Dict[str, Any]) -> List[str]:
        """Generate next steps for the applicant"""
        
        final_decision = self._extract_final_decision(workflow_state)
        
        if final_decision["decision"] == "approved":
            return [
                "Support approved - funds will be disbursed within 5 business days",
                "Enroll in recommended economic enablement programs",
                "Schedule 3-month review appointment",
                "Maintain updated contact information"
            ]
        elif final_decision["decision"] == "declined":
            return [
                "Review eligibility criteria and address identified gaps",
                "Consider reapplying after improving financial situation",
                "Access available community resources and support programs",
                "Contact social services for additional assistance options"
            ]
        else:
            return [
                "Application requires manual review",
                "Provide additional documentation as requested",
                "Contact case worker for status update"
            ]
    
    def _calculate_workflow_metrics(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate workflow performance metrics"""
        
        return {
            "total_processing_time_seconds": self._calculate_processing_time(workflow_state),
            "steps_completed": len(workflow_state["steps_completed"]),
            "total_steps": 5,
            "success_rate": 1.0 if workflow_state["status"] == WorkflowStatus.COMPLETED else 0.0,
            "error_count": len(workflow_state.get("errors", [])),
            "workflow_log_entries": len(workflow_state.get("workflow_log", []))
        } 