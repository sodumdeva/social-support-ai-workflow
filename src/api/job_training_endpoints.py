"""
Job and Training Recommendations API Endpoints

Provides REST API endpoints for job and training recommendations using ChromaDB.
Includes skill collection, recommendation generation, and personalized career guidance.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from src.agents.job_training_agent import JobTrainingRecommendationAgent
from src.utils.logging_config import get_logger

logger = get_logger("job_training_api")

# Create router
router = APIRouter(prefix="/job-training", tags=["Job & Training Recommendations"])


# Pydantic models for request/response
class SkillCollectionRequest(BaseModel):
    """Request model for skill collection"""
    pass


class JobTrainingRequest(BaseModel):
    """Request model for job and training recommendations"""
    user_skills: List[str] = Field(..., description="List of user skills and competencies")
    career_goals: str = Field("", description="User's career aspirations and goals")
    experience_level: str = Field("entry", description="Experience level: entry, intermediate, or senior")
    preferred_location: str = Field("UAE", description="Preferred work location")
    salary_expectations: str = Field("", description="Expected salary range")
    employment_type: str = Field("full_time", description="Preferred employment type")


class TrainingProgram(BaseModel):
    """Training program model"""
    name: str
    duration: str
    description: str
    provider: str
    cost: str
    relevance_score: float
    metadata: Dict[str, Any] = {}


class JobOpportunity(BaseModel):
    """Job opportunity model"""
    title: str
    company: str
    salary_range: str
    requirements: str
    description: str = ""
    relevance_score: float
    metadata: Dict[str, Any] = {}


class RecommendationsResponse(BaseModel):
    """Response model for recommendations"""
    status: str
    user_profile: Dict[str, Any]
    training_programs: List[Dict[str, Any]]
    job_opportunities: List[Dict[str, Any]]
    recommendations: Dict[str, Any]
    generated_at: str


@router.get("/skill-collection-form")
async def get_skill_collection_form():
    """
    Get the skill collection form structure for the frontend
    
    Returns the questions and options for collecting user skills and preferences
    """
    try:
        logger.info("📋 Providing skill collection form structure")
        
        agent = JobTrainingRecommendationAgent()
        form_structure = await agent.collect_user_skills([])
        
        return {
            "status": "success",
            "form_structure": form_structure,
            "instructions": {
                "title": "🎯 Career Development Recommendations",
                "subtitle": "Tell us about your skills and goals to get personalized job and training recommendations",
                "description": "We'll use ChromaDB to find the most relevant opportunities for your profile"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error providing skill collection form: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error providing form structure: {str(e)}")


@router.post("/recommendations", response_model=RecommendationsResponse)
async def get_job_training_recommendations(request: JobTrainingRequest):
    """
    Generate personalized job and training recommendations
    
    Uses ChromaDB to find relevant training programs and job opportunities
    based on user skills, experience level, and career goals.
    """
    try:
        logger.info(f"🚀 Generating recommendations for user with {len(request.user_skills)} skills")
        
        # Initialize the job training agent
        agent = JobTrainingRecommendationAgent()
        
        # Prepare input data
        input_data = {
            "user_skills": request.user_skills,
            "career_goals": request.career_goals,
            "experience_level": request.experience_level,
            "preferred_location": request.preferred_location,
            "salary_expectations": request.salary_expectations,
            "employment_type": request.employment_type
        }
        
        # Generate recommendations
        result = await agent.process(input_data)
        
        if result.get("status") == "success":
            logger.info(f"✅ Successfully generated recommendations")
            return RecommendationsResponse(**result)
        else:
            # Handle error case with fallback
            error_msg = result.get("error", "Unknown error occurred")
            logger.error(f"❌ Recommendation generation failed: {error_msg}")
            
            # Return fallback recommendations if available
            if "fallback_recommendations" in result:
                fallback = result["fallback_recommendations"]
                return RecommendationsResponse(
                    status="success_with_fallback",
                    user_profile=input_data,
                    training_programs=fallback.get("training_programs", []),
                    job_opportunities=fallback.get("job_opportunities", []),
                    recommendations=fallback.get("recommendations", {}),
                    generated_at=datetime.utcnow().isoformat()
                )
            else:
                raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {error_msg}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in recommendations endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/sample-skills")
async def get_sample_skills():
    """
    Get sample skills for user selection
    
    Provides a comprehensive list of skills organized by category
    to help users identify their competencies.
    """
    try:
        logger.info("📚 Providing sample skills list")
        
        sample_skills = {
            "technical_skills": [
                "Computer Skills", "Microsoft Office", "Data Entry", "Database Management",
                "Web Development", "Programming", "Digital Marketing", "Social Media Management",
                "Graphic Design", "Video Editing", "Technical Support", "Network Administration"
            ],
            "soft_skills": [
                "Communication", "Leadership", "Teamwork", "Problem Solving",
                "Time Management", "Customer Service", "Sales", "Negotiation",
                "Project Management", "Training & Development", "Public Speaking", "Adaptability"
            ],
            "industry_specific": [
                "Healthcare", "Education", "Finance", "Retail", "Hospitality",
                "Construction", "Manufacturing", "Transportation", "Security",
                "Food Service", "Administrative", "Legal", "Real Estate"
            ],
            "language_skills": [
                "Arabic (Native)", "English (Fluent)", "English (Basic)",
                "Hindi", "Urdu", "French", "German", "Spanish", "Other Languages"
            ]
        }
        
        return {
            "status": "success",
            "skills_by_category": sample_skills,
            "total_skills": sum(len(skills) for skills in sample_skills.values())
        }
        
    except Exception as e:
        logger.error(f"❌ Error providing sample skills: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error providing sample skills: {str(e)}")


@router.get("/career-guidance")
async def get_career_guidance():
    """
    Get general career guidance and tips
    
    Provides general career development advice and guidance
    for users exploring their options.
    """
    try:
        logger.info("💡 Providing career guidance")
        
        guidance = {
            "career_development_tips": [
                "🎯 **Identify Your Strengths**: Focus on skills you enjoy and excel at",
                "📚 **Continuous Learning**: Stay updated with industry trends and new skills",
                "🤝 **Network Building**: Connect with professionals in your field of interest",
                "📝 **Professional Profile**: Keep your resume and LinkedIn profile updated",
                "🎪 **Gain Experience**: Look for internships, volunteer work, or entry-level positions"
            ],
            "skill_development_advice": [
                "Start with in-demand skills in your area of interest",
                "Combine technical skills with soft skills for better opportunities",
                "Practice regularly and build a portfolio of your work",
                "Seek feedback from mentors or experienced professionals",
                "Consider online courses and certifications"
            ],
            "job_search_strategies": [
                "Tailor your resume for each job application",
                "Use multiple job search platforms and company websites",
                "Prepare for interviews by researching the company",
                "Follow up on applications professionally",
                "Consider working with recruitment agencies"
            ],
            "resources": {
                "training_providers": [
                    "UAE Digital Skills Academy",
                    "Technical Education Institute", 
                    "Professional Development Centers",
                    "Online Learning Platforms"
                ],
                "job_portals": [
                    "UAE Government Jobs Portal",
                    "Major Job Search Websites",
                    "Company Career Pages",
                    "Professional Networks"
                ]
            }
        }
        
        return {
            "status": "success",
            "career_guidance": guidance,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error providing career guidance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error providing career guidance: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for job training recommendations service"""
    try:
        # Test ChromaDB availability
        from src.services.vector_store import get_vector_store
        vector_store = get_vector_store()
        chromadb_status = "available"
    except Exception as e:
        chromadb_status = f"unavailable: {str(e)}"
    
    return {
        "status": "healthy",
        "service": "job_training_recommendations",
        "chromadb_status": chromadb_status,
        "timestamp": datetime.utcnow().isoformat()
    } 