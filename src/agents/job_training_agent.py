"""
Job and Training Recommendations Agent

This agent provides personalized job and training recommendations using ChromaDB
for semantic search and matching. It collects user skills and preferences to
provide targeted recommendations for career development and economic enablement.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from .base_agent import BaseAgent
from src.utils.logging_config import get_logger

logger = get_logger("job_training_agent")

# Import vector store
try:
    from src.services.vector_store import get_vector_store
    VECTOR_STORE_AVAILABLE = True
except ImportError as e:
    VECTOR_STORE_AVAILABLE = False
    print(f"Warning: Vector store not available ({str(e)}), job recommendations will use fallback")


class JobTrainingRecommendationAgent(BaseAgent):
    """
    Job and Training Recommendation Agent
    
    Provides personalized job and training recommendations using ChromaDB semantic search.
    Collects user skills, preferences, and career goals to match with relevant opportunities.
    """
    
    def __init__(self):
        super().__init__("JobTrainingRecommendationAgent")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process job and training recommendation request
        
        Args:
            input_data: {
                "user_skills": List[str] - User's skills and competencies,
                "career_goals": str - User's career aspirations,
                "experience_level": str - User's experience level,
                "preferred_location": str - Preferred work location,
                "salary_expectations": str - Salary range expectations,
                "employment_type": str - Full-time, part-time, contract, etc.
            }
            
        Returns:
            Personalized job and training recommendations with relevance scores
        """
        
        try:
            logger.info("🚀 Starting job and training recommendation process")
            
            # Extract user preferences
            user_skills = input_data.get("user_skills", [])
            career_goals = input_data.get("career_goals", "")
            experience_level = input_data.get("experience_level", "entry")
            preferred_location = input_data.get("preferred_location", "UAE")
            salary_expectations = input_data.get("salary_expectations", "")
            employment_type = input_data.get("employment_type", "full_time")
            
            logger.info(f"👤 User Profile: {len(user_skills)} skills, {experience_level} level, goals: {career_goals}")
            
            # Check if vector store is available
            if not VECTOR_STORE_AVAILABLE:
                logger.warning("Vector store not available, using fallback recommendations")
                return await self._generate_fallback_recommendations(input_data)
            
            # Create user profile for matching
            user_profile = self._create_user_profile(input_data)
            
            # Get vector store instance
            vector_store = get_vector_store()
            
            # Get relevant training programs
            logger.info("📚 Searching for relevant training programs...")
            training_programs = await vector_store.get_relevant_training_programs(
                user_profile, n_results=8
            )
            
            # Get relevant job opportunities
            logger.info("💼 Searching for relevant job opportunities...")
            job_opportunities = await vector_store.get_relevant_job_opportunities(
                user_profile, n_results=8
            )
            
            # Generate comprehensive recommendations
            recommendations = await self._generate_comprehensive_recommendations(
                user_profile,
                training_programs,
                job_opportunities
            )
            
            logger.info(f"✅ Generated recommendations: {len(training_programs)} training programs, {len(job_opportunities)} job opportunities")
            
            return {
                "agent_name": self.agent_name,
                "status": "success",
                "user_profile": user_profile,
                "training_programs": training_programs,
                "job_opportunities": job_opportunities,
                "recommendations": recommendations,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error in job and training recommendations: {str(e)}")
            return {
                "agent_name": self.agent_name,
                "status": "error",
                "error": str(e),
                "fallback_recommendations": await self._generate_fallback_recommendations(input_data),
                "generated_at": datetime.utcnow().isoformat()
            }
    
    def _create_user_profile(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive user profile for ChromaDB matching"""
        
        user_skills = input_data.get("user_skills", [])
        career_goals = input_data.get("career_goals", "")
        experience_level = input_data.get("experience_level", "entry")
        preferred_location = input_data.get("preferred_location", "UAE")
        salary_expectations = input_data.get("salary_expectations", "")
        employment_type = input_data.get("employment_type", "full_time")
        
        # Create comprehensive profile
        user_profile = {
            "skills": user_skills,
            "career_goals": career_goals,
            "experience_level": experience_level,
            "preferred_location": preferred_location,
            "salary_expectations": salary_expectations,
            "employment_type": employment_type,
            "looking_for": "career development, job opportunities, skills training"
        }
        
        logger.info(f"Created user profile: {len(user_skills)} skills, {experience_level} level")
        return user_profile
    
    async def _generate_comprehensive_recommendations(
        self,
        user_profile: Dict[str, Any],
        training_programs: List[Dict[str, Any]],
        job_opportunities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive recommendations based on ChromaDB results"""
        
        # Analyze user profile
        skills = user_profile.get("skills", [])
        career_goals = user_profile.get("career_goals", "")
        experience_level = user_profile.get("experience_level", "entry")
        
        # Generate personalized recommendations
        recommendations = {
            "immediate_actions": [],
            "skill_development_plan": [],
            "career_pathway": [],
            "next_steps": []
        }
        
        # Immediate actions based on experience level
        if experience_level == "entry":
            recommendations["immediate_actions"].extend([
                "🎯 **Start with Foundation Skills**: Focus on the highest-relevance training programs to build core competencies",
                "📝 **Update Your Profile**: Create a professional resume highlighting your current skills and training goals",
                "🔍 **Research Entry-Level Positions**: Review the recommended job opportunities to understand market requirements"
            ])
        elif experience_level == "intermediate":
            recommendations["immediate_actions"].extend([
                "📈 **Skill Enhancement**: Enroll in advanced training programs to strengthen your expertise",
                "🎯 **Target Specific Roles**: Apply for the recommended positions that match your experience level",
                "🤝 **Network Building**: Connect with professionals in your target industry"
            ])
        else:  # senior
            recommendations["immediate_actions"].extend([
                "🎯 **Leadership Development**: Focus on management and leadership training programs",
                "💼 **Senior Role Applications**: Apply for senior positions in the recommended opportunities",
                "🎓 **Mentoring Others**: Consider sharing your expertise while continuing to learn"
            ])
        
        # Skill development plan based on training programs
        if training_programs:
            high_relevance_programs = [p for p in training_programs if p.get("relevance_score", 0) > 0.4]
            if high_relevance_programs:
                recommendations["skill_development_plan"].append(
                    f"🏆 **Priority Training**: Start with '{high_relevance_programs[0].get('metadata', {}).get('program_name', 'Top Program')}' (Relevance: {high_relevance_programs[0].get('relevance_score', 0):.2f})"
                )
            
            if len(training_programs) >= 3:
                recommendations["skill_development_plan"].append(
                    f"📚 **Comprehensive Learning**: We found {len(training_programs)} relevant training programs - consider a structured learning path"
                )
        
        # Career pathway based on job opportunities
        if job_opportunities:
            entry_jobs = [j for j in job_opportunities if "entry" in j.get('metadata', {}).get('requirements', '').lower()]
            senior_jobs = [j for j in job_opportunities if any(word in j.get('metadata', {}).get('requirements', '').lower() for word in ['senior', 'manager', 'lead'])]
            
            if entry_jobs and experience_level == "entry":
                recommendations["career_pathway"].append(
                    f"🚀 **Entry Point**: Start with '{entry_jobs[0].get('metadata', {}).get('job_title', 'Entry Position')}' to gain experience"
                )
            
            if senior_jobs and experience_level in ["intermediate", "senior"]:
                recommendations["career_pathway"].append(
                    f"🎯 **Growth Opportunity**: Target '{senior_jobs[0].get('metadata', {}).get('job_title', 'Senior Position')}' for career advancement"
                )
        
        # Next steps
        recommendations["next_steps"].extend([
            "📞 **Contact Training Providers**: Reach out to the recommended training programs for enrollment details",
            "📧 **Apply for Positions**: Submit applications for the most relevant job opportunities",
            "📅 **Create a Timeline**: Set realistic goals for completing training and job applications",
            "🔄 **Regular Review**: Check back monthly for new opportunities and track your progress"
        ])
        
        # Generate summary
        summary = f"""
🎯 **Personalized Career Development Plan**

Based on your skills ({', '.join(skills[:3])}{'...' if len(skills) > 3 else ''}) and {experience_level} experience level, we've identified:

📚 **{len(training_programs)} Training Programs** - Matched to your skill profile
💼 **{len(job_opportunities)} Job Opportunities** - Aligned with your career goals

**Your personalized recommendations focus on:**
- Building relevant skills through targeted training
- Applying for positions that match your experience level
- Creating a clear pathway for career advancement

All recommendations are ranked by relevance to ensure you focus on the most suitable opportunities for your profile.
"""
        
        recommendations["summary"] = summary.strip()
        
        return recommendations
    
    async def _generate_fallback_recommendations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback recommendations when ChromaDB is not available"""
        
        logger.warning("Using fallback recommendations due to ChromaDB unavailability")
        
        user_skills = input_data.get("user_skills", [])
        experience_level = input_data.get("experience_level", "entry")
        
        # Basic fallback training programs
        fallback_training = [
            {
                "name": "Digital Skills Foundation",
                "duration": "3 months",
                "description": "Essential computer skills and digital literacy",
                "provider": "UAE Digital Academy",
                "cost": "Free",
                "relevance_score": 0.8,
                "source": "fallback"
            },
            {
                "name": "Professional Communication",
                "duration": "2 months", 
                "description": "Business communication and presentation skills",
                "provider": "Professional Development Center",
                "cost": "Subsidized",
                "relevance_score": 0.7,
                "source": "fallback"
            }
        ]
        
        # Basic fallback job opportunities
        fallback_jobs = [
            {
                "title": "Customer Service Representative",
                "company": "Various Companies",
                "salary_range": "3000-4500 AED",
                "requirements": "Basic communication skills",
                "relevance_score": 0.6,
                "source": "fallback"
            },
            {
                "title": "Administrative Assistant",
                "company": "Business Offices",
                "salary_range": "2800-3800 AED", 
                "requirements": "Computer skills, organization",
                "relevance_score": 0.5,
                "source": "fallback"
            }
        ]
        
        return {
            "training_programs": fallback_training,
            "job_opportunities": fallback_jobs,
            "recommendations": {
                "summary": "Basic career development recommendations based on general market opportunities.",
                "immediate_actions": [
                    "📚 Explore available training programs to build your skills",
                    "💼 Apply for entry-level positions to gain experience",
                    "🎯 Focus on developing in-demand skills for better opportunities"
                ]
            },
            "source": "fallback_system"
        }
    
    async def collect_user_skills(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Interactive skill collection through conversation
        
        This method can be used to collect user skills through a conversational interface
        """
        
        # This would be implemented to handle conversational skill collection
        # For now, return a template for the frontend to use
        
        return {
            "skill_collection_questions": [
                {
                    "question": "What are your main skills or areas of expertise?",
                    "type": "multi_select",
                    "options": [
                        "Computer Skills", "Customer Service", "Communication", "Sales",
                        "Administrative", "Technical", "Creative", "Management",
                        "Language Skills", "Problem Solving", "Teamwork", "Leadership"
                    ]
                },
                {
                    "question": "What is your experience level?",
                    "type": "single_select",
                    "options": ["Entry Level (0-2 years)", "Intermediate (2-5 years)", "Senior (5+ years)"]
                },
                {
                    "question": "What are your career goals?",
                    "type": "text",
                    "placeholder": "e.g., I want to work in customer service and eventually become a team leader"
                },
                {
                    "question": "What type of employment are you looking for?",
                    "type": "single_select",
                    "options": ["Full-time", "Part-time", "Contract", "Freelance", "Any"]
                },
                {
                    "question": "What salary range are you expecting? (AED per month)",
                    "type": "single_select",
                    "options": ["2000-3000", "3000-4000", "4000-5000", "5000+", "Open to discussion"]
                }
            ]
        } 