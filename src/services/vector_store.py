"""
Vector Store Service using ChromaDB for Social Support AI System

This service provides semantic search and similarity matching capabilities for:
1. Document knowledge base and retrieval
2. Inconsistency detection across documents  
3. Historical decision context
4. Economic enablement recommendations
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime

from src.utils.logging_config import get_logger
logger = get_logger("vector_store")


class SocialSupportVectorStore:
    """Vector store for social support application system using ChromaDB"""
    
    def __init__(self, persist_directory: str = "data/chroma"):
        """Initialize ChromaDB client and collections"""
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize collections for different data types
        self.collections = {}
        self._initialize_collections()
        
        logger.info(f"Vector store initialized with persistence at {persist_directory}")
    
    def _initialize_collections(self):
        """Initialize ChromaDB collections for different data types"""
        
        collection_configs = {
            "applications": {
                "metadata": {"description": "Complete application profiles for similarity matching"}
            },
            "documents": {
                "metadata": {"description": "Extracted document content for semantic search"}
            },
            "historical_decisions": {
                "metadata": {"description": "Past eligibility decisions for consistency checking"}
            },
            "training_programs": {
                "metadata": {"description": "Available training and enablement programs"}
            },
            "job_opportunities": {
                "metadata": {"description": "Job matching database for recommendations"}
            },
            "inconsistency_patterns": {
                "metadata": {"description": "Known patterns of document inconsistencies"}
            }
        }
        
        for name, config in collection_configs.items():
            try:
                # Try to get existing collection
                collection = self.client.get_collection(name)
                logger.info(f"Loaded existing collection: {name}")
            except ValueError:
                # Create new collection if it doesn't exist
                # ChromaDB will use default embedding function if none specified
                collection = self.client.create_collection(
                    name=name,
                    metadata=config["metadata"]
                )
                logger.info(f"Created new collection: {name}")
            
            self.collections[name] = collection
    
    async def store_application(self, application_data: Dict[str, Any], application_id: str) -> bool:
        """Store application data for future similarity matching"""
        
        try:
            # Create searchable text representation of application
            application_text = self._create_application_summary(application_data)
            
            # Store in applications collection
            self.collections["applications"].add(
                documents=[application_text],
                metadatas=[{
                    "application_id": application_id,
                    "employment_status": application_data.get("employment_status", "unknown"),
                    "monthly_income": application_data.get("monthly_income", 0),
                    "family_size": application_data.get("family_size", 1),
                    "housing_status": application_data.get("housing_status", "unknown"),
                    "created_at": datetime.now().isoformat()
                }],
                ids=[application_id]
            )
            
            logger.info(f"Stored application {application_id} in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error storing application {application_id}: {str(e)}")
            return False
    
    async def store_document_content(self, document_content: str, document_metadata: Dict[str, Any]) -> bool:
        """Store extracted document content for semantic search"""
        
        try:
            document_id = f"{document_metadata.get('application_id', 'unknown')}_{document_metadata.get('document_type', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.collections["documents"].add(
                documents=[document_content],
                metadatas=[document_metadata],
                ids=[document_id]
            )
            
            logger.info(f"Stored document content: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document content: {str(e)}")
            return False
    
    async def store_decision(self, application_data: Dict[str, Any], decision_data: Dict[str, Any], application_id: str) -> bool:
        """Store eligibility decision for historical context"""
        
        try:
            # Create decision summary text
            decision_text = self._create_decision_summary(application_data, decision_data)
            
            decision_metadata = {
                "application_id": application_id,
                "decision": decision_data.get("decision", "unknown"),
                "eligible": decision_data.get("eligible", False),
                "support_amount": decision_data.get("support_amount", 0),
                "employment_status": application_data.get("employment_status", "unknown"),
                "monthly_income": application_data.get("monthly_income", 0),
                "family_size": application_data.get("family_size", 1),
                "decided_at": datetime.now().isoformat()
            }
            
            decision_id = f"decision_{application_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.collections["historical_decisions"].add(
                documents=[decision_text],
                metadatas=[decision_metadata],
                ids=[decision_id]
            )
            
            logger.info(f"Stored decision for application {application_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing decision: {str(e)}")
            return False
    
    async def find_similar_applications(self, current_application: Dict[str, Any], n_results: int = 5) -> List[Dict[str, Any]]:
        """Find similar applications for consistency checking"""
        
        try:
            # Create query text from current application
            query_text = self._create_application_summary(current_application)
            
            # Search for similar applications
            results = self.collections["applications"].query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            similar_apps = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    similar_apps.append({
                        "document": doc,
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                    })
            
            logger.info(f"Found {len(similar_apps)} similar applications")
            return similar_apps
            
        except Exception as e:
            logger.error(f"Error finding similar applications: {str(e)}")
            return []
    
    async def detect_document_inconsistencies(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect inconsistencies across multiple documents using semantic similarity"""
        
        try:
            inconsistencies = []
            
            # Compare income statements across documents
            income_statements = []
            for doc in documents:
                if "income" in doc.get("content", "").lower():
                    income_statements.append(doc)
            
            # Use vector similarity to detect conflicting income information
            if len(income_statements) > 1:
                for i, doc1 in enumerate(income_statements):
                    for doc2 in income_statements[i+1:]:
                        # Query similarity between income statements
                        similarity_results = self.collections["documents"].query(
                            query_texts=[doc1["content"]],
                            where={"document_type": doc2["document_type"]},
                            n_results=1
                        )
                        
                        # If similarity is low, potential inconsistency
                        if similarity_results["distances"] and similarity_results["distances"][0]:
                            distance = similarity_results["distances"][0][0]
                            if distance > 0.7:  # High distance = low similarity
                                inconsistencies.append({
                                    "type": "income_inconsistency",
                                    "document1": doc1["document_type"],
                                    "document2": doc2["document_type"],
                                    "confidence": 1 - distance,
                                    "description": f"Potential income inconsistency between {doc1['document_type']} and {doc2['document_type']}"
                                })
            
            logger.info(f"Detected {len(inconsistencies)} potential inconsistencies")
            return inconsistencies
            
        except Exception as e:
            logger.error(f"Error detecting inconsistencies: {str(e)}")
            return []
    
    async def get_relevant_training_programs(self, user_profile: Dict[str, Any], n_results: int = 5) -> List[Dict[str, Any]]:
        """Get relevant training programs based on user profile"""
        
        try:
            # Create profile query text
            profile_text = self._create_profile_summary(user_profile)
            
            # Search for relevant training programs
            results = self.collections["training_programs"].query(
                query_texts=[profile_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            programs = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i]
                    # Convert distance to similarity score (closer to 0 = more similar)
                    relevance_score = max(0, 1 - distance) if distance <= 1 else 1 / (1 + distance)
                    
                    programs.append({
                        "program_description": doc,
                        "metadata": results["metadatas"][0][i],
                        "relevance_score": relevance_score
                    })
            
            logger.info(f"Found {len(programs)} relevant training programs")
            return programs
            
        except Exception as e:
            logger.error(f"Error finding training programs: {str(e)}")
            return []
    
    async def get_relevant_job_opportunities(self, user_profile: Dict[str, Any], n_results: int = 5) -> List[Dict[str, Any]]:
        """Get relevant job opportunities based on user profile"""
        
        try:
            # Create profile query text for job matching
            profile_text = self._create_job_profile_summary(user_profile)
            
            # Search for relevant job opportunities
            results = self.collections["job_opportunities"].query(
                query_texts=[profile_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            jobs = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i]
                    # Convert distance to similarity score (closer to 0 = more similar)
                    relevance_score = max(0, 1 - distance) if distance <= 1 else 1 / (1 + distance)
                    
                    jobs.append({
                        "job_description": doc,
                        "metadata": results["metadatas"][0][i],
                        "relevance_score": relevance_score
                    })
            
            logger.info(f"Found {len(jobs)} relevant job opportunities")
            return jobs
            
        except Exception as e:
            logger.error(f"Error finding job opportunities: {str(e)}")
            return []
    
    async def get_historical_decision_context(self, current_application: Dict[str, Any], n_results: int = 3) -> List[Dict[str, Any]]:
        """Get similar historical decisions for consistency"""
        
        try:
            # Create query from current application
            query_text = self._create_application_summary(current_application)
            
            # Search historical decisions
            results = self.collections["historical_decisions"].query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            historical_context = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    historical_context.append({
                        "decision_summary": doc,
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": 1 - results["distances"][0][i]
                    })
            
            logger.info(f"Found {len(historical_context)} similar historical decisions")
            return historical_context
            
        except Exception as e:
            logger.error(f"Error getting historical context: {str(e)}")
            return []
    
    def _create_application_summary(self, application_data: Dict[str, Any]) -> str:
        """Create searchable text summary of application"""
        
        employment_status = application_data.get("employment_status", "unknown")
        monthly_income = application_data.get("monthly_income", 0)
        family_size = application_data.get("family_size", 1)
        housing_status = application_data.get("housing_status", "unknown")
        
        return f"""
        Employment: {employment_status}
        Monthly Income: {monthly_income} AED
        Family Size: {family_size} people
        Housing: {housing_status}
        Income per person: {monthly_income / family_size if family_size > 0 else 0:.0f} AED
        """
    
    def _create_decision_summary(self, application_data: Dict[str, Any], decision_data: Dict[str, Any]) -> str:
        """Create searchable text summary of decision"""
        
        app_summary = self._create_application_summary(application_data)
        decision = decision_data.get("decision", "unknown")
        support_amount = decision_data.get("support_amount", 0)
        
        return f"""
        {app_summary}
        Decision: {decision}
        Support Amount: {support_amount} AED
        Reasoning: {decision_data.get("reason", "No reason provided")}
        """
    
    def _create_profile_summary(self, user_profile: Dict[str, Any]) -> str:
        """Create searchable text summary of user profile for training matching"""
        
        employment_status = user_profile.get("employment_status", "unknown")
        skills = user_profile.get("skills", [])
        education = user_profile.get("education_level", "unknown")
        
        return f"""
        Employment Status: {employment_status}
        Skills: {', '.join(skills) if skills else 'No specific skills listed'}
        Education Level: {education}
        Looking for: career development, job opportunities, skills training
        """
    
    def _create_job_profile_summary(self, user_profile: Dict[str, Any]) -> str:
        """Create searchable text summary of user profile for job matching"""
        
        employment_status = user_profile.get("employment_status", "unknown")
        skills = user_profile.get("skills", [])
        education = user_profile.get("education_level", "unknown")
        monthly_income = user_profile.get("monthly_income", 0)
        work_experience = user_profile.get("work_experience_years", 0)
        
        return f"""
        Employment Status: {employment_status}
        Current Income: {monthly_income} AED
        Skills: {', '.join(skills) if skills else 'No specific skills listed'}
        Education Level: {education}
        Work Experience: {work_experience} years
        Looking for: job opportunities, employment, career advancement
        """
    
    async def populate_sample_data(self):
        """Populate vector store with sample training programs and job opportunities"""
        
        try:
            # Sample training programs
            training_programs = [
                {
                    "content": "Digital Skills Training Program: Learn essential computer skills, Microsoft Office, and basic digital literacy for office jobs",
                    "metadata": {
                        "program_name": "Digital Skills Training",
                        "duration": "3 months",
                        "target_audience": "unemployed, basic education",
                        "skills": ["computer_skills", "microsoft_office", "digital_literacy"],
                        "cost": "free"
                    }
                },
                {
                    "content": "Vocational Training Certificate: Hands-on training in trades like plumbing, electrical work, automotive repair for technical careers",
                    "metadata": {
                        "program_name": "Vocational Training",
                        "duration": "6 months", 
                        "target_audience": "unemployed, manual_work_experience",
                        "skills": ["plumbing", "electrical", "automotive"],
                        "cost": "subsidized"
                    }
                },
                {
                    "content": "English Language Course: Improve English communication skills for better job opportunities in international companies",
                    "metadata": {
                        "program_name": "English Language Course",
                        "duration": "4 months",
                        "target_audience": "all_levels",
                        "skills": ["english", "communication"],
                        "cost": "free"
                    }
                }
            ]
            
            # Add training programs to vector store
            for i, program in enumerate(training_programs):
                self.collections["training_programs"].add(
                    documents=[program["content"]],
                    metadatas=[program["metadata"]],
                    ids=[f"training_program_{i}"]
                )
            
            # Sample job opportunities
            job_opportunities = [
                {
                    "content": "Customer Service Representative: Handle customer inquiries, basic computer skills required, flexible schedule available",
                    "metadata": {
                        "job_title": "Customer Service Representative",
                        "company": "Various Companies",
                        "salary_range": "3000-4500 AED",
                        "requirements": ["basic_english", "computer_skills"],
                        "employment_type": "full_time"
                    }
                },
                {
                    "content": "Retail Sales Associate: Assist customers in shopping centers, customer service skills, flexible hours",
                    "metadata": {
                        "job_title": "Retail Sales Associate",
                        "company": "Shopping Centers",
                        "salary_range": "2500-3500 AED",
                        "requirements": ["customer_service", "flexible_schedule"],
                        "employment_type": "part_time"
                    }
                }
            ]
            
            # Add job opportunities to vector store
            for i, job in enumerate(job_opportunities):
                self.collections["job_opportunities"].add(
                    documents=[job["content"]],
                    metadatas=[job["metadata"]],
                    ids=[f"job_opportunity_{i}"]
                )
            
            logger.info("Sample data populated successfully")
            
        except Exception as e:
            logger.error(f"Error populating sample data: {str(e)}")


# Singleton instance
_vector_store_instance = None

def get_vector_store() -> SocialSupportVectorStore:
    """Get singleton vector store instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = SocialSupportVectorStore()
    return _vector_store_instance 