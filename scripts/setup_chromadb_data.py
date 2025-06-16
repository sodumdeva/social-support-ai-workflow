#!/usr/bin/env python3
"""
Setup ChromaDB with Sample Data for Social Support AI Workflow

This script populates the ChromaDB vector store with sample training programs
and job opportunities for testing the economic enablement integration.
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.services.vector_store import get_vector_store
from src.utils.logging_config import get_logger

logger = get_logger("chromadb_setup")


async def populate_training_programs():
    """Populate ChromaDB with comprehensive training programs"""
    
    vector_store = get_vector_store()
    
    training_programs = [
        {
            "content": "Digital Skills Training Program: Comprehensive computer literacy course covering Microsoft Office, email, internet browsing, and basic digital tools for modern workplace",
            "metadata": {
                "program_name": "Digital Skills Training",
                "duration": "3 months",
                "target_audience": "unemployed, basic education",
                "skills": ["computer_skills", "microsoft_office", "digital_literacy", "email", "internet"],
                "cost": "free",
                "provider": "UAE Digital Skills Academy",
                "contact": "800-SKILLS"
            }
        },
        {
            "content": "Vocational Training Certificate: Hands-on training in technical trades including plumbing, electrical work, automotive repair, and HVAC systems",
            "metadata": {
                "program_name": "Vocational Training",
                "duration": "6 months", 
                "target_audience": "unemployed, manual_work_experience",
                "skills": ["plumbing", "electrical", "automotive", "hvac", "technical_skills"],
                "cost": "subsidized",
                "provider": "Technical Education Institute",
                "contact": "04-123-4567"
            }
        },
        {
            "content": "English Language Course: Improve English communication skills for better job opportunities in international companies and customer service roles",
            "metadata": {
                "program_name": "English Language Course",
                "duration": "4 months",
                "target_audience": "all_levels",
                "skills": ["english", "communication", "customer_service"],
                "cost": "free",
                "provider": "Community Learning Center",
                "contact": "02-987-6543"
            }
        },
        {
            "content": "Customer Service Excellence: Professional training in customer service skills, phone etiquette, complaint handling, and sales techniques",
            "metadata": {
                "program_name": "Customer Service Excellence",
                "duration": "2 months",
                "target_audience": "unemployed, retail_experience",
                "skills": ["customer_service", "communication", "sales", "phone_skills"],
                "cost": "free",
                "provider": "Service Excellence Institute",
                "contact": "800-SERVICE"
            }
        },
        {
            "content": "Food Safety and Hospitality Training: Certification in food handling, kitchen safety, restaurant service, and hospitality management",
            "metadata": {
                "program_name": "Food Safety & Hospitality",
                "duration": "1 month",
                "target_audience": "unemployed, hospitality_interest",
                "skills": ["food_safety", "hospitality", "restaurant_service", "kitchen_skills"],
                "cost": "subsidized",
                "provider": "Hospitality Training Center",
                "contact": "04-555-FOOD"
            }
        },
        {
            "content": "Basic Accounting and Bookkeeping: Learn fundamental accounting principles, bookkeeping, and financial record management for small businesses",
            "metadata": {
                "program_name": "Basic Accounting",
                "duration": "3 months",
                "target_audience": "unemployed, high_school_education",
                "skills": ["accounting", "bookkeeping", "financial_management", "excel"],
                "cost": "subsidized",
                "provider": "Business Skills Institute",
                "contact": "02-ACCOUNT"
            }
        }
    ]
    
    # Add training programs to ChromaDB
    for i, program in enumerate(training_programs):
        try:
            # Convert list metadata to strings for ChromaDB compatibility
            metadata = program["metadata"].copy()
            if "skills" in metadata and isinstance(metadata["skills"], list):
                metadata["skills"] = ", ".join(metadata["skills"])
            
            vector_store.collections["training_programs"].add(
                documents=[program["content"]],
                metadatas=[metadata],
                ids=[f"training_program_{i}"]
            )
            logger.info(f"✅ Added training program: {program['metadata']['program_name']}")
        except Exception as e:
            logger.error(f"❌ Error adding training program {i}: {str(e)}")
    
    logger.info(f"📚 Successfully populated {len(training_programs)} training programs")


async def populate_job_opportunities():
    """Populate ChromaDB with comprehensive job opportunities"""
    
    vector_store = get_vector_store()
    
    job_opportunities = [
        {
            "content": "Customer Service Representative: Handle customer inquiries via phone and email, resolve complaints, process orders, basic computer skills required",
            "metadata": {
                "job_title": "Customer Service Representative",
                "company_type": "Various Companies",
                "salary_range": "3000-4500 AED",
                "requirements": ["basic_english", "computer_skills", "communication"],
                "employment_type": "full_time",
                "contact": "UAE Employment Center - 800-JOBS"
            }
        },
        {
            "content": "Retail Sales Associate: Assist customers in shopping centers, handle cash transactions, maintain store displays, flexible schedule available",
            "metadata": {
                "job_title": "Retail Sales Associate",
                "company_type": "Shopping Centers",
                "salary_range": "2500-3500 AED",
                "requirements": ["customer_service", "cash_handling", "flexible_schedule"],
                "employment_type": "part_time",
                "contact": "Retail Jobs Portal - jobs.uae.gov"
            }
        },
        {
            "content": "Food Service Worker: Prepare food, serve customers, maintain kitchen cleanliness, food safety certification provided by employer",
            "metadata": {
                "job_title": "Food Service Worker",
                "company_type": "Restaurants & Hotels",
                "salary_range": "2200-3200 AED",
                "requirements": ["food_safety", "teamwork", "physical_stamina"],
                "employment_type": "full_time",
                "contact": "Hospitality Jobs Center - 04-555-0123"
            }
        },
        {
            "content": "Office Assistant: Basic administrative tasks, filing, data entry, answering phones, Microsoft Office skills preferred",
            "metadata": {
                "job_title": "Office Assistant",
                "company_type": "Small Businesses",
                "salary_range": "2800-3800 AED",
                "requirements": ["microsoft_office", "data_entry", "organization"],
                "employment_type": "full_time",
                "contact": "Business Support Services - 02-OFFICE"
            }
        },
        {
            "content": "Warehouse Worker: Package handling, inventory management, forklift operation (training provided), physical work environment",
            "metadata": {
                "job_title": "Warehouse Worker",
                "company_type": "Logistics Companies",
                "salary_range": "2600-3400 AED",
                "requirements": ["physical_fitness", "teamwork", "attention_to_detail"],
                "employment_type": "full_time",
                "contact": "Logistics Employment - 800-WAREHOUSE"
            }
        },
        {
            "content": "Security Guard: Monitor premises, check visitor credentials, patrol assigned areas, security training certification required",
            "metadata": {
                "job_title": "Security Guard",
                "company_type": "Security Companies",
                "salary_range": "2400-3200 AED",
                "requirements": ["security_certification", "alertness", "physical_fitness"],
                "employment_type": "shift_work",
                "contact": "Security Services - 04-SECURE"
            }
        },
        {
            "content": "Delivery Driver: Deliver packages and food orders, valid UAE driving license required, own vehicle preferred but not mandatory",
            "metadata": {
                "job_title": "Delivery Driver",
                "company_type": "Delivery Services",
                "salary_range": "2800-4000 AED",
                "requirements": ["uae_driving_license", "navigation_skills", "customer_service"],
                "employment_type": "flexible_hours",
                "contact": "Delivery Jobs - 800-DELIVER"
            }
        },
        {
            "content": "Housekeeping Staff: Clean and maintain residential and commercial properties, attention to detail, flexible working hours",
            "metadata": {
                "job_title": "Housekeeping Staff",
                "company_type": "Cleaning Services",
                "salary_range": "2200-2800 AED",
                "requirements": ["attention_to_detail", "reliability", "physical_stamina"],
                "employment_type": "part_time",
                "contact": "Cleaning Services - 02-CLEAN"
            }
        }
    ]
    
    # Add job opportunities to ChromaDB
    for i, job in enumerate(job_opportunities):
        try:
            # Convert list metadata to strings for ChromaDB compatibility
            metadata = job["metadata"].copy()
            if "requirements" in metadata and isinstance(metadata["requirements"], list):
                metadata["requirements"] = ", ".join(metadata["requirements"])
            
            vector_store.collections["job_opportunities"].add(
                documents=[job["content"]],
                metadatas=[metadata],
                ids=[f"job_opportunity_{i}"]
            )
            logger.info(f"✅ Added job opportunity: {job['metadata']['job_title']}")
        except Exception as e:
            logger.error(f"❌ Error adding job opportunity {i}: {str(e)}")
    
    logger.info(f"💼 Successfully populated {len(job_opportunities)} job opportunities")


async def test_chromadb_integration():
    """Test ChromaDB integration with sample queries"""
    
    vector_store = get_vector_store()
    
    # Test user profile
    test_profile = {
        "employment_status": "unemployed",
        "monthly_income": 0,
        "education_level": "high_school",
        "skills": ["basic_computer", "customer_service"],
        "family_size": 3,
        "work_experience_years": 2
    }
    
    logger.info("🧪 Testing ChromaDB integration...")
    
    # Test training program matching
    training_programs = await vector_store.get_relevant_training_programs(test_profile, n_results=3)
    logger.info(f"📚 Found {len(training_programs)} relevant training programs:")
    for program in training_programs:
        metadata = program.get("metadata", {})
        relevance = program.get("relevance_score", 0)
        logger.info(f"  - {metadata.get('program_name', 'Unknown')} (Relevance: {relevance:.2f})")
    
    # Test job opportunity matching
    job_opportunities = await vector_store.get_relevant_job_opportunities(test_profile, n_results=3)
    logger.info(f"💼 Found {len(job_opportunities)} relevant job opportunities:")
    for job in job_opportunities:
        metadata = job.get("metadata", {})
        relevance = job.get("relevance_score", 0)
        logger.info(f"  - {metadata.get('job_title', 'Unknown')} (Relevance: {relevance:.2f})")
    
    logger.info("✅ ChromaDB integration test completed successfully!")


async def main():
    """Main setup function"""
    
    logger.info("🚀 Starting ChromaDB setup for Social Support AI Workflow")
    
    try:
        # Initialize vector store
        vector_store = get_vector_store()
        logger.info("📊 ChromaDB vector store initialized")
        
        # Populate training programs
        await populate_training_programs()
        
        # Populate job opportunities
        await populate_job_opportunities()
        
        # Test integration
        await test_chromadb_integration()
        
        logger.info("🎉 ChromaDB setup completed successfully!")
        logger.info("💡 Economic enablement recommendations will now use ChromaDB for personalized matching")
        
    except Exception as e:
        logger.error(f"❌ ChromaDB setup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 