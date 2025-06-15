"""
ML Model API Endpoints for Social Support AI Workflow

Provides REST API endpoints for:
- Training ML models
- Making predictions
- Model evaluation
- Model management
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

from ..models.ml_models import SocialSupportMLModels
from ..data.synthetic_data import SyntheticDataGenerator
from loguru import logger

# Create API router
router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Global ML models instance
ml_models = None

# Pydantic models for API
class PredictionRequest(BaseModel):
    application_data: Dict[str, Any]
    extracted_documents: Optional[Dict[str, Any]] = {}

class TrainingRequest(BaseModel):
    n_samples: int = 1000
    force_retrain: bool = False

class EligibilityPrediction(BaseModel):
    eligible: bool
    confidence: float
    probability_eligible: float
    feature_importance: Dict[str, float]
    model_type: str

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    features: List[str]
    trained: bool

def get_ml_models():
    """Get or initialize ML models instance (singleton pattern)"""
    global ml_models
    if ml_models is None:
        ml_models = SocialSupportMLModels()  # Singleton - loads models only once
        logger.info("Initialized ML models singleton instance")
    return ml_models

@router.get("/status")
async def get_ml_status():
    """Get ML system status and model information"""
    
    models = get_ml_models()
    model_info = models.get_model_info()
    
    return {
        "status": "operational",
        "models_loaded": len([m for m in model_info.values() if m['trained']]),
        "total_models": len(model_info),
        "model_details": model_info,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/predict/eligibility")
async def predict_eligibility(request: PredictionRequest):
    """
    Predict eligibility using Random Forest classifier
    
    Returns detailed eligibility prediction with confidence scores
    """
    
    try:
        models = get_ml_models()
        
        prediction = models.predict_eligibility(
            request.application_data, 
            request.extracted_documents
        )
        
        if "error" in prediction:
            raise HTTPException(status_code=500, detail=prediction["error"])
        
        return {
            "prediction": prediction,
            "model_used": "random_forest_classifier",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in eligibility prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/risk")
async def predict_risk_level(request: PredictionRequest):
    """
    Predict risk level using Gradient Boosting classifier
    
    Returns risk level (low/medium/high) with confidence
    """
    
    try:
        models = get_ml_models()
        
        prediction = models.predict_risk_level(
            request.application_data,
            request.extracted_documents
        )
        
        if "error" in prediction:
            raise HTTPException(status_code=500, detail=prediction["error"])
        
        return {
            "prediction": prediction,
            "model_used": "gradient_boosting_classifier",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in risk prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/support-amount")
async def predict_support_amount(request: PredictionRequest):
    """
    Predict support amount bracket using Multi-class Random Forest
    
    Returns predicted support amount bracket and estimated amount
    """
    
    try:
        models = get_ml_models()
        
        prediction = models.predict_support_amount(
            request.application_data,
            request.extracted_documents
        )
        
        if "error" in prediction:
            raise HTTPException(status_code=500, detail=prediction["error"])
        
        return {
            "prediction": prediction,
            "model_used": "random_forest_multiclass",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in support amount prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/fraud")
async def detect_fraud(request: PredictionRequest):
    """
    Detect potential fraud using Isolation Forest + SVM
    
    Returns fraud detection results with anomaly score
    """
    
    try:
        models = get_ml_models()
        
        prediction = models.detect_fraud(
            request.application_data,
            request.extracted_documents
        )
        
        if "error" in prediction:
            raise HTTPException(status_code=500, detail=prediction["error"])
        
        return {
            "prediction": prediction,
            "model_used": "isolation_forest_svm",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in fraud detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/programs")
async def match_economic_programs(request: PredictionRequest):
    """
    Match applicant to economic enablement programs using K-Means + Logistic Regression
    
    Returns ranked program recommendations
    """
    
    try:
        models = get_ml_models()
        
        prediction = models.match_economic_programs(
            request.application_data,
            request.extracted_documents
        )
        
        if "error" in prediction:
            raise HTTPException(status_code=500, detail=prediction["error"])
        
        return {
            "prediction": prediction,
            "model_used": "kmeans_logistic_regression",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in program matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/comprehensive")
async def comprehensive_prediction(request: PredictionRequest):
    """
    Run all ML predictions in one endpoint
    
    Returns comprehensive assessment using all ML models
    """
    
    try:
        models = get_ml_models()
        
        # Run all predictions
        eligibility = models.predict_eligibility(
            request.application_data, request.extracted_documents
        )
        risk = models.predict_risk_level(
            request.application_data, request.extracted_documents
        )
        support = models.predict_support_amount(
            request.application_data, request.extracted_documents
        )
        fraud = models.detect_fraud(
            request.application_data, request.extracted_documents
        )
        programs = models.match_economic_programs(
            request.application_data, request.extracted_documents
        )
        
        # Combine results
        comprehensive_result = {
            "eligibility": eligibility,
            "risk_assessment": risk,
            "support_amount": support,
            "fraud_detection": fraud,
            "program_matching": programs,
            "overall_assessment": {
                "eligible": eligibility.get("eligible", False),
                "risk_level": risk.get("risk_level", "unknown"),
                "estimated_support": support.get("estimated_amount", 0),
                "fraud_risk": fraud.get("risk_level", "unknown"),
                "recommended_program": programs.get("top_program", "none")
            }
        }
        
        return {
            "comprehensive_prediction": comprehensive_result,
            "models_used": [
                "random_forest_eligibility",
                "gradient_boosting_risk",
                "random_forest_support",
                "isolation_forest_svm_fraud",
                "kmeans_logistic_programs"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def train_models_background(n_samples: int, force_retrain: bool = False):
    """Background task for training ML models"""
    
    try:
        logger.info(f"Starting background ML training with {n_samples} samples")
        
        # Generate training data
        data_generator = SyntheticDataGenerator()
        applications = data_generator.generate_application_data(count=n_samples)
        training_data = pd.DataFrame(applications)
        
        # Add required features
        training_data['per_capita_income'] = training_data['monthly_income'] / training_data['family_size']
        training_data['debt_to_income_ratio'] = training_data['debt_amount'] / training_data['monthly_income'].clip(lower=1)
        training_data['total_assets'] = training_data['debt_amount'] * 2  # Simplified
        training_data['total_liabilities'] = training_data['debt_amount']
        
        # Encode categoricals
        employment_mapping = {'employed': 3, 'self_employed': 2, 'unemployed': 1, 'student': 1, 'retired': 2}
        housing_mapping = {'owned': 3, 'family_house': 2, 'rented': 1, 'shared': 0}
        education_mapping = {'university': 3, 'college': 2, 'secondary': 1, 'primary': 0, 'no_education': 0}
        
        training_data['employment_status'] = training_data['employment_status'].map(employment_mapping).fillna(1)
        training_data['housing_type'] = training_data['housing_type'].map(housing_mapping).fillna(1)
        training_data['education_level'] = training_data['education_level'].map(education_mapping).fillna(1)
        
        # Initialize and train models
        global ml_models
        ml_models = SocialSupportMLModels()
        
        training_results = ml_models.train_models(training_data)
        
        if training_results['status'] == 'success':
            logger.info(f"✅ Background ML training completed successfully!")
            logger.info(f"Models trained: {training_results['models_trained']}")
        else:
            logger.error(f"❌ Background ML training failed: {training_results.get('error')}")
            
    except Exception as e:
        logger.error(f"❌ Background ML training failed: {e}")

@router.post("/train")
async def train_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Train ML models with synthetic data
    
    Trains all scikit-learn models in the background
    """
    
    try:
        # Check if models exist and force_retrain is False
        models = get_ml_models()
        model_info = models.get_model_info()
        
        trained_models = [name for name, info in model_info.items() if info['trained']]
        
        if trained_models and not request.force_retrain:
            return {
                "message": "Models already trained. Use force_retrain=true to retrain.",
                "trained_models": trained_models,
                "total_models": len(model_info),
                "training_samples": request.n_samples
            }
        
        # Start background training
        background_tasks.add_task(
            train_models_background,
            request.n_samples,
            request.force_retrain
        )
        
        return {
            "message": "Model training started in background",
            "training_samples": request.n_samples,
            "force_retrain": request.force_retrain,
            "estimated_time_minutes": max(2, request.n_samples // 500),
            "status": "training_initiated"
        }
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    """List all available ML models and their status"""
    
    try:
        models = get_ml_models()
        model_info = models.get_model_info()
        
        model_list = []
        for name, info in model_info.items():
            model_list.append({
                "name": name,
                "type": info['type'],
                "features": info['features'],
                "feature_count": len(info['features']),
                "trained": info['trained'],
                "description": _get_model_description(name)
            })
        
        return {
            "models": model_list,
            "total_models": len(model_list),
            "trained_models": len([m for m in model_list if m['trained']])
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate")
async def evaluate_models():
    """Evaluate trained models with test data"""
    
    try:
        models = get_ml_models()
        
        # Generate test data
        data_generator = SyntheticDataGenerator()
        test_applications = data_generator.generate_application_data(count=100)
        
        # Run predictions on test data
        results = []
        for app in test_applications[:10]:  # Test first 10
            prediction = models.predict_eligibility(app, {})
            if "error" not in prediction:
                results.append({
                    "income": app["monthly_income"],
                    "family_size": app["family_size"],
                    "predicted_eligible": prediction["eligible"],
                    "confidence": prediction["confidence"]
                })
        
        return {
            "evaluation_results": results,
            "test_samples": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _get_model_description(model_name: str) -> str:
    """Get description for each model"""
    
    descriptions = {
        "eligibility": "Random Forest classifier for eligibility assessment",
        "risk": "Gradient Boosting classifier for risk level prediction",
        "support_amount": "Multi-class Random Forest for support amount prediction",
        "fraud_anomaly": "Isolation Forest for anomaly detection in applications",
        "fraud_classifier": "SVM classifier for fraud detection",
        "applicant_clustering": "K-Means clustering for applicant segmentation",
        "program_matcher": "Logistic Regression for economic program matching"
    }
    
    return descriptions.get(model_name, "Machine learning model for social support assessment") 