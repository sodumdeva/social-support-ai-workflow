#!/usr/bin/env python3
"""
Training Script for Social Support ML Models

Trains scikit-learn classification models using synthetic data:
1. Generates comprehensive training dataset
2. Trains all ML models (Random Forest, Gradient Boosting, SVM, etc.)
3. Evaluates model performance
4. Saves trained models for production use

Usage:
    python scripts/train_ml_models.py [--samples 1000] [--retrain]
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.ml_models import SocialSupportMLModels
from src.data.synthetic_data import SyntheticDataGenerator
from config import get_model_path
from loguru import logger


def generate_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate comprehensive training dataset for ML models"""
    
    logger.info(f"Generating {n_samples} synthetic training samples...")
    
    # Initialize synthetic data generator
    data_generator = SyntheticDataGenerator()
    
    # Generate application data
    applications = data_generator.generate_application_data(count=n_samples)
    
    # Convert to DataFrame
    df = pd.DataFrame(applications)
    
    # Add calculated features
    df['per_capita_income'] = df['monthly_income'] / df['family_size']
    df['debt_to_income_ratio'] = df['debt_amount'] / np.maximum(df['monthly_income'], 1)
    df['housing_cost_ratio'] = df['monthly_rent'] / np.maximum(df['monthly_income'], 1)
    
    # Add assets and liabilities (synthetic)
    np.random.seed(42)
    df['total_assets'] = np.random.lognormal(mean=8, sigma=1.5, size=len(df))  # Assets
    df['total_liabilities'] = df['debt_amount']  # Same as debt amount
    
    # Encode categorical variables numerically BEFORE other operations
    employment_mapping = {'employed': 3, 'self_employed': 2, 'unemployed': 1, 'student': 1, 'retired': 2}
    housing_mapping = {'owned': 3, 'family_house': 2, 'rented': 1, 'shared': 0}
    education_mapping = {'university': 3, 'college': 2, 'secondary': 1, 'primary': 0, 'no_education': 0}
    
    # Create numeric columns directly
    df['employment_status'] = df['employment_status'].map(employment_mapping).fillna(1).astype(int)
    df['housing_type'] = df['housing_type'].map(housing_mapping).fillna(1).astype(int) 
    df['education_level'] = df['education_level'].map(education_mapping).fillna(1).astype(int)
    
    # Ensure all required columns exist with proper data types
    required_columns = [
        'monthly_income', 'family_size', 'number_of_dependents',
        'employment_duration_months', 'total_assets', 'total_liabilities',
        'monthly_rent', 'credit_score', 'employment_status',
        'housing_type', 'has_medical_conditions', 'debt_to_income_ratio',
        'previous_applications', 'has_criminal_record', 'education_level'
    ]
    
    # Add missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            if col.startswith('has_'):
                df[col] = 0  # Convert boolean to int
            else:
                df[col] = 0
    
    # Convert boolean columns to integers
    df['has_medical_conditions'] = df['has_medical_conditions'].astype(int)
    df['has_criminal_record'] = df['has_criminal_record'].astype(int)
    
    # Ensure all numeric columns are proper numeric types
    numeric_columns = [col for col in required_columns if not col.startswith('has_')]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    logger.info(f"Generated training dataset with {len(df)} samples and {len(df.columns)} features")
    
    return df


def evaluate_model_performance(ml_models: SocialSupportMLModels, test_data: pd.DataFrame) -> dict:
    """Evaluate trained models on test data"""
    
    logger.info("Evaluating model performance...")
    
    results = {}
    
    # Test eligibility model
    if hasattr(ml_models.models['eligibility'], 'feature_importances_'):
        # Get feature importance for eligibility model
        feature_importance = dict(zip(
            ml_models.feature_columns['eligibility'],
            ml_models.models['eligibility'].feature_importances_
        ))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        results['eligibility'] = {
            'model_type': 'RandomForestClassifier',
            'n_features': len(ml_models.feature_columns['eligibility']),
            'top_features': sorted_features[:5],
            'feature_importance': feature_importance
        }
    
    # Test a few predictions
    sample_applications = test_data.head(10)
    predictions = []
    
    for idx, row in sample_applications.iterrows():
        app_data = row.to_dict()
        
        # Predict eligibility
        eligibility = ml_models.predict_eligibility(app_data, {})
        risk = ml_models.predict_risk_level(app_data, {})
        support = ml_models.predict_support_amount(app_data, {})
        fraud = ml_models.detect_fraud(app_data, {})
        
        predictions.append({
            'income': app_data['monthly_income'],
            'family_size': app_data['family_size'],
            'eligible': eligibility.get('eligible', False),
            'confidence': eligibility.get('confidence', 0),
            'risk_level': risk.get('risk_level', 'unknown'),
            'support_amount': support.get('estimated_amount', 0),
            'fraud_risk': fraud.get('risk_level', 'unknown')
        })
    
    results['sample_predictions'] = predictions
    
    return results


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train Social Support ML Models')
    parser.add_argument('--samples', type=int, default=1000, 
                       help='Number of training samples to generate')
    parser.add_argument('--retrain', action='store_true',
                       help='Force retraining even if models exist')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model performance after training')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting ML Model Training for Social Support System")
    logger.info(f"Training samples: {args.samples}")
    
    try:
        # Initialize ML models
        ml_models = SocialSupportMLModels()
        
        # Check if models already exist
        model_path = get_model_path()
        existing_models = [f for f in os.listdir(model_path) if f.endswith('.joblib')] if os.path.exists(model_path) else []
        
        if existing_models and not args.retrain:
            logger.info(f"Found {len(existing_models)} existing model files. Use --retrain to force retraining.")
            
            # Try loading existing models
            try:
                ml_models.load_models()
                logger.info("✅ Successfully loaded existing models")
                
                if args.evaluate:
                    # Generate test data for evaluation
                    test_data = generate_training_data(n_samples=200)
                    results = evaluate_model_performance(ml_models, test_data)
                    
                    logger.info("📊 Model Performance Summary:")
                    logger.info(f"Eligibility Model: {results['eligibility']['model_type']}")
                    logger.info(f"Top Features: {[f[0] for f in results['eligibility']['top_features']]}")
                    
                return
                
            except Exception as e:
                logger.warning(f"Could not load existing models: {e}")
                logger.info("Proceeding with retraining...")
        
        # Generate training data
        training_data = generate_training_data(n_samples=args.samples)
        
        # Train models
        logger.info("🤖 Training ML models...")
        training_results = ml_models.train_models(training_data)
        
        if training_results['status'] == 'success':
            logger.info("✅ Model training completed successfully!")
            logger.info(f"Models trained: {training_results['models_trained']}")
            
            # Print training results
            for model_name, metrics in training_results['training_results'].items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    logger.info(f"{model_name}: Accuracy={metrics['accuracy']:.3f}, CV={metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f}")
        
        # Evaluate performance
        if args.evaluate:
            test_data = generate_training_data(n_samples=200)
            results = evaluate_model_performance(ml_models, test_data)
            
            logger.info("\n📊 Model Performance Summary:")
            logger.info(f"Eligibility Model: {results['eligibility']['model_type']}")
            logger.info(f"Number of features: {results['eligibility']['n_features']}")
            
            logger.info("\n🔝 Top 5 Most Important Features:")
            for feature, importance in results['eligibility']['top_features']:
                logger.info(f"  {feature}: {importance:.3f}")
            
            logger.info("\n📝 Sample Predictions:")
            for i, pred in enumerate(results['sample_predictions'][:5]):
                logger.info(f"  Sample {i+1}: Income={pred['income']}, Family={pred['family_size']}, "
                          f"Eligible={pred['eligible']}, Support={pred['support_amount']}")
        
        # Model info
        model_info = ml_models.get_model_info()
        logger.info("\n📋 Trained Models:")
        for model_name, info in model_info.items():
            logger.info(f"  {model_name}: {info['type']} ({'✅ Trained' if info['trained'] else '❌ Not Trained'})")
        
        logger.info(f"\n💾 Models saved to: {model_path}")
        logger.info("🎉 Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 