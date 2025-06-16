"""
Simplified ML Models for Social Support AI Workflow

This module provides simplified machine learning models for:
1. Eligibility Classification (Random Forest)
2. Support Amount Prediction (Random Forest Regressor)

The models use only essential features that are consistently available from user input.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from typing import Dict, Any, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SocialSupportMLModels:
    """
    ML Models for Social Support Eligibility Assessment
    
    Implements scikit-learn Random Forest models for eligibility classification
    and support amount prediction. Uses singleton pattern for efficient model
    loading and provides automatic training with synthetic data generation.
    """
    
    _instance = None  # Singleton pattern
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Define simplified feature sets
        self.feature_sets = {
            'eligibility': ['monthly_income', 'family_size', 'employment_status'],
            'support_amount': ['monthly_income', 'family_size', 'employment_status', 'housing_status']
        }
        
        # Model parameters
        self.model_params = {
            'eligibility': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'support_amount': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        }
        
        # Load or create models
        self._load_models_once()
        self._initialized = True
    
    def _load_models_once(self):
        """Load models once using singleton pattern"""
        
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Try to load existing models
        models_loaded = self._load_models()
        
        if not models_loaded:
            logger.info("No pre-trained models found, will create minimal training on first use")
    
    def _load_models(self) -> bool:
        """Load pre-trained models from disk"""
        
        try:
            logger.info("Attempting to load models from fixed file paths...")
            
            model_files = {
                'eligibility': 'models/eligibility_model.joblib',
                'support_amount': 'models/support_amount_model.joblib'
            }
            
            scaler_files = {
                'eligibility': 'models/eligibility_scaler.joblib',
                'support_amount': 'models/support_amount_scaler.joblib'
            }
            
            models_loaded = 0
            
            # Load models
            for model_name, file_path in model_files.items():
                if os.path.exists(file_path):
                    self.models[model_name] = joblib.load(file_path)
                    logger.debug(f"Loaded model '{model_name}' from {file_path}")
                    models_loaded += 1
            else:
                    logger.warning(f"Model file not found for '{model_name}': {file_path}")
            
            # Load scalers
            for scaler_name, file_path in scaler_files.items():
                if os.path.exists(file_path):
                    self.scalers[scaler_name] = joblib.load(file_path)
                    logger.debug(f"Loaded scaler '{scaler_name}' from {file_path}")
                else:
                    logger.info(f"Scaler file not found for '{scaler_name}': {file_path}")
            
            if models_loaded == 0:
                logger.warning("No models were found or loaded from fixed paths.")
                return False
            
            logger.info(f"Successfully loaded {models_loaded} models")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def _ensure_models_ready(self):
        """Ensure models are trained and ready"""
        
        if not self.models:
            logger.info("Models not fitted, creating minimal training data...")
            self._create_minimal_training()
    
    def _create_minimal_training(self):
        """Create minimal synthetic training data for simplified models"""
        
        logger.info("Creating minimal synthetic training data for simplified models...")
        
        # Clear any existing model files for fresh training
        model_files = [
            'models/eligibility_model.joblib',
            'models/support_amount_model.joblib',
            'models/eligibility_scaler.joblib', 
            'models/support_amount_scaler.joblib'
        ]
        
        for file_path in model_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        logger.info("🗑️ Cleared existing model files for fresh training")
        
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 200
        
        # Generate features
        monthly_income = np.random.uniform(500, 8000, n_samples)
        family_size = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05])
        employment_status = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])  # unemployed, employed, self_employed, retired
        housing_status = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])  # rent, owned, family
        
        # Create DataFrame
        data = pd.DataFrame({
            'monthly_income': monthly_income,
            'family_size': family_size,
            'employment_status': employment_status,
            'housing_status': housing_status
        })
        
        logger.info(f"Generated {n_samples} training samples with features: {list(data.columns)}")
        
        # Generate targets
        eligibility_targets, support_targets = self._prepare_simplified_training_targets(data)
        
        # Train models
        logger.info("Training eligibility model with 3 features...")
        self._train_single_model('eligibility', data[self.feature_sets['eligibility']], eligibility_targets)
        
        logger.info("Training support_amount model with 4 features...")
        self._train_single_model('support_amount', data[self.feature_sets['support_amount']], support_targets)
        
        # Save models
        self.save_models()
        
        logger.success("✅ Minimal training completed successfully!")
        logger.info(f"Eligibility accuracy: {getattr(self, '_last_eligibility_accuracy', 0.95)}")
        logger.info(f"Support amount R²: {getattr(self, '_last_support_r2', 0.95)}")
    
    def _prepare_simplified_training_targets(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic targets for training"""
        
        # Eligibility logic: income per person < 3000 AED = eligible
        income_per_person = data['monthly_income'] / data['family_size']
        eligibility = (income_per_person < 3000).astype(int)
        
        # Support amount logic: base + family supplement + income gap
        base_support = 500
        family_supplement = (data['family_size'] - 1) * 200
        income_gap = np.maximum(0, (3000 * data['family_size'] - data['monthly_income']) * 0.6)
        
        support_amounts = base_support + family_supplement + income_gap
        support_amounts = np.clip(support_amounts, 500, 5000)  # Min 500, Max 5000 AED
        
        # Only eligible people get support amounts
        support_amounts = support_amounts * eligibility
        
        logger.info(f"Created targets - Eligibility: {eligibility.sum()}/{len(eligibility)} eligible")
        logger.info(f"Support amounts range: {support_amounts[support_amounts > 0].min():.0f} - {support_amounts.max():.0f} AED")
        
        return eligibility, support_amounts
    
    def _train_single_model(self, model_name: str, X: pd.DataFrame, y: np.ndarray):
        """Train a single model"""
        
        logger.info(f"Training {model_name} model...")
        
        # Convert to numpy array
        X_array = X.values.astype(float)
        
        logger.info(f"Training data shape: {X_array.shape}, Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_name == 'eligibility':
            model = RandomForestClassifier(**self.model_params[model_name])
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            self._last_eligibility_accuracy = accuracy
            
            logger.info(f"✅ {model_name} model trained successfully. Accuracy: {accuracy:.3f}")
            
        else:  # support_amount
            model = RandomForestRegressor(**self.model_params[model_name])
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            self._last_support_r2 = r2
            
            logger.info(f"✅ {model_name} model trained successfully. R²: {r2:.3f}, MAE: {mae:.0f}")
        
        # Store model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
    
    def prepare_features(self, application_data: Dict, extracted_documents: Dict, model_type: str) -> np.ndarray:
        """Prepare features for prediction"""
        
        features = []
        feature_names = self.feature_sets[model_type]
        
        for feature in feature_names:
            if feature == 'monthly_income':
                value = float(application_data.get('monthly_income', 0))
            elif feature == 'family_size':
                value = float(application_data.get('family_size', 1))
            elif feature == 'employment_status':
                # Encode: unemployed=0, employed=1, self_employed=2, retired=3
                status = application_data.get('employment_status', 'unemployed')
                status_map = {'unemployed': 0, 'employed': 1, 'self_employed': 2, 'retired': 3}
                value = float(status_map.get(status, 0))
            elif feature == 'housing_status':
                # Encode: rent=0, owned=1, family=2
                status = application_data.get('housing_status', 'rent')
                status_map = {'rent': 0, 'rented': 0, 'owned': 1, 'own': 1, 'family': 2}
                value = float(status_map.get(status, 0))
            else:
                value = 0.0
            
            features.append(value)
        
        features_array = np.array(features).reshape(1, -1)
        logger.debug(f"Prepared {len(features)} features for {model_type}: {features}")
        
        return features_array
    
    def predict_eligibility(self, application_data: Dict, extracted_documents: Dict = None) -> Dict[str, Any]:
        """Predict eligibility using simplified model"""
        
        self._ensure_models_ready()
        
        if 'eligibility' not in self.models:
            raise ValueError("Eligibility model not available")
        
        # Prepare features
        features = self.prepare_features(application_data, extracted_documents or {}, 'eligibility')
        
        # Scale features
        if 'eligibility' in self.scalers:
            features_scaled = self.scalers['eligibility'].transform(features)
        else:
            features_scaled = features
        
        # Predict
        model = self.models['eligibility']
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0].max()
        
        # Generate reasoning
        monthly_income = application_data.get('monthly_income', 0)
        family_size = application_data.get('family_size', 1)
        employment_status = application_data.get('employment_status', 'unknown')
        
        income_per_person = monthly_income / family_size if family_size > 0 else monthly_income
        
        reasoning = f"Eligible based on income per person ({income_per_person:.0f} AED), family size ({family_size}), and employment status ({employment_status})"
        
        eligible = bool(prediction)
        
        logger.info(f"✅ Eligibility prediction: {eligible} (confidence: {confidence:.3f})")
        
        return {
            'eligible': eligible,
            'confidence': confidence,
            'eligibility_score': confidence,
            'reasoning': reasoning
        }
    
    def predict_support_amount(self, application_data: Dict, extracted_documents: Dict = None) -> Dict[str, Any]:
        """Predict support amount using simplified model"""
        
        self._ensure_models_ready()
        
        if 'support_amount' not in self.models:
            raise ValueError("Support amount model not available")
        
        # Prepare features
        features = self.prepare_features(application_data, extracted_documents or {}, 'support_amount')
        
        # Scale features
        if 'support_amount' in self.scalers:
            features_scaled = self.scalers['support_amount'].transform(features)
        else:
            features_scaled = features
        
        # Predict
        model = self.models['support_amount']
        predicted_amount = model.predict(features_scaled)[0]
        
        # Ensure minimum amount and round
        predicted_amount = max(500, round(predicted_amount))
        
        # Determine amount range
        if predicted_amount < 1000:
            amount_range = "Basic Support (500-999 AED)"
        elif predicted_amount < 2500:
            amount_range = "Standard Support (1000-2499 AED)"
        elif predicted_amount < 4000:
            amount_range = "Enhanced Support (2500-3999 AED)"
        else:
            amount_range = "Maximum Support (4000+ AED)"
        
        # Generate reasoning
        family_size = application_data.get('family_size', 1)
        employment_status = application_data.get('employment_status', 'unknown')
        housing_status = application_data.get('housing_status', 'unknown')
        
        reasoning = f"Amount calculated based on family size ({family_size} people), employment status ({employment_status}), and housing situation"
        
        logger.info(f"✅ Support amount prediction: {predicted_amount:.0f} AED")
        
        return {
            'predicted_amount': predicted_amount,
            'amount_range': amount_range,
            'reasoning': reasoning
        }
    
    def save_models(self):
        """Save all models and scalers to disk"""
            
        logger.info("Saving models...")

        os.makedirs("models", exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            file_path = f"models/{model_name}_model.joblib"
            joblib.dump(model, file_path)
            logger.debug(f"Saved model '{model_name}' to {file_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            file_path = f"models/{scaler_name}_scaler.joblib"
            joblib.dump(scaler, file_path)
            logger.debug(f"Saved scaler for '{scaler_name}' to {file_path}")
        
        logger.success("All models and scalers have been saved.")

    def force_retrain_models(self) -> Dict[str, Any]:
        """Force retrain all models with fresh data"""
        
        logger.info("🔄 Force retraining models...")
        
        # Clear existing models
        self.models.clear()
        self.scalers.clear()
        
        # Retrain
        self._create_minimal_training()

        return {
            "status": "success",
            "message": "Models retrained successfully",
            "models_trained": list(self.models.keys())
        }