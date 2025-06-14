"""
Machine Learning Models for Social Support AI Workflow

Implements scikit-learn classification models for:
1. Eligibility Classification (Random Forest)
2. Risk Assessment (Gradient Boosting)
3. Support Amount Prediction (Multi-class Classification)
4. Fraud Detection (Isolation Forest + SVM)
5. Economic Program Matching (K-Means + Logistic Regression)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from loguru import logger

from config import get_model_path


class SocialSupportMLModels:
    """
    Ensemble of ML models for social support application processing
    
    Model Selection Justification:
    - Random Forest: Handles mixed data types, provides feature importance, robust to outliers
    - Gradient Boosting: High accuracy for eligibility decisions, handles imbalanced data
    - SVM: Effective for fraud detection with non-linear patterns
    - Isolation Forest: Anomaly detection for fraud/unusual applications
    - K-Means + Logistic: Economic program matching based on applicant clustering
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        self.model_path = get_model_path()
        
        # Initialize models with justified hyperparameters
        self._initialize_models()
        
        # Try to load existing models, if none exist, create minimal training
        self._ensure_models_ready()
        
    def _initialize_models(self):
        """Initialize all ML models with optimized hyperparameters"""
        
        # 1. ELIGIBILITY CLASSIFIER - Random Forest
        # Justification: Handles mixed data types, provides interpretability via feature importance
        self.models['eligibility'] = RandomForestClassifier(
            n_estimators=100,           # Good balance of accuracy vs speed
            max_depth=10,               # Prevents overfitting on small datasets
            min_samples_split=5,        # Robust to noise
            min_samples_leaf=2,         # Ensures stable predictions
            random_state=42,
            class_weight='balanced'     # Handles imbalanced eligible/not eligible
        )
        
        # 2. RISK ASSESSMENT - Gradient Boosting
        # Justification: Excellent for risk prediction, handles feature interactions
        self.models['risk'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # 3. SUPPORT AMOUNT PREDICTION - Multi-class Classification
        # Justification: Predicts support brackets (0-1000, 1000-3000, 3000+)
        self.models['support_amount'] = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            class_weight='balanced'
        )
        
        # 4. FRAUD DETECTION - Isolation Forest + SVM
        # Justification: Isolation Forest for anomaly detection, SVM for classification
        self.models['fraud_anomaly'] = IsolationForest(
            contamination=0.1,          # Expect 10% anomalies
            random_state=42
        )
        
        self.models['fraud_classifier'] = SVC(
            kernel='rbf',               # Non-linear patterns in fraud
            C=1.0,
            gamma='scale',
            probability=True,           # For probability estimates
            random_state=42
        )
        
        # 5. ECONOMIC PROGRAM MATCHING - K-Means + Logistic Regression
        # Justification: Cluster applicants by profile, then classify best programs
        self.models['applicant_clustering'] = KMeans(
            n_clusters=5,               # 5 applicant archetypes
            random_state=42
        )
        
        self.models['program_matcher'] = LogisticRegression(
            multi_class='multinomial',   # Multiple program types
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        
        # Initialize preprocessing components
        self._initialize_preprocessors()
    
    def _initialize_preprocessors(self):
        """Initialize data preprocessing components"""
        
        # Feature definitions for each model
        self.feature_columns = {
            'eligibility': [
                'monthly_income', 'family_size', 'number_of_dependents',
                'employment_duration_months', 'total_assets', 'total_liabilities',
                'monthly_rent', 'credit_score', 'employment_status',
                'housing_type', 'has_medical_conditions'
            ],
            'risk': [
                'monthly_income', 'credit_score', 'debt_to_income_ratio',
                'employment_duration_months', 'previous_applications',
                'has_criminal_record', 'employment_status'
            ],
            'support_amount': [
                'monthly_income', 'family_size', 'number_of_dependents',
                'monthly_rent', 'total_assets', 'employment_status',
                'housing_type', 'has_medical_conditions'
            ],
            'fraud': [
                'monthly_income', 'family_size', 'total_assets',
                'employment_duration_months', 'previous_applications',
                'credit_score', 'debt_to_income_ratio'
            ],
            'program_matching': [
                'monthly_income', 'family_size', 'employment_status',
                'education_level', 'employment_duration_months',
                'has_medical_conditions', 'number_of_dependents'
            ]
        }
        
        # Initialize scalers for each model
        for model_name in self.feature_columns:
            self.scalers[model_name] = StandardScaler()
            self.encoders[model_name] = {}
    
    def prepare_features(self, application_data: Dict[str, Any], 
                        extracted_documents: Dict[str, Any], 
                        model_type: str) -> np.ndarray:
        """
        Prepare features for ML model prediction
        
        Args:
            application_data: Basic application information
            extracted_documents: Data extracted from documents
            model_type: Type of model ('eligibility', 'risk', etc.)
            
        Returns:
            Feature array ready for model prediction
        """
        
        # Extract features based on model type
        features = {}
        
        # Basic application features
        features['monthly_income'] = application_data.get('monthly_income', 0)
        features['family_size'] = application_data.get('family_size', 1)
        features['number_of_dependents'] = application_data.get('number_of_dependents', 0)
        features['employment_duration_months'] = application_data.get('employment_duration_months', 0)
        features['monthly_rent'] = application_data.get('monthly_rent', 0)
        features['credit_score'] = application_data.get('credit_score', 500)
        features['previous_applications'] = application_data.get('previous_applications', 0)
        features['has_medical_conditions'] = int(application_data.get('has_medical_conditions', False))
        features['has_criminal_record'] = int(application_data.get('has_criminal_record', False))
        
        # Enhanced features from extracted documents
        bank_data = extracted_documents.get('bank_statement', {}).get('structured_data', {})
        assets_data = extracted_documents.get('assets', {}).get('structured_data', {})
        
        # Financial features
        if bank_data:
            features['monthly_income'] = max(features['monthly_income'], 
                                           bank_data.get('monthly_income', 0))
        
        if assets_data:
            features['total_assets'] = assets_data.get('total_assets', 0)
            features['total_liabilities'] = assets_data.get('total_liabilities', 0)
        else:
            features['total_assets'] = 0
            features['total_liabilities'] = 0
        
        # Calculated features
        features['debt_to_income_ratio'] = (
            features['total_liabilities'] / max(features['monthly_income'], 1)
        )
        
        # Categorical features
        employment_status = application_data.get('employment_status', 'unemployed')
        housing_type = application_data.get('housing_type', 'unknown')
        education_level = application_data.get('education_level', 'unknown')
        
        # Encode categorical variables
        employment_mapping = {'employed': 3, 'self_employed': 2, 'unemployed': 1, 'student': 1, 'retired': 2}
        housing_mapping = {'owned': 3, 'family_house': 2, 'rented': 1, 'shared': 0}
        education_mapping = {'university': 3, 'college': 2, 'secondary': 1, 'primary': 0, 'no_education': 0}
        
        features['employment_status'] = employment_mapping.get(employment_status, 1)
        features['housing_type'] = housing_mapping.get(housing_type, 1)
        features['education_level'] = education_mapping.get(education_level, 1)
        
        # Select features for specific model
        model_features = []
        for feature_name in self.feature_columns[model_type]:
            model_features.append(features.get(feature_name, 0))
        
        return np.array(model_features).reshape(1, -1)
    
    def predict_eligibility(self, application_data: Dict[str, Any], 
                          extracted_documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict eligibility using Random Forest classifier
        
        Returns:
            Dictionary with eligibility prediction and confidence
        """
        
        if 'eligibility' not in self.models:
            return {"error": "Eligibility model not trained"}
        
        try:
            # Prepare features
            features = self.prepare_features(application_data, extracted_documents, 'eligibility')
            
            # Scale features
            if hasattr(self.scalers['eligibility'], 'mean_'):
                features_scaled = self.scalers['eligibility'].transform(features)
            else:
                features_scaled = features
            
            # Predict
            prediction = self.models['eligibility'].predict(features_scaled)[0]
            probabilities = self.models['eligibility'].predict_proba(features_scaled)[0]
            
            # Get feature importance
            feature_importance = dict(zip(
                self.feature_columns['eligibility'],
                self.models['eligibility'].feature_importances_
            ))
            
            return {
                'eligible': bool(prediction),
                'confidence': float(max(probabilities)),
                'probability_eligible': float(probabilities[1] if len(probabilities) > 1 else probabilities[0]),
                'feature_importance': feature_importance,
                'model_type': 'random_forest',
                'prediction_method': 'ml_classification'
            }
            
        except Exception as e:
            logger.error(f"Error in eligibility prediction: {e}")
            return {"error": str(e)}
    
    def predict_risk_level(self, application_data: Dict[str, Any], 
                          extracted_documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict risk level using Gradient Boosting classifier
        
        Returns:
            Dictionary with risk level prediction (low, medium, high)
        """
        
        if 'risk' not in self.models:
            return {"error": "Risk model not trained"}
        
        try:
            features = self.prepare_features(application_data, extracted_documents, 'risk')
            
            if hasattr(self.scalers['risk'], 'mean_'):
                features_scaled = self.scalers['risk'].transform(features)
            else:
                features_scaled = features
            
            prediction = self.models['risk'].predict(features_scaled)[0]
            probabilities = self.models['risk'].predict_proba(features_scaled)[0]
            
            risk_levels = ['low', 'medium', 'high']
            
            return {
                'risk_level': risk_levels[prediction] if prediction < len(risk_levels) else 'medium',
                'confidence': float(max(probabilities)),
                'risk_probabilities': {
                    level: float(prob) for level, prob in zip(risk_levels, probabilities)
                },
                'model_type': 'gradient_boosting'
            }
            
        except Exception as e:
            logger.error(f"Error in risk prediction: {e}")
            return {"error": str(e)}
    
    def predict_support_amount(self, application_data: Dict[str, Any], 
                             extracted_documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict support amount bracket using Random Forest classifier
        
        Returns:
            Dictionary with predicted support amount bracket
        """
        
        if 'support_amount' not in self.models:
            return {"error": "Support amount model not trained"}
        
        try:
            features = self.prepare_features(application_data, extracted_documents, 'support_amount')
            
            if hasattr(self.scalers['support_amount'], 'mean_'):
                features_scaled = self.scalers['support_amount'].transform(features)
            else:
                features_scaled = features
            
            prediction = self.models['support_amount'].predict(features_scaled)[0]
            probabilities = self.models['support_amount'].predict_proba(features_scaled)[0]
            
            # Support amount brackets
            support_brackets = {
                0: {'range': '0-1000', 'amount': 500},
                1: {'range': '1000-3000', 'amount': 2000},
                2: {'range': '3000-5000', 'amount': 4000},
                3: {'range': '5000+', 'amount': 6000}
            }
            
            predicted_bracket = support_brackets.get(prediction, support_brackets[1])
            
            return {
                'support_bracket': predicted_bracket['range'],
                'estimated_amount': predicted_bracket['amount'],
                'confidence': float(max(probabilities)),
                'model_type': 'random_forest_multiclass'
            }
            
        except Exception as e:
            logger.error(f"Error in support amount prediction: {e}")
            return {"error": str(e)}
    
    def detect_fraud(self, application_data: Dict[str, Any], 
                    extracted_documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect potential fraud using Isolation Forest + SVM
        
        Returns:
            Dictionary with fraud detection results
        """
        
        # Check if models exist
        if 'fraud_anomaly' not in self.models or 'fraud_classifier' not in self.models:
            logger.warning("Fraud detection models not trained, using rule-based fallback")
            return self._fallback_fraud_detection(application_data, extracted_documents)
        
        try:
            features = self.prepare_features(application_data, extracted_documents, 'fraud')
            
            # Check if IsolationForest is fitted - use a more robust check
            try:
                # Try to access the fitted attributes - this will raise an exception if not fitted
                _ = self.models['fraud_anomaly'].decision_function(features)
                
                # If we get here, the model is fitted
                anomaly_score = self.models['fraud_anomaly'].decision_function(features)[0]
                is_anomaly = self.models['fraud_anomaly'].predict(features)[0] == -1
                
            except Exception as e:
                logger.warning(f"IsolationForest not fitted or prediction failed: {e}, using rule-based fallback")
                return self._fallback_fraud_detection(application_data, extracted_documents)
            
            # Fraud classification
            try:
                if hasattr(self.scalers['fraud'], 'mean_'):
                    features_scaled = self.scalers['fraud'].transform(features)
                else:
                    features_scaled = features
                
                fraud_probability = self.models['fraud_classifier'].predict_proba(features_scaled)[0]
                fraud_prob_value = float(fraud_probability[1] if len(fraud_probability) > 1 else fraud_probability[0])
            except Exception as e:
                logger.warning(f"Fraud classifier prediction failed: {e}, using anomaly score only")
                fraud_prob_value = 0.5 if is_anomaly else 0.1
            
            return {
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'fraud_probability': fraud_prob_value,
                'risk_level': 'high' if is_anomaly or fraud_prob_value > 0.7 else 'low',
                'model_type': 'isolation_forest_svm',
                'status': 'ml_prediction'
            }
            
        except Exception as e:
            logger.error(f"Error in fraud detection: {e}, using rule-based fallback")
            return self._fallback_fraud_detection(application_data, extracted_documents)
    
    def _fallback_fraud_detection(self, application_data: Dict[str, Any], 
                                 extracted_documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based fraud detection fallback when ML models are not available
        """
        try:
            # Extract key indicators
            monthly_income = application_data.get('monthly_income', 0)
            family_size = application_data.get('family_size', 1)
            employment_status = application_data.get('employment_status', 'unknown')
            
            # Rule-based fraud indicators
            fraud_indicators = []
            fraud_score = 0.0
            
            # Income anomalies
            if monthly_income > 15000:  # Very high income
                fraud_indicators.append("unusually_high_income")
                fraud_score += 0.3
            elif monthly_income == 0 and employment_status == "employed":
                fraud_indicators.append("income_employment_mismatch")
                fraud_score += 0.4
            
            # Family size anomalies
            if family_size > 12:  # Unusually large family
                fraud_indicators.append("unusually_large_family")
                fraud_score += 0.2
            elif family_size <= 0:
                fraud_indicators.append("invalid_family_size")
                fraud_score += 0.5
            
            # Document inconsistencies (if available)
            if extracted_documents:
                bank_data = extracted_documents.get('bank_statement', {})
                if bank_data and 'monthly_income' in bank_data:
                    bank_income = bank_data.get('monthly_income', 0)
                    if abs(bank_income - monthly_income) > monthly_income * 0.5:  # 50% difference
                        fraud_indicators.append("income_document_mismatch")
                        fraud_score += 0.4
            
            # Determine risk level
            if fraud_score >= 0.7:
                risk_level = 'high'
            elif fraud_score >= 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'is_anomaly': fraud_score >= 0.5,
                'anomaly_score': fraud_score,
                'fraud_probability': fraud_score,
                'risk_level': risk_level,
                'fraud_indicators': fraud_indicators,
                'model_type': 'rule_based_fallback',
                'status': 'fallback_prediction'
            }
            
        except Exception as e:
            logger.error(f"Error in fallback fraud detection: {e}")
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'fraud_probability': 0.0,
                'risk_level': 'unknown',
                'error': str(e),
                'model_type': 'fallback_error',
                'status': 'error'
            }
    
    def match_economic_programs(self, application_data: Dict[str, Any], 
                               extracted_documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match applicant to economic enablement programs using K-Means + Logistic Regression
        
        Returns:
            Dictionary with recommended programs
        """
        
        if 'applicant_clustering' not in self.models or 'program_matcher' not in self.models:
            return {"error": "Program matching models not trained"}
        
        try:
            features = self.prepare_features(application_data, extracted_documents, 'program_matching')
            
            # Cluster applicant
            cluster = self.models['applicant_clustering'].predict(features)[0]
            
            # Predict best programs
            program_probabilities = self.models['program_matcher'].predict_proba(features)[0]
            
            programs = ['upskilling', 'job_matching', 'entrepreneurship', 'education_support']
            
            # Rank programs by probability
            program_rankings = [
                {
                    'program': program,
                    'probability': float(prob),
                    'recommended': prob > 0.3
                }
                for program, prob in zip(programs, program_probabilities)
            ]
            
            program_rankings.sort(key=lambda x: x['probability'], reverse=True)
            
            return {
                'applicant_cluster': int(cluster),
                'program_recommendations': program_rankings,
                'top_program': program_rankings[0]['program'],
                'model_type': 'kmeans_logistic_regression'
            }
            
        except Exception as e:
            logger.error(f"Error in program matching: {e}")
            return {"error": str(e)}
    
    def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all ML models using synthetic or historical data
        
        Args:
            training_data: DataFrame with features and target variables
            
        Returns:
            Training results and model performance metrics
        """
        
        results = {}
        
        try:
            # Prepare different target variables
            targets = self._prepare_training_targets(training_data)
            
            # Train each model - handle fraud classifier target key properly
            model_target_mapping = {
                'eligibility': 'eligibility',
                'risk': 'risk', 
                'support_amount': 'support_amount',
                'fraud_classifier': 'fraud_classifier'  # This was the mismatch
            }
            
            for model_name, target_key in model_target_mapping.items():
                if target_key in targets:
                    try:
                        result = self._train_single_model(training_data, targets[target_key], model_name)
                        results[model_name] = result
                        logger.info(f"✅ Trained {model_name} successfully")
                    except Exception as e:
                        logger.error(f"❌ Failed to train {model_name}: {e}")
                        results[model_name] = {'status': 'failed', 'error': str(e)}
            
            # Train clustering model (unsupervised) - handle separately
            try:
                self._train_clustering_model(training_data)
                results['clustering'] = {'status': 'trained', 'n_clusters': 5}
                logger.info("✅ Trained clustering model successfully")
            except Exception as e:
                logger.error(f"❌ Failed to train clustering model: {e}")
                results['clustering'] = {'status': 'failed', 'error': str(e)}
            
            # Save models even if some failed
            trained_models = [name for name, result in results.items() 
                            if isinstance(result, dict) and result.get('status') != 'failed']
            
            if trained_models:
                self.save_models()
                logger.info(f"💾 Saved {len(trained_models)} trained models")
            else:
                logger.warning("⚠️ No models to save - all training failed")
            
            return {
                'status': 'success' if trained_models else 'partial_failure',
                'training_results': results,
                'models_trained': len(trained_models),
                'total_models': len(model_target_mapping) + 1  # +1 for clustering
            }
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _prepare_training_targets(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare target variables for training"""
        
        targets = {}
        
        # Eligibility target (based on income and family size)
        targets['eligibility'] = (
            (data['monthly_income'] / data['family_size'] <= 2000) | 
            (data['family_size'] >= 4)
        ).astype(int)
        
        # Risk target (based on credit score and employment)
        risk_conditions = [
            data['credit_score'] >= 700,
            (data['credit_score'] >= 600) & (data['credit_score'] < 700),
            data['credit_score'] < 600
        ]
        targets['risk'] = np.select(risk_conditions, [0, 1, 2], default=1)  # 0=low, 1=medium, 2=high
        
        # Support amount target (based on need level)
        support_conditions = [
            data['monthly_income'] >= 4000,
            (data['monthly_income'] >= 2000) & (data['monthly_income'] < 4000),
            (data['monthly_income'] >= 1000) & (data['monthly_income'] < 2000),
            data['monthly_income'] < 1000
        ]
        targets['support_amount'] = np.select(support_conditions, [0, 1, 2, 3], default=1)
        
        # Fraud target (synthetic - based on anomalous patterns)
        fraud_indicators = (
            (data['monthly_income'] > 10000) & (data['total_assets'] < 5000) |
            (data['previous_applications'] > 5) |
            (data['family_size'] > 10)
        )
        targets['fraud_classifier'] = fraud_indicators.astype(int)
        
        return targets
    
    def _train_single_model(self, data: pd.DataFrame, target: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Train a single model and return performance metrics"""
        
        # Prepare features
        feature_cols = self.feature_columns[model_name]
        X = data[feature_cols].fillna(0)
        y = target
        
        # Scale features
        X_scaled = self.scalers[model_name].fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.models[model_name].fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.models[model_name].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.models[model_name], X_scaled, y, cv=5)
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_features': len(feature_cols),
            'n_samples': len(X)
        }
    
    def _train_clustering_model(self, data: pd.DataFrame):
        """Train clustering model for applicant segmentation"""
        
        feature_cols = self.feature_columns['program_matching']
        X = data[feature_cols].fillna(0)
        X_scaled = self.scalers['program_matching'].fit_transform(X)
        
        # Fit clustering
        self.models['applicant_clustering'].fit(X_scaled)
        
        # Train program matcher based on clusters
        clusters = self.models['applicant_clustering'].labels_
        
        # Create synthetic program targets based on clusters
        program_targets = self._create_program_targets(data, clusters)
        
        self.models['program_matcher'].fit(X_scaled, program_targets)
    
    def _create_program_targets(self, data: pd.DataFrame, clusters: np.ndarray) -> np.ndarray:
        """Create program matching targets based on applicant characteristics"""
        
        # Simple heuristic: match programs based on employment and education
        targets = []
        
        for idx, cluster in enumerate(clusters):
            row = data.iloc[idx]
            
            if row['employment_status'] == 1:  # unemployed
                target = 0  # upskilling
            elif row['education_level'] <= 1:  # low education
                target = 1  # job_matching
            elif row['family_size'] >= 4:
                target = 3  # education_support
            else:
                target = 2  # entrepreneurship
            
            targets.append(target)
        
        return np.array(targets)
    
    def save_models(self):
        """Saves all trained models and scalers to disk, overwriting existing files."""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        logger.info("Saving models...")

        for name, model in self.models.items():
            model_file = os.path.join(self.model_path, f"{name}_model.joblib")
            joblib.dump(model, model_file)
            logger.debug(f"Saved model '{name}' to {model_file}")

        for name, scaler in self.scalers.items():
            if hasattr(scaler, 'mean_'):  # Only save fitted scalers
                scaler_file = os.path.join(self.model_path, f"{name}_scaler.joblib")
                joblib.dump(scaler, scaler_file)
                logger.debug(f"Saved scaler for '{name}' to {scaler_file}")
        
        logger.success("All models and scalers have been saved.")

    def load_models(self) -> bool:
        """
        Loads all models and scalers from disk from fixed filenames.
        """
        if not os.path.exists(self.model_path):
            logger.warning("Model directory not found. Models cannot be loaded.")
            return False

        logger.info("Attempting to load models from fixed file paths...")
        loaded_any = False

        for name in self.models.keys():
            model_file = os.path.join(self.model_path, f"{name}_model.joblib")
            if os.path.exists(model_file):
                try:
                    self.models[name] = joblib.load(model_file)
                    logger.debug(f"Loaded model '{name}' from {model_file}")
                    loaded_any = True
                except Exception as e:
                    logger.error(f"Failed to load model {name} from {model_file}: {e}")
            else:
                logger.warning(f"Model file not found for '{name}': {model_file}")


        for name in self.scalers.keys():
            scaler_file = os.path.join(self.model_path, f"{name}_scaler.joblib")
            if os.path.exists(scaler_file):
                try:
                    self.scalers[name] = joblib.load(scaler_file)
                    logger.debug(f"Loaded scaler '{name}' from {scaler_file}")
                except Exception as e:
                    logger.error(f"Failed to load scaler {name} from {scaler_file}: {e}")
            else:
                # This might be okay if a model doesn't have a scaler
                logger.info(f"Scaler file not found for '{name}': {scaler_file}")

        if loaded_any:
            logger.success("Successfully loaded available models.")
        else:
            logger.warning("No models were found or loaded from fixed paths.")
            
        return loaded_any
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        
        info = {}
        
        for model_name, model in self.models.items():
            info[model_name] = {
                'type': type(model).__name__,
                'features': self.feature_columns.get(model_name, []),
                'trained': hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')
            }
        
        return info
    
    def _ensure_models_ready(self):
        """Ensure models are ready for prediction by loading or creating minimal training"""
        try:
            # Try to load existing models
            if not self.load_models():
                logger.info("Could not load models, creating minimal training data...")
                self._create_minimal_training()

            # Check if critical models are fitted
            if not self._check_models_fitted():
                logger.info("Models not fitted, creating minimal training data...")
                self._create_minimal_training()
                
        except Exception as e:
            logger.warning(f"Could not load models: {e}, creating minimal training data...")
            self._create_minimal_training()
    
    def _check_models_fitted(self) -> bool:
        """Check if critical models are fitted and ready for prediction"""
        try:
            # Test fraud detection model (most critical)
            test_features = np.array([[1000, 2, 5000, 12, 0, 600, 0.5]]).reshape(1, -1)
            
            # Try to predict with fraud model
            _ = self.models['fraud_anomaly'].decision_function(test_features)
            
            return True
        except:
            return False
    
    def _create_minimal_training(self):
        """Create minimal synthetic training data to make models functional"""
        try:
            logger.info("Creating minimal synthetic training data...")
            
            # Create minimal synthetic data
            np.random.seed(42)
            n_samples = 100
            
            # Generate synthetic features
            data = {
                'monthly_income': np.random.normal(2000, 800, n_samples),
                'family_size': np.random.randint(1, 8, n_samples),
                'number_of_dependents': np.random.randint(0, 5, n_samples),
                'employment_duration_months': np.random.randint(0, 120, n_samples),
                'total_assets': np.random.normal(10000, 5000, n_samples),
                'total_liabilities': np.random.normal(5000, 3000, n_samples),
                'monthly_rent': np.random.normal(800, 300, n_samples),
                'credit_score': np.random.randint(300, 850, n_samples),
                'employment_status': np.random.randint(1, 4, n_samples),
                'housing_type': np.random.randint(0, 4, n_samples),
                'has_medical_conditions': np.random.randint(0, 2, n_samples),
                'debt_to_income_ratio': np.random.uniform(0, 2, n_samples),
                'previous_applications': np.random.randint(0, 5, n_samples),
                'has_criminal_record': np.random.randint(0, 2, n_samples),
                'education_level': np.random.randint(0, 4, n_samples)
            }
            
            # Ensure non-negative values where appropriate
            data['monthly_income'] = np.abs(data['monthly_income'])
            data['total_assets'] = np.abs(data['total_assets'])
            data['total_liabilities'] = np.abs(data['total_liabilities'])
            data['monthly_rent'] = np.abs(data['monthly_rent'])
            
            df = pd.DataFrame(data)
            
            # Train models with minimal data
            training_result = self.train_models(df)
            logger.info(f"Minimal training completed: {training_result.get('status', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to create minimal training: {e}")
            # If training fails, at least ensure fraud detection has a fallback
            logger.info("Training failed, fraud detection will use rule-based fallback") 