import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import logging
from datetime import datetime
import os
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class FraudDetectionSystem:
    def __init__(self):
        try:
            self.scaler = StandardScaler()
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.xgb_model = xgb.XGBClassifier(random_state=42)
            self.initialize_models()
            logger.info("Fraud Detection System initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Fraud Detection System: {str(e)}")
            raise

    def load_data(self):
        logger.info("Loading credit card transaction data...")
        try:
            # Read the CSV file
            df = pd.read_csv("C:/Users/SATHVIK U S/Downloads/archive/creditcard.csv")

            # Store feature names excluding the target variable
            self.feature_names = [col for col in df.columns if col != 'Class']

            logger.info(f"Dataset Shape: {df.shape}")
            logger.info(f"Features: {self.feature_names}")
            logger.info(f"Class Distribution:\n{df['Class'].value_counts()}")

            # Handle any missing values if present
            if df.isnull().sum().any():
                logger.info("Handling missing values...")
                df = df.fillna(df.mean())

            return df

        except FileNotFoundError:
            logger.error("Credit card data file not found. Please ensure 'credit_card_data.csv' is in the correct location.")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, df):
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def initialize_models(self):
        try:
            df = self.load_data()
            X_train, X_test, y_train, y_test = self.preprocess_data(df)
            self.train_models(X_train, y_train)

            # Evaluate models and store metrics
            self.rf_metrics = self.evaluate_model(self.rf_model, X_test, y_test, "Random Forest")
            self.xgb_metrics = self.evaluate_model(self.xgb_model, X_test, y_test, "XGBoost")

            logger.info("Models trained and evaluated successfully")
        except Exception as e:
            logger.error(f"Error in model initialization: {str(e)}")
            raise

    def train_models(self, X_train, y_train):
        logger.info("Training models...")
        self.rf_model.fit(X_train, y_train)
        self.xgb_model.fit(X_train, y_train)

    def predict_transaction(self, transaction_data):
        try:
            # Create DataFrame with the same structure as training data
            input_data = {}
            fields = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            
            # Convert input to DataFrame
            input_df = pd.DataFrame([dict(zip(fields, transaction_data))])
            
            # Scale the input using the same scaler used for training
            scaled_data = self.scaler.transform(input_df)
            
            # Get predictions from both models
            rf_pred = self.rf_model.predict(scaled_data)[0]
            rf_prob = self.rf_model.predict_proba(scaled_data)[0]
            rf_risk = float(rf_prob[1]) * 100  # Convert to percentage
            
            xgb_pred = self.xgb_model.predict(scaled_data)[0]
            xgb_prob = self.xgb_model.predict_proba(scaled_data)[0]
            xgb_risk = float(xgb_prob[1]) * 100  # Convert to percentage
            
            # Get feature importance and identify risk factors
            risk_factors = []
            suspicious_factors = []
            
            # Get top contributing features
            feature_importance = []
            for i, feature in enumerate(fields):
                rf_imp = self.rf_model.feature_importances_[i]
                xgb_imp = self.xgb_model.feature_importances_[i]
                avg_imp = (rf_imp + xgb_imp) / 2
                scaled_value = abs(scaled_data[0][i])
                contribution = avg_imp * scaled_value
                feature_importance.append((feature, contribution))
            
            # Sort by contribution and get top factors
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            for feature, contribution in feature_importance[:5]:
                if contribution > 0.1:
                    if feature == 'Amount':
                        risk_factors.append(f"Unusual transaction amount: ₹{input_df['Amount'].iloc[0]:,.2f}")
                    else:
                        risk_factors.append(f"Unusual pattern in {feature}")
                elif contribution > 0.05:
                    if feature == 'Amount':
                        suspicious_factors.append(f"Slightly unusual amount: ₹{input_df['Amount'].iloc[0]:,.2f}")
                    else:
                        suspicious_factors.append(f"Slightly unusual pattern in {feature}")
            
            # Calculate risk score based primarily on number of risk factors
            num_risk_factors = len(risk_factors)
            num_suspicious_factors = len(suspicious_factors)
            
            # Automatic high risk if 3 or more risk factors
            if num_risk_factors >= 3:
                risk_level = "HIGH"
                final_risk_score = min(100, 70 + (num_risk_factors * 5))  # Base 70% + 5% per risk factor
                recommendation = "Transaction is HIGH RISK - Multiple significant risk factors detected"
            # Automatic medium risk if 2 risk factors
            elif num_risk_factors >= 2:
                risk_level = "MEDIUM"
                final_risk_score = min(100, 50 + (num_risk_factors * 10))  # Base 50% + 10% per risk factor
                recommendation = "Transaction requires verification - Multiple risk factors detected"
            else:
                # Calculate weighted risk score
                model_risk = (rf_risk + xgb_risk) / 2
                risk_factor_score = num_risk_factors * 30  # Each risk factor adds 30%
                suspicious_factor_score = num_suspicious_factors * 15  # Each suspicious factor adds 15%
                
                final_risk_score = min(100, (
                    0.2 * model_risk +  # Model predictions (reduced weight)
                    0.6 * risk_factor_score +  # High risk factors (increased weight)
                    0.2 * suspicious_factor_score  # Suspicious factors
                ))
                
                if final_risk_score > 70:
                    risk_level = "HIGH"
                    recommendation = "Transaction is HIGH RISK - Multiple risk indicators present"
                elif final_risk_score > 30:
                    risk_level = "MEDIUM"
                    recommendation = "Transaction requires verification - Some risk factors present"
                else:
                    risk_level = "LOW"
                    recommendation = "Transaction appears legitimate - Low risk detected"

            # Add risk factors to recommendation
            if risk_factors:
                recommendation += "\nHigh-risk factors identified:\n- " + "\n- ".join(risk_factors)
            if suspicious_factors:
                recommendation += "\nSuspicious factors identified:\n- " + "\n- ".join(suspicious_factors)

            return {
                'random_forest': {
                    'prediction': 'FRAUDULENT' if rf_pred == 1 else 'LEGITIMATE',
                    'confidence': float(max(rf_prob)),
                    'risk_score': rf_risk
                },
                'xgboost': {
                    'prediction': 'FRAUDULENT' if xgb_pred == 1 else 'LEGITIMATE',
                    'confidence': float(max(xgb_prob)),
                    'risk_score': xgb_risk
                },
                'overall_assessment': {
                    'risk_level': risk_level,
                    'risk_score': final_risk_score,
                    'final_verdict': 'FRAUDULENT' if final_risk_score > 50 else 'LEGITIMATE',
                    'recommendation': recommendation,
                    'risk_factors': risk_factors,
                    'suspicious_factors': suspicious_factors
                }
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def evaluate_model(self, model, X_test, y_test, model_name):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)

        # Generate confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        cm_image = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix_image': cm_image
        }

# Initialize the fraud detection system
fraud_system = FraudDetectionSystem()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No data provided")

        # Map frontend fields to model features
        mapped_data = {
            'Time': float(data.get('transaction_time', 0)) * 3600,  # Convert hours to seconds
            'Amount': float(data.get('transaction_amount', 0)),
            'V1': float(data.get('card_present', False)),
            'V2': float(data.get('cvv_provided', False)),
            'V3': float(data.get('distance_from_home', 0)) / 100,
            'V4': float(data.get('ratio_to_median_purchase', 1)),
            'V5': float(data.get('repeat_retailer', False)),
            'V6': float(data.get('used_chip', False)),
            'V7': float(data.get('used_pin_number', False)),
            'V8': float(data.get('online_order', False)),
            'V9': float(data.get('fraud_history', False)),
            'V10': float(data.get('transaction_freq_24h', 0)) / 24,
            'V11': float(data.get('transaction_freq_7d', 0)) / 168,
            'V12': float(data.get('daily_rate_compared_to_avg', 1)),
            'V13': float(data.get('medium_purchase_price', 0)) / 1000,
            'V14': float(data.get('purchase_time_variance', 0)) / 24,
            'V15': 0 if data.get('transaction_type') == 'In-store' else 1,
            'V16': (float(data.get('customer_age', 18)) - 18) / 82,
            'V17': {'Retail': 0, 'Travel': 1, 'Entertainment': 2, 'Groceries': 3, 
                   'Restaurant': 4, 'Services': 5, 'Healthcare': 6, 
                   'Gas/Automotive': 7, 'Electronics': 8, 'Other': 9
                   }.get(data.get('merchant_category', 'Other'), 9) / 9,
            'V18': (float(data.get('merchant_rating', 3)) - 1) / 4,
            'V19': float(data.get('num_declined_24h', 0)) / 10,
            'V20': float(data.get('num_declined_7d', 0)) / 50,
            'V21': float(data.get('foreign_transaction', False)),
            'V22': float(data.get('high_risk_country', False)),
            'V23': float(data.get('high_risk_email', False)),
            'V24': float(data.get('high_risk_ip', False)),
            'V25': float(data.get('high_risk_device', False)),
            'V26': float(data.get('device_change', False)),
            'V27': float(data.get('location_change', False)),
            'V28': float(data.get('billing_shipping_mismatch', False))
        }

        # Convert mapped data to list in correct order
        required_fields = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        transaction_data = [mapped_data[field] for field in required_fields]

        # Get model predictions
        result = fraud_system.predict_transaction(transaction_data)

        # Calculate risk scores and levels
        rf_risk = result['random_forest']['risk_score']
        xgb_risk = result['xgboost']['risk_score']
        avg_risk = (rf_risk + xgb_risk) / 2

        # Determine risk level based on multiple factors
        risk_factors_count = len(result['overall_assessment']['risk_factors'])
        suspicious_factors_count = len(result['overall_assessment']['suspicious_factors'])
        
        # Calculate weighted risk score
        weighted_risk = (
            0.4 * avg_risk +  # Model predictions
            0.3 * (risk_factors_count * 20) +  # High risk factors
            0.2 * (suspicious_factors_count * 10) +  # Suspicious factors
            0.1 * (100 if data.get('fraud_history', False) else 0)  # Fraud history
        )

        # Prepare response with transaction details
        response = {
            'success': True,
            'transaction_details': {
                'amount': float(data['transaction_amount']),
                'timestamp': datetime.now().isoformat(),
                'transaction_type': data.get('transaction_type', 'Unknown'),
                'merchant_category': data.get('merchant_category', 'Unknown')
            },
            'predictions': {
                'random_forest': {
                    'prediction': result['random_forest']['prediction'],
                    'probability': result['random_forest']['confidence'],
                    'risk_score': rf_risk
                },
                'xgboost': {
                    'prediction': result['xgboost']['prediction'],
                    'probability': result['xgboost']['confidence'],
                    'risk_score': xgb_risk
                },
                'overall_assessment': {
                    'risk_level': result['overall_assessment']['risk_level'],
                    'risk_score': weighted_risk,
                    'final_verdict': result['overall_assessment']['final_verdict'],
                    'recommendation': result['overall_assessment']['recommendation'],
                    'risk_factors': result['overall_assessment']['risk_factors'],
                    'suspicious_factors': result['overall_assessment']['suspicious_factors']
                }
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/model-metrics', methods=['GET'])
def model_metrics():
    try:
        return jsonify({
            'success': True,
            'random_forest': fraud_system.rf_metrics,
            'xgboost': fraud_system.xgb_metrics
        })
    except Exception as e:
        logger.error(f"Error retrieving model metrics: {str(e)}")
        return jsonify({
            'success': False,
            'error': "An error occurred while retrieving model metrics"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)