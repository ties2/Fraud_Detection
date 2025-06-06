import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import requests
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self, model_path='fraud_model.pkl', scaler_path='scaler.pkl'):
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.api_endpoint = "https://api.transaction-m.com/transactions"  # Hypothetical API endpoint
        self.api_key = "your_api_key_here"  # Replace with actual API key

    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic transaction data for training"""
        np.random.seed(42)
        data = {
            'transaction_amount': np.random.uniform(1, 10000, n_samples),
            'transaction_time': [datetime.now().timestamp() - np.random.randint(0, 86400) for _ in range(n_samples)],
            'merchant_category': np.random.choice(['grocery', 'electronics', 'clothing', 'travel'], n_samples),
            'distance_from_home': np.random.uniform(0, 1000, n_samples),
            'is_fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])  # 2% fraud rate
        }
        df = pd.DataFrame(data)
        df['merchant_category'] = df['merchant_category'].astype('category').cat.codes
        logger.info("Synthetic data generated")
        return df

    def preprocess_data(self, df):
        """Preprocess transaction data"""
        features = ['transaction_amount', 'transaction_time', 'merchant_category', 'distance_from_home']
        X = df[features]
        y = df['is_fraud'] if 'is_fraud' in df.columns else None
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        logger.info("Data preprocessed")
        return X_scaled, y

    def train_model(self, X, y):
        """Train the fraud detection model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        logger.info("Model trained")

        # Evaluate model
        y_pred = self.model.predict(X_test)
        logger.info("Model evaluation:\n" + classification_report(y_test, y_pred))
        logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

    def fetch_transactions_from_api(self):
        """Fetch new transactions from the transaction system (Payment Transaction Simulator)"""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.get(self.api_endpoint, headers=headers)
            response.raise_for_status()
            transactions = response.json()
            df = pd.DataFrame(transactions)
            logger.info(f"Fetched {len(df)} transactions from API")
            return df
        except requests.RequestException as e:
            logger.error(f"Error fetching transactions: {e}")
            return None

    def predict_fraud(self, transactions):
        """Predict fraud for new transactions"""
        X, _ = self.preprocess_data(transactions)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        return predictions, probabilities

    def send_predictions_to_api(self, transaction_ids, predictions, probabilities):
        """Send fraud predictions back to the transaction system (M)"""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            payload = [
                {'transaction_id': t_id, 'is_fraud': int(pred), 'fraud_probability': float(prob)}
                for t_id, pred, prob in zip(transaction_ids, predictions, probabilities)
            ]
            response = requests.post(self.api_endpoint + '/predictions', json=payload, headers=headers)
            response.raise_for_status()
            logger.info("Predictions sent to API")
        except requests.RequestException as e:
            logger.error(f"Error sending predictions: {e}")

    def run(self):
        """Main execution flow"""
        # Generate or load data (in practice, this would come from the API)
        data = self.generate_synthetic_data()
        X, y = self.preprocess_data(data)
        self.train_model(X, y)

        # Fetch new transactions from API
        new_transactions = self.fetch_transactions_from_api()
        if new_transactions is not None:
            predictions, probabilities = self.predict_fraud(new_transactions)
            self.send_predictions_to_api(new_transactions['transaction_id'], predictions, probabilities)
            logger.info("Fraud detection completed")

if __name__ == "__main__":
    detector = FraudDetector()
    detector.run()