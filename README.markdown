# Fraud Detection System for Credit Card Transactions

## Overview
This project implements a machine learning-based system to detect fraudulent credit card transactions. It integrates with the Payment Transaction Simulator hosted at `http://transaction-simulator-frontend-roolpho0.s3-website-us-east-1.amazonaws.com` to fetch real-time transaction data and send fraud predictions. The system uses a Random Forest classifier trained on synthetic transaction data to identify potentially fraudulent activities.

## Features
- **Machine Learning Model**: Utilizes a Random Forest classifier to predict fraud based on transaction features (amount, time, merchant category, distance from home).
- **API Integration**: Connects to the Payment Transaction Simulator's API to retrieve transactions and send fraud predictions.
- **Scalability**: Handles large transaction volumes with efficient preprocessing and model persistence.
- **Error Handling**: Includes robust logging and error handling for reliable API communication.
- **Synthetic Data**: Generates synthetic transaction data for training when real data is unavailable.

## Prerequisites
- Python 3.8+
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `requests`, `joblib`
- Access to the Payment Transaction Simulator's API (authentication key, if required)
- LaTeX environment (for compiling the portfolio document)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd fraud-detection-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn requests joblib
   ```

3. **Configure API Access**:
   - Open `fraud_detection.py` and update the `api_key` variable with your Payment Transaction Simulator API key (if required). Set to `None` if no authentication is needed:
     ```python
     self.api_key = "your_api_key_here"  # or None
     ```
   - Verify the API endpoint:
     ```python
     self.api_endpoint = "http://transaction-simulator-frontend-roolpho0.s3-website-us-east-1.amazonaws.com/api/transactions"
     ```
     Adjust if the Payment Transaction Simulator uses a different endpoint.

## Usage
1. **Run the Fraud Detection System**:
   ```bash
   python fraud_detection.py
   ```
   - The script generates synthetic data, trains the model (if not already trained), fetches transactions from the Payment Transaction Simulator, predicts fraud, and sends results back to the API.
   - Model and scaler are saved to `fraud_model.pkl` and `scaler.pkl` for reuse.

2. **Compile the Portfolio**:
   - The project includes a LaTeX portfolio (`portfolio.tex`). Compile it to PDF using:
     ```bash
     latexmk -pdf portfolio.tex
     ```
   - Requires a LaTeX distribution (e.g., TeX Live).

## Project Structure
- `fraud_detection.py`: Main Python script containing the `FraudDetector` class for training, prediction, and API integration.
- `portfolio.tex`: LaTeX document summarizing the project for stakeholders.
- `fraud_model.pkl`: Saved Random Forest model (generated after first run).
- `scaler.pkl`: Saved StandardScaler for feature preprocessing (generated after first run).

## API Integration
The system integrates with the Payment Transaction Simulator:
- **GET Request**: Fetches transactions from `http://transaction-simulator-frontend-roolpho0.s3-website-us-east-1.amazonaws.com/api/transactions`.
- **POST Request**: Sends fraud predictions to `/api/transactions/predictions`.
- **Assumptions**:
  - The API returns transaction data in JSON format with fields like `transaction_amount`, `transaction_time`, `merchant_category`, and `distance_from_home`.
  - Authentication may be required (Bearer token).
  - If the API structure differs, update the `fetch_transactions_from_api` and `send_predictions_to_api` methods in `fraud_detection.py`.

## Notes
- **Synthetic Data**: The system uses synthetic data for training due to the absence of real transaction data. Update `generate_synthetic_data` if real data becomes available.
- **API Endpoint**: The provided URL is an S3 static website endpoint. The code assumes a backend API at `/api/transactions`. If the Payment Transaction Simulator uses a different endpoint or HTTPS, modify `api_endpoint` accordingly.
- **Model Tuning**: The Random Forest model uses default parameters. Tune hyperparameters (e.g., `n_estimators`, `max_depth`) for better performance on real data.
- **Logging**: Check logs in the console for debugging and monitoring.

## Future Enhancements
- Implement real-time feature engineering for dynamic fraud patterns.
- Explore deep learning models for improved accuracy.
- Enhance API security with advanced authentication protocols.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details (if applicable).

## Contact
For issues or questions, please contact the project maintainer or open an issue on the repository.