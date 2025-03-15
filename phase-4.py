import time
import numpy as np
import dask
from dask.distributed import Client
import shap

# -------------------------------
# Dummy Ensemble Fraud Scoring Function
# -------------------------------
def ensemble_fraud_scoring(transaction_text):
    # For demonstration, we compute a pseudo fraud score based on text length.
    score = (len(transaction_text) % 100) / 100.0  # pseudo score in [0,1]
    return score

# -------------------------------
# Dummy Feature Extraction & Fraud Model for XAI
# -------------------------------
def extract_features(text):
    """Extract simple features from text: length, digit count, uppercase count."""
    length = len(text)
    count_digits = sum(c.isdigit() for c in text)
    count_upper = sum(c.isupper() for c in text)
    return np.array([length, count_digits, count_upper])

def fraud_model(features):
    """
    A dummy linear model that outputs a fraud score.
    Accepts a 2D numpy array of shape (n_samples, n_features) and returns a 1D array of scores.
    """
    features = np.atleast_2d(features)  # Ensure the input is 2D
    # Compute a score for each sample.
    return 0.01 * features[:, 0] + 0.05 * features[:, 1] + 0.02 * features[:, 2]

# -------------------------------
# Process a Single Transaction (ETL → Embedding → Vector Search → Fraud Scoring → Alert)
# -------------------------------
def process_transaction(transaction_text):
    # ETL Preprocessing (simulate cleaning)
    cleaned = transaction_text.strip().lower()
    
    # Simulate embedding generation and vector search (dummy processing time)
    time.sleep(0.1)
    
    # Fraud scoring using the ensemble function
    score = ensemble_fraud_scoring(cleaned)
    alert = score > 0.5  # set threshold for alert
    result = {
        "transaction": transaction_text,
        "cleaned": cleaned,
        "fraud_score": score,
        "alert": alert
    }
    
    # If the transaction is flagged, generate an explanation using SHAP.
    if alert:
        features = extract_features(transaction_text)  # shape: (3,)
        features_2d = features.reshape(1, -1)           # shape: (1, 3)
        # Use the features_2d as both the background data and the sample for explanation.
        explainer = shap.KernelExplainer(fraud_model, features_2d, link="identity")
        shap_values = explainer.shap_values(features_2d)
        result["shap_values"] = shap_values
    return result

# -------------------------------
# Main Pipeline: Simulate Streaming and Process Transactions in Parallel
# -------------------------------
def main():
    # Start a local Dask cluster for parallel processing.
    client = Client()
    
    # Simulated stream of transaction messages.
    transactions = [
        "Transaction at Acme Corp for $100.50 on 2023-03-15.",
        "Payment to Global Bank of $2300 flagged as suspicious.",
        "Refund processed at Retailer Inc.",
        "Large transaction at Tech Inc. for $5000.",
        "Small purchase at Corner Shop for $15.",
        "Suspicious transaction at Unknown Vendor for $9999."
    ]
    
    # Submit transactions for parallel processing.
    futures = client.map(process_transaction, transactions)
    results = client.gather(futures)
    
    # Process results and trigger alerts.
    for res in results:
        print("Transaction:", res["transaction"])
        print("Fraud Score:", res["fraud_score"])
        if res["alert"]:
            print("ALERT: Fraudulent transaction detected!")
            print("SHAP Explanation:", res.get("shap_values"))
        print("-" * 40)
    
    client.close()

if __name__ == "__main__":
    main()