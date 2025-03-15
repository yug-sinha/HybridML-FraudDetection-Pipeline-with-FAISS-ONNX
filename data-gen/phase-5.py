import asyncio
import time
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import TransportError
from prometheus_client import CollectorRegistry, Counter, generate_latest, CONTENT_TYPE_LATEST
from twilio.rest import Client as TwilioClient
import uvicorn

# -------------------------------
# Global and Configurations
# -------------------------------
FRAUD_THRESHOLD = 0.5

def update_threshold_with_feedback(feedback_value: bool):
    global FRAUD_THRESHOLD
    if feedback_value:
        FRAUD_THRESHOLD = max(0.1, FRAUD_THRESHOLD - 0.05)
    else:
        FRAUD_THRESHOLD = min(0.9, FRAUD_THRESHOLD + 0.05)
    return FRAUD_THRESHOLD

class Feedback(BaseModel):
    alert_id: str
    is_correct: bool

# -------------------------------
# FastAPI App Setup
# -------------------------------
app = FastAPI()

# Elasticsearch client (adjust as needed)
es = Elasticsearch(hosts=["http://localhost:9200"])

# Create a custom Prometheus registry and define the alert counter.
registry = CollectorRegistry()
alert_counter = Counter('fraud_alerts_total', 'Total number of fraud alerts triggered', registry=registry)

# Twilio configuration (replace with your credentials or set as environment variables)
TWILIO_ACCOUNT_SID = "your_twilio_account_sid"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
TWILIO_FROM_NUMBER = "+1234567890"
TWILIO_TO_NUMBER = "+0987654321"

twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    try:
        twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    except Exception as e:
        print("Error initializing Twilio client:", e)

# -------------------------------
# WebSocket Manager for Real-Time Alerts
# -------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print("Error broadcasting message:", e)

manager = ConnectionManager()

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket disconnected.")

# -------------------------------
# Helper Functions for Alerts
# -------------------------------
def send_sms_alert(message: str):
    if twilio_client:
        try:
            twilio_client.messages.create(
                body=message,
                from_=TWILIO_FROM_NUMBER,
                to=TWILIO_TO_NUMBER
            )
            print("SMS alert sent.")
        except Exception as e:
            print("Error sending SMS alert:", e)
    else:
        print("Twilio client not configured. SMS alert not sent.")

def log_alert_to_es(alert_data: dict):
    try:
        es.index(index="fraud-alerts", document=alert_data)
        print("Alert logged to Elasticsearch.")
    except TransportError as e:
        print("Error logging alert to Elasticsearch:", e)

# -------------------------------
# REST Endpoint: Trigger Alert
# -------------------------------
@app.post("/trigger_alert")
async def trigger_alert(transaction: dict):
    transaction_text = transaction.get("transaction", "")
    fraud_score = transaction.get("fraud_score", 0)
    
    if fraud_score > FRAUD_THRESHOLD:
        alert_data = {
            "transaction": transaction_text,
            "fraud_score": fraud_score,
            "timestamp": time.time()
        }
        log_alert_to_es(alert_data)
        alert_counter.inc()
        await manager.broadcast(f"ALERT: {transaction_text} with score {fraud_score}")
        send_sms_alert(f"Fraud Alert: {transaction_text} with score {fraud_score}")
        return {"status": "alert triggered", "alert": alert_data}
    else:
        return {"status": "no alert", "message": "Fraud score below threshold"}

# -------------------------------
# REST Endpoint: Expose Prometheus Metrics
# -------------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

# -------------------------------
# REST Endpoint: Receive Feedback (Human-In-The-Loop)
# -------------------------------
@app.post("/feedback")
async def receive_feedback(feedback: Feedback):
    new_threshold = update_threshold_with_feedback(feedback.is_correct)
    print(f"Feedback for alert {feedback.alert_id} received. Updated threshold: {new_threshold}")
    return {"status": "feedback received", "new_threshold": new_threshold}

# -------------------------------
# REST Endpoint: Simulate a Transaction Stream
# -------------------------------
@app.get("/simulate_stream")
async def simulate_stream():
    dummy_transactions = [
        {"transaction": "Transaction at Acme Corp for $100.50 on 2023-03-15.", "fraud_score": random.uniform(0, 1)},
        {"transaction": "Payment to Global Bank of $2300 flagged as suspicious.", "fraud_score": random.uniform(0, 1)},
        {"transaction": "Refund processed at Retailer Inc.", "fraud_score": random.uniform(0, 1)},
        {"transaction": "Large transaction at Tech Inc. for $5000.", "fraud_score": random.uniform(0, 1)},
        {"transaction": "Small purchase at Corner Shop for $15.", "fraud_score": random.uniform(0, 1)},
        {"transaction": "Suspicious transaction at Unknown Vendor for $9999.", "fraud_score": random.uniform(0, 1)}
    ]
    for txn in dummy_transactions:
        await asyncio.sleep(2)
        response = await trigger_alert(txn)
        print("Simulated alert response:", response)
    return {"status": "stream simulation complete"}

# -------------------------------
# Main: Run the FastAPI App
# -------------------------------
if __name__ == "__main__":
    uvicorn.run("phase-5:app", host="0.0.0.0", port=8000, reload=True)