# fraud_api.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random
import strawberry
from strawberry.fastapi import GraphQLRouter
from pydantic import BaseModel

# Dummy fraud detection function (replace with your real model)
def detect_fraud(transaction_text: str) -> dict:
    fraud_score = random.uniform(0, 1)
    is_fraud = fraud_score > 0.5
    return {"transaction": transaction_text, "fraud_score": fraud_score, "is_fraud": is_fraud}

# REST API setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development; restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/detect_fraud")
async def detect_fraud_endpoint(transaction: str):
    result = detect_fraud(transaction)
    return result

# GraphQL API using Strawberry
@strawberry.type
class FraudDetection:
    transaction: str
    fraud_score: float
    is_fraud: bool

@strawberry.type
class Query:
    @strawberry.field
    def detect(self, transaction: str) -> FraudDetection:
        result = detect_fraud(transaction)
        return FraudDetection(**result)

schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/")
async def root():
    return {"message": "Fraud Detection API"}

if __name__ == "__main__":
    uvicorn.run("fraud_api:app", host="0.0.0.0", port=8002, reload=True)