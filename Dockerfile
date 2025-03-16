# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/ .

# Expose the port that the app runs on
EXPOSE 8002

# Command to run the app with uvicorn
CMD ["uvicorn", "fraud_api:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]