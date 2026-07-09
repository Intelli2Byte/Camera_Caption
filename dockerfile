# Use an official, lightweight Python base image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by OpenCV and core utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /lib/apt/lists/*

# Create the specific input and output directories required by the hackathon
RUN mkdir -p /input /output

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to take advantage of Docker caching
COPY requirements.txt .

# Install Python dependencies cleanly
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY src/ /app/src/

# Set the environment variable fallback for the path setup
ENV PYTHONPATH=/app

# Define the entrypoint command that execution environments will invoke on startup
ENTRYPOINT ["python3", "src/agent.py"]