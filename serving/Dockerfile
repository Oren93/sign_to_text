# Use a Python base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for MediaPipe and nginx
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python application code into the container
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install mediapipe
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the Python application code into the container
COPY . /app

# Create nginx user
RUN useradd -r nginx

# Copy the nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose ports for uvicorn and nginx
EXPOSE 8000 8080

# Command to start both uvicorn server and nginx when the container starts
CMD ["bash"]
