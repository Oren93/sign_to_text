# Use a minimal Python image as the base
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Create a user
RUN useradd -m oren

# Fix permission issue
RUN mkdir -p /var/lib/apt/lists/partial \
    && chmod 644 /var/lib/apt/lists/partial

# Install required dependencies for MediaPipe
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your application code into the container
COPY . /app

# Install MediaPipe
RUN pip install mediapipe
RUN pip install -r requirements.txt

# Command to start an interactive shell
CMD ["bash"]
