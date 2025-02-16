# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    portaudio19-dev \
    ffmpeg \
    x11-apps \
    libx11-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the DISPLAY environment variable
ENV DISPLAY=:0

# Make port 80 available to the world outside this container
EXPOSE 80

# Run Main.py when the container launches
CMD ["python", "Main.py"]