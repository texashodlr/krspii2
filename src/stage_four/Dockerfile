FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Grabbing Python dependencies
RUN apt-get update && apt-get install -y \
	python3.9 \
	python3.9-dev \
	python3-pip \
	git \
	tzdata && \
    	ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    	echo $TZ > /etc/timezone \
	&& rm -rf /var/lib/apt/lists/*

# Working Directory
WORKDIR /app

# Copy Requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the preprocessing script
COPY model_chat.py .

ARG HF_TOKEN
ENV HUGGINFACE_TOKEN=$HF_TOKEN

# Command to run the script
ENTRYPOINT ["python3", "model_chat.py"]
