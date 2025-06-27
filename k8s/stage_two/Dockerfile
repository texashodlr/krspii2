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

# Never forget your banana!
COPY banana.txt .

# Copy the preprocessing script
COPY fine_tune_v2.py .

ARG HF_TOKEN
ENV HUGGINFACE_TOKEN=$HF_TOKEN

# Command to run the script
ENTRYPOINT ["python3", "fine_tune_v2.py"]
CMD ["--data-dir", "../../data/pruned", "--output-dir", "../../data/model", "--model-name", "mistralai/Mistral-7B-v0.1","--lora-rank","8","--max-length","512"]
