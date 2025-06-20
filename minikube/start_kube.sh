#!/bin/bash

# Making the node directory, assuming it didn't already exist
pdf_data='/mnt/data'

sudo mkdir -p "$pdf_data"

echo "Directory '$pdf_data' created (if it didn't already exist)."

# Starting Minikube
minikube start --driver=docker --container-runtime=docker --gpus all

kubectl create ns data-prep
kubectl apply -f k8s/yaml/pdf-data-pv.yaml
kubectl apply -f k8s/yaml/pdf-data-pvc.yaml
kubectl apply -f k8s/yaml/pdf_loader.yaml

# Wait until the pdf-loader pod is ready
echo "Waiting for pod pdf-uploader in data-prep to be ready..."
until kubectl -n data-prep get pod pdf-uploader -o jsonpath="{.status.containerStatuses[0].ready}" 2>/dev/null | grep -q true; do
	echo -n "."
	sleep 2
done

echo -e "\nPod is ready. Running command: "

kubectl cp ./data/pdfs/. data-prep/pdf-uploader:/data/pdfs
kubectl -n data-prep exec pdf-uploader -- sh -c "ls -lh /data/pdfs"

# Applying the yaml for PDF to JSONL Job.
kubectl apply -f k8s/yaml/preprocess_pdfs_job.yaml
