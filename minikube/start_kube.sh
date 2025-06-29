#!/bin/bash

# Making the node directory, assuming it didn't already exist
pdf_data='/mnt/data'

sudo mkdir -p "$pdf_data"

echo "Directory '$pdf_data' created (if it didn't already exist)."

# Starting Minikube
minikube start --driver=docker --container-runtime=docker --gpus all

kubectl create ns data-prep
kubectl apply -f k8s/yaml/stage_one/pdf-data-pv.yaml
kubectl apply -f k8s/yaml/stage_one/pdf-data-pvc.yaml
kubectl apply -f k8s/yaml/stage_one/pdf_loader.yaml

# Wait until the pdf-loader pod is ready
echo "Waiting for pod pdf-uploader in data-prep to be ready..."
until kubectl -n data-prep get pod pdf-uploader -o jsonpath="{.status.containerStatuses[0].ready}" 2>/dev/null | grep -q true; do
        echo -n "."
        sleep 2
done

echo -e "\nPod is ready. Running command: "

kubectl cp ./data/pdfs/. data-prep/pdf-uploader:/data/pdfs/pdfs
kubectl -n data-prep exec pdf-uploader -- sh -c "ls -lh /data/pdfs/pdfs"

# Applying the yaml for PDF to JSONL Job.
kubectl apply -f k8s/yaml/stage_one/preprocess_pdfs_job.yaml

# Validating that the processed PDFs are occuring.
echo "Waiting for the PDF to JSONL processing to complete..."
target_job=4
target_count=24
while true; do
	# Jobs

	completed=$(kubectl -n data-prep get job pdf-preprocess -o jsonpath='{.status.succeeded}')
	completed=${completed:-0}
	echo "Current completions: $completed"
	if ["$completed" -eq "$target_job"]; then
		echo "Processing job has reached $target completions."
	fi

	file_count=$(kubectl -n data-prep exec pdf-uploader -- sh -c "ls -l /data/pdfs/processed | wc -l")
	#file_count=$(ls -l "/data/pdfs/processed" | wc -l)
	echo "Current file count: $file_count"
	if ["$file_count" -eq "$target_count"]; then
		echo "Directory now has $target_count files!"
		echo -n "."
		sleep 2
		break
	fi
done
echo "All PDFs processed!"
