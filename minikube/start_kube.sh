#!/bin/bash

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

kubectl cp ./data/pdfs/. data-prep/pdf-uploader:/data
kubectl -n data-prep exec pdf-uploader -- sh -c "ls -lh /data"
