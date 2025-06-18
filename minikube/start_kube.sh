#!/bin/bash

minikube start --driver=docker --container-runtime=docker --gpus all

kubectl apply -f k8s/yaml/pdf-data-pv.yaml
kubectl apply -f k8s/yaml/pdf-data-pvc.yaml
kubectl apply -f k8s/yaml/pdf_uploader
kubectl cp ./data/pdfs/. data-prep/pdf-uploader:/data
