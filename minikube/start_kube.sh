#!/bin/bash

minikube start --driver=docker --container-runtime=docker --gpus all

kubectl create ns data-prep
kubectl apply -f k8s/yaml/pdf-data-pv.yaml
kubectl apply -f k8s/yaml/pdf-data-pvc.yaml
kubectl apply -f k8s/yaml/pdf_uploader.yaml
kubectl cp ./data/pdfs/. data-prep/pdf-uploader:/data
