# krspii2
## KubeRnetes-Scaled PIpelIne (for LLMs)
(The 'ii2' is in the word pipeline)

This repo is to explore a little bit of a whole lot: Python&CUDA, DevOps/IaC and some LLMs!

## Terraform

## Kubernetes
Once we've terraformed out homelab nodes we need to then spin up a k8s cluster.

During this initial prototyping we'll use Minikube, then once we get further along we'll use VMware's free ESXi 8

Initially I'm running this on a BM Server:

`Distributor ID: Ubuntu`
`Description:    Ubuntu 22.04.5 LTS`
`Release:        22.04`
`Codename:       jammy`

With an Nvidia GTX 3070 Ti

### Docker Installation:
1. `sudo apt update`
2. `sudo apt install -y apt-transport-https ca-certificates curl software-properties-common`
3. `curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg`
4. `echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null`
5. `sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io`
6. `sudo systemctl start docker && sudo systemctl enable docker`
7. `sudo usermod -aG docker $USER`

### Minikube Installation
1. `curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64`
2. `sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64`
3. `minikube version`
4. `minikube start --driver=docker`
5. `minikube config set driver docker`

### Kubectl Installation
1. `curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"`
2. `sudo install kubectl /usr/local/bin/kubectl`
3. `kubectl version --client`

### Nvidia Container/Docker Toolkit
1. `curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list`
2. `sudo apt-get update`
3. `NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1`
4. `sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}`


### General Minikube
1. `minikube start --driver=docker --container-runtime=docker --gpus all`
2. `minikube addons enable nvidia-device-plugin`
3. `kubectl get nodes -o json | jq .items[].status.capacity`
4. Verify this outputs 0: `sudo sysctl net.core.bpf_jit_harden`
5. `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`
6. `minikube delete`
7. `minikube start --driver docker --container-runtime docker --gpus all`
8. Nvidia should now print: `kubectl get nodes -ojson | jq .items[].status.capacity`
