#!/bin/bash

# Collect ssh keys
/usr/bin/geni-get key > ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa
ssh-keygen -y -f ~/.ssh/id_rsa > ~/.ssh/id_rsa.pub
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 644 ~/.ssh/authorized_keys

# Install docker
sudo apt-get update
sudo apt-get remove docker docker-engine docker.io -y
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker

# Run Jaeger server
if [[ $(hostname) =~ "node0" ]]; then
  sudo docker run -d --rm --name jaeger -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p 5775:5775/udp -p 6831:6831/udp -p \
  6832:6832/udp -p 5778:5778 -p 16686:16686 -p 14268:14268 -p 14250:14250 -p 9411:9411 jaegertracing/all-in-one:1.19
else
  SERVER_IP=$(grep 'node0' /etc/hosts | awk '{print $1}')
  sudo docker run -d --rm -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp -p5778:5778/tcp \
  jaegertracing/jaeger-agent:1.19 --reporter.grpc.host-port=${SERVER_IP}:14250
fi

# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /local/miniconda.sh
bash /local/miniconda.sh -b -p /local/miniconda
eval "$(/local/miniconda/bin/conda shell.bash hook)"
conda create -n distbelief python=3.8 -y
conda activate distbelief

# Set up environment for distbelief
if [ ! -d "/local/distbelief" ]
then
  git clone https://github.com/165749/distbelief.git /local/distbelief
fi
cd /local/distbelief
pip install -r requirements.txt
pip install .
