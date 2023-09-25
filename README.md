# ASR_wac2vec2
This repo contains codes about loading audio data, training wav2vec2 model with wolof language dataset 
- wolof contains the wolof alffa train datasets 01, 02, 04, 07, 08, 09

- customWav2Vec2_notebook is the notebook used to train and push the model to the hub

- The file requirements.txt contains all packages to install

- The module utils.py defines the class for the fastAPI inference

- main.py constructs the fastapi interface

- gradio_demo and flagged are for the gradio interface

- key and key.pub are the private ant the public key for docker

- Dockerfile is to build docker images and run containers

- captures des results contains all screenshoots of the results obtained for the training and the interfaces (fastapi, gradio, docker-build, docker-run, docker-inspect)
