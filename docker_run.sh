#!/bin/bash

nvidia-docker run --ipc=host -v $PWD:/workspace -v /media:/media -v /media/rbrisk/storage01/pretrained_models:/root/.cache/torch/hub/checkpoints --rm -it docbrisky/monai:latest
