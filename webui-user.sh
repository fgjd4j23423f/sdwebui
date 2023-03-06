#!/bin/bash
export COMMANDLINE_ARGS="--xformers --enable-insecure-extension-access --cloudflared --share --vae-path /content/stable-diffusion-webui/models/Stable-diffusion/main.vae.pt --no-half-vae --ckpt /content/stable-diffusion-webui/models/Stable-diffusion/model.ckpt --disable-safe-unpickle --disable-console-progressbars --ui-settings-file settings.json --skip-torch-cuda-test"
export ACCELERATE="True"
python launch.py