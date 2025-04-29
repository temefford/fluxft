#!/bin/bash
set -e  # Exit script on first error
sleep 5 # Wait for the pod to fully start

if [ -n "$RUNPOD_POD_ID" ]; then
    if [ ! -L "examples" ]; then
        echo "~_~S Linking examples folder..."
        ln -s /workspace/axolotl/examples .
    fi

    if [ -n "$HF_TOKEN" ]; then
        echo "~_~T~Q Logging in to Hugging Face..."
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    else
echo "~Z| ~O Warning: HF_TOKEN is not set. Skipping Hugging Face login."
    fi

    if [ ! -L "outputs" ]; then
        echo "~_~S Linking outputs folder..."
        ln -s /workspace/data/axolotl-artifacts .
        mv axolotl-artifacts outputs
    fi
else
    if [ ! -d "outputs" ]; then
        echo "~_~S Creating outputs folder..."
        mkdir outputs
    fi
fi

# check if any env var starting with "AXOLOTL_" is set
if [ -n "$(env | grep '^AXOLOTL_')" ]; then
    echo "~L~[ Preparing..."

    if ! python3 configure.py --template config_template.yaml --output config.yaml; then
        echo "~]~L Configuration failed!"
    fi
fi

# show message of the day at the Pod logs
cat /etc/motd

# Keeps the container running
sleep infinity