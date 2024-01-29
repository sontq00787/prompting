#!/bin/bash

# Install python3-pip
sudo apt install -y python3-pip

# Upgrade bittensor
python3 -m pip install --upgrade bittensor

# Used to checkout wallet tree; tree ~/.bittensor
apt install tree

# Install PM2 to manage workflows
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update

# Clone the repository
git clone https://github.com/opentensor/prompting.git

# Change to the prompting directory
cd prompting

# Install Python dependencies
python3 -m pip install -r requirements.txt

python3 -m pip install -r neurons/miners/openai/requirements.txt

# Install prompting package
python3 -m pip install -e .

echo "Script completed successfully."
