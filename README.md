# 🩸 HEMO-AI

🔬 Overview

HEMO-AI is a deep learning framework for quantitative prediction of hemoglobin (Hb) and hematocrit (Hct) from time-series blood spot images.

The framework combines:

Dynamic Vision Transformer (DynamicViT) with patch pruning

Recurrent neural networks (LSTM / GRU) for temporal modeling

Efficient inference via adaptive patch pruning

This repository is designed for research reproducibility, clinical AI prototyping, and POCT (Point-of-Care Testing) applications.

✔ System Requirements

Python ≥ 3.8

Ubuntu ≥ 16.04

NVIDIA GPU (recommended for training)

PyTorch ≥ 1.12.1

✔ Environment Setting

1. Install Git

sudo apt-get install git
Configure Git (optional but recommended):
git config --global user.name <your_name>
git config --global user.email <your_email>

2. Clone the Repository

cd <your_path>
git clone https://github.com/<your-github-id>/HEMO-AI.git
cd HEMO-AI

3. Create a Virtual Environment (Optional but Recommended)

Install venv 
sudo apt-get install python3-venv

Create a virtual environment 
python3 -m venv <venv_name>

Activate the virtual environment 
source ./<venv_name>/bin/activate

4. Install Required Packages
pip install -r requirements.txt


📁 Project Structure

HEMO-AI/

config.py          # Central configuration file
model.py           # DynamicViT + RNN architecture
train.py           # Training pipeline
test.py            # Evaluation / inference pipeline
main.py            # End-to-end execution script
requirements.txt   # Python dependencies
README.md


