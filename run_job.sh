#!/bin/bash
#SBATCH -N 12              # Request 1 node
#SBATCH -p GPU-shared       # Specify GPU shared partition
#SBATCH -t 24:00:00         # Time limit of 24 hours
#SBATCH --gpus=v100-32:4    # Request 4 v100 GPUs (adjust according to availability)
#SBATCH --output=<output_path>  # Path to store output logs

# Activate conda environment
# source ~/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate minitorch

# pip install pycuda

# pip install -r requirements.extra.txt
# pip install -r requirements.txt
# pip install -e .

# Check GPU availability
# nvidia-smi

# # Change directory to the project directory
# cd $PROJECT/project/llmsys-project-flashattn

bash compile_cuda.sh

python -m pytest tests/test_flash_attention.py

# python tests/speed_test_flash_attention.py

# python kernel_tests/test_softmax_fw.py

