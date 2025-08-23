# Setup

To get started, first run `data_preprocess.sh`. This script will:
1. Set up the required environment, and  
2. Download and preprocess the necessary datasets.

# Run Generation

Execute `run_qwen.sh` to perform Optical Music Recognition (OMR) using the pretrained Qwen-VL model. This process converts music score images into ABC notation.

# Convert Ground Truth

Run `ground_truth_convert.sh` to convert the ground truth XML files into ABC notation format.

# Evaluation

Run the following command to evaluate the results:

```bash
python evaluate.py