# The Hidden Bias in Group-Relative Reinforcement Learning


## Overview

HA-DW (History-Aware Adaptive Difficulty Weighting) is a framework for:
1. **Group-Relative Bias Analysis**: Identifying and analyzing systematic bias in group-relative advantage estimation under finite rollouts
2. **History-Aware Difficulty Estimation**: Maintaining an evolving difficulty anchor that incorporates long-term reward trends across batches
3. **Plug-and-Play Advantage Reweighting**: Correcting biased advantage signals through adaptive prompt-level weighting without changing the underlying RLVR framework

## Model

Please download the base model from Hugging Face:

https://huggingface.co/Qwen/Qwen3-4B-Base

Then set the model path in the training script:

```bash
MODEL_PATH=/path/to/Qwen3-4B-Base
```

## Installation

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd HA-DW

# Install dependencies
pip install -r requirements.txt
```

## Training

Please refer to the `hadw_trainer` in the `examples` folder for training using GRPO and GSPO with HA-DW. To quickly start training, you can use the following command as an example:

```bash
bash examples/hadw_trainer/grpo_hadw.sh
```

