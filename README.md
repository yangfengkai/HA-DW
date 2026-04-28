# The Hidden Bias in Group-Relative Reinforcement Learning


## Overview

HA-DW (History-Aware Adaptive Difficulty Weighting) is a framework for:
1. **Group-Relative Bias Analysis**: Identifying and analyzing systematic bias in group-relative advantage estimation under finite rollouts
2. **History-Aware Difficulty Estimation**: Maintaining an evolving difficulty anchor that incorporates long-term reward trends across batches
3. **Plug-and-Play Advantage Reweighting**: Correcting biased advantage signals through adaptive prompt-level weighting without changing the underlying RLVR framework

## Dataset

You can find the datasets used in this work from the following links:

- [MATH-Train](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval): This dataset contains competition-style math problems and provides rule-based ground-truth answers, making it suitable for RLVR training on reasoning tasks.

- [MATH500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500): MATH-500 contains 500 problems sampled from the MATH benchmark and was created by OpenAI in the *Let's Verify Step by Step* work. It is commonly used as a held-out benchmark for evaluating LLM reasoning performance on competition-level math problems.


### Dataset Format
The training data should be stored in Parquet format. Each row contains one prompt example with the following fields:

- `data_source`: the source of the dataset
- `prompt`: a list of chat messages, where each message contains `role` and `content`
- `ability`: the task type, e.g., `math`
- `reward_model`: reward metadata, including the ground-truth answer and reward style

An example row is shown below:

```python
{
    "data_source": "DigitalLearningGmbH/MATH-lighteval",
    "prompt": [
        {
            "role": "user",
            "content": "Let \\[f(x) = \\left\\{\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\ x-5 &\\text{ if } -2 \\le x \\le 2, \\\\ 2x-b &\\text{ if } x <-2.\\end{array}\\right.\\] Find $a+b$ if the piecewise function is continuous. Let's think step by step and output the final answer within \\boxed{}."
        }
    ],
    "ability": "math",
    "reward_model": {
        "ground_truth": "0",
        "style": "rule"
    }
}
```

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

