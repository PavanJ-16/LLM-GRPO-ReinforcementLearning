# LLM GRPO Reinforcement Learning

This repository contains a Jupyter Notebook (`grpo_thinking.ipynb`) for training a Large Language Model (LLM) to "think" or reason using **Group Relative Policy Optimization (GRPO)**.

It uses [Unsloth](https://github.com/unslothai/unsloth) for efficient 4-bit training and is designed to run on a free **Google Colab T4 GPU** instance.

## Overview

The notebook trains a base Instruct model (e.g., `Qwen2.5-3B-Instruct`) to generate reasoning traces enclosed in `<think>` tags before providing the final answer. It uses the `open-r1/DAPO-Math-17k-Processed` dataset and reinforces correct behavior using two reward functions:
1.  **Format Reward**: Ensures the model uses the `<think>...</think>` structure.
2.  **Correctness Reward**: Verifies the final numerical answer is correct.

## Key Hyperparameters

The training configuration in the notebook is optimized for a T4 GPU (16GB VRAM). Here is a description of the key parameters:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `lora_rank` | `32` | The rank of the LoRA adapters. Higher rank (e.g., 64, 128) allows the model to learn more complex behaviors ("smarter") but increases VRAM usage and training time. `32` is a balanced default. |
| `gpu_memory_utilization` | `0.6` | Limits the memory Unsloth reserves during loading. `0.6` (60%) leaves room for activations and vLLM during the generation phase. |
| `learning_rate` | `5e-6` | The step size for the optimizer. A lower rate helps fine-tune without destroying the pre-trained knowledge. |
| `num_generations` | `4` | **Crucial for GRPO**. This is the number of alternative outputs generated for *each* prompt. GRPO compares these outputs against the group average to calculate advantages. |
| `gradient_accumulation_steps` | `4` | Accumulates gradients over 4 steps. In this setup, it effectively matches `num_generations` to ensure stable updates. |
| `per_device_train_batch_size` | `1` | Kept at 1 to minimize memory usage, relying on gradient accumulation for the effective batch size. |
| `max_completion_length` | `200` | The maximum number of tokens the model can generate during the RL phase. Kept low to ensure training is fast on the T4. Increase this if you want longer reasoning chains. |
| `beta` | `0.1` | (Implicit in GRPO) The KL divergence penalty coefficient. Controls how much the model effectively deviates from the reference model. |

## Requirements

- **vLLM**: Version `0.9.2` is required for T4 compatibility.
- **Unsloth**: Latest version via `uv` install.

## How to Run

1.  Upload `grpo_thinking.ipynb` to Google Colab.
2.  Set the Runtime to **T4 GPU**.
3.  Run all cells.
