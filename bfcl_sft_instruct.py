import os
import re
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import transformers
import wandb

from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed, GenerationConfig
from peft import LoraConfig, PeftModel, get_peft_model
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Use the Instruct model instead of the base model
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
PROJECT_NAME = "bfcl"
HF_USER = "SArmagan"

RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Overall hyperparameters
EPOCHS = 1
BATCH_SIZE = 1  # 4
MAX_SEQUENCE_LENGTH = 1024
GRADIENT_ACCUMULATION_STEPS = 64  # 16

# QLoRA hyperparameters
LORA_R = 32
LORA_ALPHA = LORA_R * 2
ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]
TARGET_MODULES = ATTENTION_LAYERS + MLP_LAYERS
LORA_DROPOUT = 0.05

# Training hyperparameters
LEARNING_RATE = 1e-4  # 1e-4
WARMUP_RATIO = 0.03
LR_SCHEDULER_TYPE = "cosine"
WEIGHT_DECAY = 0.01
OPTIMIZER = "paged_adamw_32bit"

capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8

# Tracking
LOG_STEPS = 10
SAVE_STEPS = 20
LOG_TO_WANDB = True

# Dataset split
TRAIN_SPLIT = 0.95
SEED = 42


SYSTEM_PROMPT = (
    "You are a helpful assistant with access to functions. "
    "When the user asks a question, analyze it and decide which function(s) to call with the appropriate arguments. "
    "Respond ONLY with a JSON array of function calls in the following format:\n"
    '[{"name": "function_name", "arguments": {"arg1": "value1", "arg2": "value2"}}]\n\n'
    "If multiple function calls are needed, include all of them in the array. "
    "Do not include any other text, explanation, or commentary in your response."
)


def format_tools_for_prompt(tools_json: str) -> str:
    """Format tool definitions into a readable string for the prompt."""
    tools = json.loads(tools_json)
    tool_descriptions = []
    for tool in tools:
        name = tool["name"]
        desc = tool.get("description", "No description available.")
        params = tool.get("parameters", {})

        param_lines = []
        for param_name, param_info in params.items():
            p_type = param_info.get("type", "any")
            p_desc = param_info.get("description", "")
            p_default = param_info.get("default", None)
            default_str = f" (default: {p_default})" if p_default is not None else ""
            param_lines.append(f"    - {param_name} ({p_type}): {p_desc}{default_str}")

        params_str = "\n".join(param_lines) if param_lines else "    No parameters."
        tool_descriptions.append(f"  {name}: {desc}\n  Parameters:\n{params_str}")

    return "Available functions:\n" + "\n\n".join(tool_descriptions)


def format_example(example, tokenizer):
    """
    Build a single training string using the Instruct model's chat template.
    Returns {"text": "<full_chat_formatted_string>"} for SFTTrainer.
    """
    tools_str = format_tools_for_prompt(example["tools"])
    system_message = SYSTEM_PROMPT + "\n\n" + tools_str
    user_message = example["query"]

    try:
        answers = json.loads(example["answers"])
        assistant_response = json.dumps(answers, indent=None)
    except json.JSONDecodeError:
        assistant_response = example["answers"]

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response},
    ]

    # apply_chat_template produces the full tokenizer-native format
    # tokenize=False returns the string so SFTTrainer can handle tokenization
    text = tokenizer.apply_chat_template(messages, tokenize=False)

    return {"text": text}


def filter_by_length(dataset, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
    """
    Filter out examples whose total token count exceeds max_length.
    """
    original_size = len(dataset)

    def is_within_limit(example):
        token_count = len(tokenizer.encode(example["text"], add_special_tokens=False))
        return token_count <= max_length

    filtered_dataset = dataset.filter(
        is_within_limit,
        desc=f"Filtering examples > {max_length} tokens",
    )

    filtered_size = len(filtered_dataset)
    removed = original_size - filtered_size
    print(f"Length filter: kept {filtered_size}/{original_size} examples "
          f"(removed {removed}, {removed / original_size * 100:.1f}%)")

    return filtered_dataset


def prepare_datasets(tokenizer):
    """Load and prepare the xlam function-calling dataset."""
    ds = load_dataset("Salesforce/xlam-function-calling-60k")
    dataset = ds["train"]

    # Pass tokenizer via closure so format_example can use apply_chat_template
    dataset = dataset.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=dataset.column_names,
        desc="Formatting dataset",
    )

    print(f"Dataset size: {len(dataset)}")

    # Filter out examples that exceed the context window
    dataset = filter_by_length(dataset, tokenizer, max_length=MAX_SEQUENCE_LENGTH)

    split = dataset.train_test_split(test_size=1 - TRAIN_SPLIT, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    print("\n--- Sample ---")
    print(f"TEXT (last 400 chars): ...{train_dataset[0]['text'][-400:]}")
    print()

    return train_dataset, eval_dataset


def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map="auto"
    )

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.1f} MB")
    return model, tokenizer


def SFT_with_QLoRA():
    """Fine-tune Llama-Instruct on xlam-function-calling-60k using SFT with QLoRA."""
    set_seed(SEED)
    model, tokenizer = get_model_and_tokenizer(BASE_MODEL)

    train_dataset, eval_dataset = prepare_datasets(tokenizer=tokenizer)

    lora_parameters = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    train_parameters = SFTConfig(
        output_dir=PROJECT_RUN_NAME,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        save_steps=SAVE_STEPS,
        save_total_limit=10,
        logging_steps=LOG_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=False,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="wandb" if LOG_TO_WANDB else None,
        run_name=RUN_NAME,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_NAME,
        hub_private_repo=True,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Use the "text" field produced by format_example
        dataset_text_field="text",
    )

    fine_tuning = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_parameters,
        args=train_parameters,
    )

    torch.cuda.empty_cache()
    fine_tuning.train()
    fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
    tokenizer.push_to_hub(PROJECT_RUN_NAME, private=True)
    print(f"Saved to the hub: {PROJECT_RUN_NAME}")
    wandb.finish()


def test_inference(adapter_path: str, prompt: str, tools_json: str):
    """Quick inference test with a trained adapter."""
    model, tokenizer = get_model_and_tokenizer(BASE_MODEL)
    model = PeftModel.from_pretrained(model, adapter_path)

    tools_str = format_tools_for_prompt(tools_json)
    system_message = SYSTEM_PROMPT + "\n\n" + tools_str

    # Use the chat template for inference too
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"\nQuery: {prompt}")
    print(f"Response: {response}")
    return response


if __name__ == "__main__":
    SFT_with_QLoRA()

    # # --- Test inference with a dataset example ---
    # ADAPTER_PATH = "SArmagan/bfcl-2026-02-23_00.16.06"

    # ds = load_dataset("Salesforce/xlam-function-calling-60k")
    # example = ds["train"][0]

    # print("=" * 60)
    # print(f"Query: {example['query']}")
    # print(f"Expected: {example['answers']}")
    # print("=" * 60)

    # test_inference(
    #     adapter_path=ADAPTER_PATH,
    #     prompt=example["query"],
    #     tools_json=example["tools"],
    # )