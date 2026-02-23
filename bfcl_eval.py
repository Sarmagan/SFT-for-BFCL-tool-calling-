import os
import re
import json
import ast
import torch
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel


REVISION = "256bd7dba947fcb6d8a784eb67da9e10272c239e"
# "cd4587bfd7aad619818693ee3598417bc7849208" # 100
# "ad848efa4da64d927d9ab26506cf2959cd3d37b8" # 80

INSTRUCT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
TEST_FILE = "BFCL_v4_multiple.json"
GT_FILE = "BFCL_v4_multiple_GT.json"
OUTPUT_FILE = "eval_results.json"
BATCH_SIZE = 1
MAX_NEW_TOKENS = 512

capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8

# Must match training script exactly
SFT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to functions. "
    "When the user asks a question, analyze it and decide which function(s) to call with the appropriate arguments. "
    "Respond ONLY with a JSON array of function calls in the following format:\n"
    '[{"name": "function_name", "arguments": {"arg1": "value1", "arg2": "value2"}}]\n\n'
    "If multiple function calls are needed, include all of them in the array. "
    "Do not include any other text, explanation, or commentary in your response."
)


def load_quantized_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=quant_config
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def load_jsonl(path: str):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# =============================================================================
# Prompt Building
# =============================================================================

def build_prompt_instruct(sample: dict, tokenizer) -> str:
    """Original instruct-style prompt using chat template."""
    functions = sample["function"]
    messages = sample["question"][0]
    func_descriptions = [json.dumps(func, indent=2) for func in functions]
    system_message = (
        "You are a helpful assistant with access to the following functions. "
        "When the user asks a question, determine which function to call and respond "
        "with a JSON object (or list of JSON objects) in the following format:\n"
        '{"name": "<function_name>", "arguments": {<arg_name>: <arg_value>, ...}}\n\n'
        "Available functions:\n" + "\n\n".join(func_descriptions) + "\n\n"
        "Only respond with the JSON function call(s). Do not include any other text."
    )
    chat_messages = [{"role": "system", "content": system_message}]
    for msg in messages:
        chat_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        return tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = ["<|begin_of_text|>"]
        for msg in chat_messages:
            parts.append(f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)


def build_prompt_sft(sample: dict, tokenizer) -> str:
    """Plain-text prompt matching the SFT training format exactly. No special tokens."""
    functions = sample["function"]
    messages = sample["question"][0]

    tool_lines = []
    for func in functions:
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        properties = params.get("properties", params)

        param_parts = []
        for p_name, p_info in properties.items():
            if p_name in ("type", "required", "additionalProperties"):
                continue
            if isinstance(p_info, dict):
                p_type = p_info.get("type", "any")
                p_desc = p_info.get("description", "")
                p_default = p_info.get("default", None)
                default_str = f" (default: {p_default})" if p_default is not None else ""
                param_parts.append(f"    - {p_name} ({p_type}): {p_desc}{default_str}")

        params_str = "\n".join(param_parts) if param_parts else "    No parameters."
        tool_lines.append(f"  {name}: {desc}\n  Parameters:\n{params_str}")

    tools_str = "Available functions:\n" + "\n\n".join(tool_lines)
    system_message = SFT_SYSTEM_PROMPT + "\n\n" + tools_str

    # Combine all user messages into one
    user_text = "\n".join(msg["content"] for msg in messages if msg["role"] == "user")

    # Plain-text format — must match training exactly
    return (
        f"System: {system_message}\n"
        f"User: {user_text}\n"
        f"Assistant: "
    )


# =============================================================================
# Parsing & Normalization
# =============================================================================

def deep_parse(val):
    if isinstance(val, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(val)
                if parsed != val:
                    return deep_parse(parsed)
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
        return val
    if isinstance(val, (list, tuple)):
        return [deep_parse(v) for v in val]
    if isinstance(val, dict):
        return {k: deep_parse(v) for k, v in val.items()}
    return val


def values_equal(a, b, tol=1e-6) -> bool:
    a, b = deep_parse(a), deep_parse(b)
    if isinstance(a, bool) and isinstance(b, bool):
        return a == b
    if isinstance(a, bool) or isinstance(b, bool):
        truthy = {True, "true", "True", 1, "1"}
        falsy = {False, "false", "False", 0, "0"}
        return (a in truthy and b in truthy) or (a in falsy and b in falsy)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < tol
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(values_equal(a[k], b[k], tol) for k in a)
    try:
        return abs(float(a) - float(b)) < tol
    except (ValueError, TypeError):
        pass
    return str(a).strip().lower() == str(b).strip().lower()


def parse_function_calls(text: str) -> list[dict]:
    text = text.strip()
    calls = []
    try:
        parsed = json.loads(text)
        parsed = parsed if isinstance(parsed, list) else [parsed]
        return [
            {"name": c["name"], "arguments": deep_parse(c.get("arguments", {}))}
            for c in parsed
            if isinstance(c, dict) and "name" in c
        ]
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    i = 0
    while i < len(text):
        if text[i] == '{':
            depth, start = 0, i
            found = False
            while i < len(text):
                if text[i] == '{': depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:i+1])
                            if "name" in obj:
                                calls.append(obj)
                        except json.JSONDecodeError:
                            pass
                        found = True
                        break
                i += 1
            if not found and depth > 0:
                candidate = text[start:] + ("}" * depth)
                try:
                    obj = json.loads(candidate)
                    if "name" in obj:
                        calls.append(obj)
                except json.JSONDecodeError:
                    pass
        i += 1
    return [{"name": c["name"], "arguments": deep_parse(c.get("arguments", {}))} for c in calls]


# =============================================================================
# Matching
# =============================================================================

def arg_matches(predicted, acceptable_values: list) -> bool:
    if predicted is None:
        return "" in acceptable_values
    predicted = deep_parse(predicted)
    for gt_val in acceptable_values:
        if gt_val == "":
            continue
        if values_equal(predicted, gt_val):
            return True
        if isinstance(predicted, dict) and isinstance(gt_val, dict):
            if set(predicted.keys()) == set(gt_val.keys()):
                if all(
                    arg_matches(predicted[k], gt_val[k]) if isinstance(gt_val[k], list)
                    else values_equal(predicted[k], gt_val[k])
                    for k in gt_val
                ):
                    return True
        if isinstance(predicted, list) and isinstance(gt_val, list):
            if len(predicted) == len(gt_val):
                used = set()
                all_found = True
                for p in predicted:
                    found = False
                    for j, g in enumerate(gt_val):
                        if j not in used and values_equal(p, g):
                            used.add(j)
                            found = True
                            break
                    if not found:
                        all_found = False
                        break
                if all_found:
                    return True
    return False


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_single(predicted_calls: list[dict], gt_calls: list[dict]) -> dict:
    if not predicted_calls and not gt_calls:
        return {"function_name_match": True, "required_args_match": True, "all_args_match": True, "details": []}

    matched_gt = set()
    call_results = []

    for pred in predicted_calls:
        pred_name = pred["name"]
        pred_args = pred["arguments"]
        best_idx, best_score = None, -1

        for i, gt in enumerate(gt_calls):
            if i in matched_gt:
                continue
            gt_name = list(gt.keys())[0]
            if pred_name == gt_name:
                score = sum(1 for a, v in gt[gt_name].items() if arg_matches(pred_args.get(a), v))
                if score > best_score:
                    best_score, best_idx = score, i

        if best_idx is not None:
            matched_gt.add(best_idx)
            gt = gt_calls[best_idx]
            gt_name = list(gt.keys())[0]
            gt_args = gt[gt_name]

            arg_results = {}
            for arg_name, acceptable in gt_args.items():
                matched = arg_matches(pred_args.get(arg_name), acceptable)
                arg_results[arg_name] = {
                    "predicted": pred_args.get(arg_name),
                    "expected": acceptable,
                    "match": matched,
                }

            call_results.append({
                "function_name": gt_name, "name_match": True,
                "args": arg_results,
                "all_args_match": all(a["match"] for a in arg_results.values()),
                "extra_args": list(set(pred_args.keys()) - set(gt_args.keys())),
            })
        else:
            call_results.append({"function_name": pred_name, "name_match": False, "args": {}, "all_args_match": False})

    for i, gt in enumerate(gt_calls):
        if i not in matched_gt:
            call_results.append({"function_name": list(gt.keys())[0], "name_match": False, "args": {}, "all_args_match": False, "missed": True})

    count_ok = len(predicted_calls) == len(gt_calls)
    fname_match = count_ok and all(c["name_match"] for c in call_results)
    all_match = fname_match and all(c["all_args_match"] for c in call_results)

    req_match = fname_match
    if fname_match:
        for cr in call_results:
            for a_name, a_info in cr.get("args", {}).items():
                if "" not in a_info["expected"] and not a_info["match"]:
                    req_match = False

    return {
        "function_name_match": fname_match,
        "required_args_match": req_match,
        "all_args_match": all_match,
        "details": call_results,
    }


# =============================================================================
# Main
# =============================================================================

def run_evaluation(
    model_name=BASE_MODEL, adapter_path=None, prompt_style="instruct",
    test_file=TEST_FILE, gt_file=GT_FILE, output_file=OUTPUT_FILE,
    batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS,
):
    print(f"Loading model: {model_name}")
    model, tokenizer = load_quantized_model(model_name)

    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, revision=REVISION)
        # model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    test_data = load_jsonl(test_file) # [:5]
    gt_data = load_jsonl(gt_file)# [:5]
    gt_by_id = {item["id"]: item["ground_truth"] for item in gt_data}

    samples = []
    for s in test_data:
        if s["id"] in gt_by_id:
            if prompt_style == "sft":
                prompt_text = build_prompt_sft(s, tokenizer)
            else:
                prompt_text = build_prompt_instruct(s, tokenizer)
            samples.append({"id": s["id"], "prompt_text": prompt_text, "gt": gt_by_id[s["id"]]})

    print(f"Evaluating {len(samples)} samples (prompt_style={prompt_style})...")
    all_results = []

    # Build EOS token list based on prompt style
    if prompt_style == "instruct":
        eos_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
    else:
        # Plain-text format: only use the standard EOS token
        eos_ids = [tokenizer.eos_token_id]

    for batch_start in tqdm(range(0, len(samples), batch_size)):
        batch = samples[batch_start:batch_start + batch_size]
        prompts = [s["prompt_text"] for s in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=eos_ids,
            )

        for i, s in enumerate(batch):
            input_len = inputs["input_ids"][i].shape[0]
            gen_text = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
            try:
                pred_calls = parse_function_calls(gen_text)
                eval_result = evaluate_single(pred_calls, s["gt"])
            except Exception as e:
                print(f"\n[WARNING] Error evaluating sample {s['id']}: {e}")
                pred_calls = []
                eval_result = {
                    "function_name_match": False,
                    "required_args_match": False,
                    "all_args_match": False,
                    "details": [],
                    "error": str(e),
                }
            all_results.append({
                "id": s["id"], "generated_text": gen_text,
                "predicted_calls": pred_calls, "ground_truth": s["gt"],
                "evaluation": eval_result,
            })

    # Metrics
    n = len(all_results)
    fname_acc = sum(1 for r in all_results if r["evaluation"]["function_name_match"]) / n if n else 0
    req_acc = sum(1 for r in all_results if r["evaluation"]["required_args_match"]) / n if n else 0
    all_acc = sum(1 for r in all_results if r["evaluation"]["all_args_match"]) / n if n else 0

    total_args = sum(1 for r in all_results for c in r["evaluation"]["details"] for _ in c.get("args", {}))
    matched_args = sum(1 for r in all_results for c in r["evaluation"]["details"] for a in c.get("args", {}).values() if a["match"])
    arg_acc = matched_args / total_args if total_args else 0

    print(f"\n{'='*50}")
    print(f"Model:              {model_name}")
    if adapter_path:
        print(f"Adapter:            {adapter_path}")
    print(f"Prompt style:       {prompt_style}")
    print(f"Samples:            {n}")
    print(f"Function name acc:  {fname_acc:.2%}")
    print(f"Required args acc:  {req_acc:.2%}")
    print(f"All args acc:       {all_acc:.2%}")
    print(f"Arg-level acc:      {arg_acc:.2%} ({matched_args}/{total_args})")
    print(f"{'='*50}")

    output = {
        "summary": {
            "model": model_name, "adapter": adapter_path, "prompt_style": prompt_style,
            "timestamp": datetime.now().isoformat(), "total_samples": n,
            "function_name_accuracy": round(fname_acc, 4),
            "required_args_accuracy": round(req_acc, 4),
            "all_args_accuracy": round(all_acc, 4),
            "arg_level_accuracy": round(arg_acc, 4),
        },
        "results": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=INSTRUCT_MODEL)
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    p.add_argument("--prompt-style", choices=["instruct", "sft"], default="instruct",
                    help="instruct = chat template for Instruct model, sft = plain-text for base+adapter")
    p.add_argument("--test-file", default=TEST_FILE)
    p.add_argument("--gt-file", default=GT_FILE)
    p.add_argument("--output-file", default=OUTPUT_FILE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = p.parse_args()
    run_evaluation(args.model, args.adapter, args.prompt_style, args.test_file, args.gt_file, args.output_file, args.batch_size, args.max_new_tokens)