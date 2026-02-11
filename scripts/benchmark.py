#!/usr/bin/env python3

import argparse
import json
import os
import re
import time

import requests
from mathruler.grader import extract_boxed_content, grade_answer


DEFAULT_SOLVER_URL = "http://127.0.0.1:15250/generate"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_REQUEST_TIMEOUT = 600
DEFAULT_REQUEST_RETRIES = 0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_REQUEST_RETRIES = 3

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a solver on a local JSONL benchmark.")
    parser.add_argument("--model-path", required=True, help="Model path/name for bookkeeping in outputs.")
    parser.add_argument("--dataset-path", required=True, help="Path to JSONL benchmark file.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output JSONL path. Defaults to $STORAGE_PATH/evaluation/<model>_results.jsonl",
    )
    parser.add_argument(
        "--solver-url",
        default=os.getenv("BENCHMARK_SOLVER_URL", DEFAULT_SOLVER_URL),
        help="SGLang /generate endpoint.",
    )
    parser.add_argument("--prompt-key", default="prompt", help="Prompt column name in input JSONL.")
    parser.add_argument("--label-key", default="label", help="Label column name in input JSONL.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--request-retries", type=int, default=DEFAULT_REQUEST_RETRIES)
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of examples (0 means all).")
    return parser.parse_args()


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def sanitize_name(value):
    cleaned = value.replace("/", "_").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)


def extract_prediction(response_text):
    response = normalize_text(response_text)
    if not response:
        return ""

    try:
        boxed = extract_boxed_content(response)
        if isinstance(boxed, str) and boxed.strip():
            return boxed.strip()
        if isinstance(boxed, list):
            for item in boxed:
                item_text = normalize_text(item)
                if item_text:
                    return item_text
    except Exception:
        pass

    answer_match = re.search(r"(?im)^\s*Answer\s*:\s*(.+?)\s*$", response)
    if answer_match:
        return normalize_text(answer_match.group(1))

    return ""


def safe_grade(prediction, label):
    pred_text = normalize_text(prediction)
    label_text = normalize_text(label)
    if not pred_text or not label_text:
        return False

    try:
        return bool(grade_answer(pred_text, label_text))
    except Exception:
        return pred_text == label_text


def load_jsonl_rows(dataset_path, prompt_key, label_key, limit=0):
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc

            if prompt_key not in item or label_key not in item:
                raise ValueError(
                    f"Line {line_number} missing required keys: '{prompt_key}' and '{label_key}'"
                )

            rows.append(item)
            if limit > 0 and len(rows) >= limit:
                break

    if not rows:
        raise ValueError(f"No usable rows found in {dataset_path}")
    return rows


def parse_texts_from_response(response_json, expected_count):
    texts = []
    if isinstance(response_json, dict):
        raw_texts = response_json.get("text", [])
        if isinstance(raw_texts, str):
            texts = [raw_texts]
        elif isinstance(raw_texts, list):
            texts = [text if isinstance(text, str) else "" for text in raw_texts]
    elif isinstance(response_json, list):
        for item in response_json:
            if isinstance(item, dict):
                text = item.get("text", "")
                texts.append(text if isinstance(text, str) else "")
            elif isinstance(item, str):
                texts.append(item)
            else:
                texts.append("")
    else:
        raise ValueError(f"Unexpected response type from SGLang: {type(response_json).__name__}")

    if len(texts) < expected_count:
        texts.extend([""] * (expected_count - len(texts)))
    elif len(texts) > expected_count:
        texts = texts[:expected_count]
    return texts


def generate_batch(solver_url, prompts, sampling_params, request_timeout, request_retries):
    payload = {"text": prompts, "sampling_params": sampling_params}
    last_error = None

    for attempt in range(request_retries + 1):
        try:
            response = requests.post(solver_url, json=payload, timeout=request_timeout)
            response.raise_for_status()
            return parse_texts_from_response(response.json(), len(prompts))
        except Exception as exc:
            last_error = exc
            if attempt < request_retries:
                time.sleep(1)

    raise RuntimeError(f"Failed to get generation response after retries: {last_error}")


def default_output_path(model_path):
    storage_path = os.getenv("STORAGE_PATH", "/root/dataset/R-Zero")
    output_dir = os.path.join(storage_path, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{sanitize_name(model_path)}_results.jsonl")


def main():
    args = parse_args()

    output_path = args.output_path or default_output_path(args.model_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = load_jsonl_rows(
        dataset_path=args.dataset_path,
        prompt_key=args.prompt_key,
        label_key=args.label_key,
        limit=args.limit,
    )

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "skip_special_tokens": True,
        "no_stop_trim": True,
    }

    total = len(rows)
    num_correct = 0
    start = time.time()

    with open(output_path, "w", encoding="utf-8") as fout:
        for start_idx in range(0, total, args.batch_size):
            chunk = rows[start_idx:start_idx + args.batch_size]
            prompts = [normalize_text(item[args.prompt_key]) for item in chunk]
            responses = generate_batch(
                solver_url=args.solver_url,
                prompts=prompts,
                sampling_params=sampling_params,
                request_timeout=args.request_timeout,
                request_retries=args.request_retries,
            )

            for item, response_text in zip(chunk, responses):
                label = normalize_text(item[args.label_key])
                prediction = extract_prediction(response_text)
                correct = safe_grade(prediction, label)
                if correct:
                    num_correct += 1

                record = {
                    "model_path": args.model_path,
                    "label": label,
                    "prediction": prediction,
                    "correct": correct,
                    "response": response_text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            processed = min(start_idx + args.batch_size, total)
            elapsed = time.time() - start
            print(f"Processed {processed}/{total} | elapsed={elapsed:.1f}s")

    accuracy = num_correct / total if total else 0.0
    elapsed = time.time() - start
    print(f"Accuracy: {num_correct}/{total} = {accuracy:.4%}")
    print(f"Saved results to: {output_path}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
