#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
import requests
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer


DEFAULT_STORAGE_PATH = "/root/dataset/R-Zero"
DEFAULT_SOLVER_URL = "http://127.0.0.1:15200/generate"
DEFAULT_MODEL_PATH = "/root/model/R-Zero/solver_reward_hf"
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_REQUEST_TIMEOUT = 600
DEFAULT_REQUEST_RETRIES = 2
DEFAULT_PROGRESS_EVERY = 20

try:
    import stopit

    @stopit.threading_timeoutable(default="TIMED_OUT")
    def grade_answer_with_timeout(res1, res2):
        return grade_answer(res1, res2)

    STOPIT_AVAILABLE = True
except Exception:
    STOPIT_AVAILABLE = False

    def grade_answer_with_timeout(res1, res2):
        return grade_answer(res1, res2)


def build_prompt(tokenizer, question):
    chat = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": question},
    ]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True)
    return "system: " + chat[0]["content"] + "\n" + "user: " + chat[1]["content"]


def sglang_generate_batch(solver_url, prompts, sampling_params, request_timeout):
    payload = {"text": prompts, "sampling_params": sampling_params}
    res = requests.post(solver_url, json=payload, timeout=request_timeout)
    res.raise_for_status()
    obj = res.json()
    texts = []
    if isinstance(obj, dict):
        raw_texts = obj.get("text", [])
        if isinstance(raw_texts, str):
            texts = [raw_texts]
        elif isinstance(raw_texts, list):
            texts = [text for text in raw_texts if isinstance(text, str)]
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                text = item.get("text", "")
                if isinstance(text, str):
                    texts.append(text)
            elif isinstance(item, str):
                texts.append(item)
    else:
        raise ValueError(f"Unexpected response type from SGLang: {type(obj).__name__}")

    if len(texts) < len(prompts):
        texts.extend([""] * (len(prompts) - len(texts)))
    elif len(texts) > len(prompts):
        texts = texts[: len(prompts)]
    return texts


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated questions using SGLang.")
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model_path",
        default=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
        help="HF path used for tokenization.",
    )
    parser.add_argument("--num_samples", type=int, default=int(os.getenv("QUESTION_EVAL_NUM_SAMPLES", 9)))
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=int(os.getenv("QUESTION_EVAL_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)),
    )
    parser.add_argument(
        "--request-timeout",
        dest="request_timeout",
        type=int,
        default=int(os.getenv("QUESTION_EVAL_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)),
    )
    parser.add_argument(
        "--request-retries",
        dest="request_retries",
        type=int,
        default=int(os.getenv("QUESTION_EVAL_REQUEST_RETRIES", DEFAULT_REQUEST_RETRIES)),
    )
    parser.add_argument(
        "--progress-every",
        dest="progress_every",
        type=int,
        default=int(os.getenv("QUESTION_EVAL_PROGRESS_EVERY", DEFAULT_PROGRESS_EVERY)),
    )
    parser.add_argument("--suffix", type=str, default="0", help="Shard suffix (typically GPU index).")
    parser.add_argument("--save_name", type=str, required=True, help="Base name for input/output files.")
    parser.add_argument(
        "--solver-url",
        dest="solver_url",
        default=os.getenv("RZERO_SOLVER_RM_URL")
        or os.getenv("SOLVER_SGLANG_URL")
        or os.getenv("SOLVER_URL")
        or DEFAULT_SOLVER_URL,
    )
    return parser.parse_args()


def _normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_grade(lhs, rhs):
    lhs_text = _normalize_text(lhs)
    rhs_text = _normalize_text(rhs)
    if not lhs_text or not rhs_text:
        return False

    try:
        if STOPIT_AVAILABLE:
            matched = grade_answer_with_timeout(lhs_text, rhs_text, timeout=10)
            return matched != "TIMED_OUT" and bool(matched)
        return bool(grade_answer_with_timeout(lhs_text, rhs_text))
    except Exception:
        return False


def _safe_remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


def main():
    args = _parse_args()

    storage_path = os.getenv("STORAGE_PATH", DEFAULT_STORAGE_PATH)
    input_file = os.path.join(storage_path, "generated_question", f"{args.save_name}_{args.suffix}.json")
    output_file = os.path.join(storage_path, "generated_question", f"{args.save_name}_{args.suffix}_results.json")

    print(f"[{args.suffix}] Loading data from: {input_file}")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[{args.suffix}] ERROR: Input file not found. Exiting.")
        return

    correct_data = [item for item in data if item.get("score") == 0]
    if not correct_data:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        _safe_remove_file(input_file)
        return

    questions, answers = [], []
    for item in correct_data:
        question = _normalize_text(item.get("question"))
        answer = _normalize_text(item.get("answer"))
        if not question or not answer:
            continue
        questions.append(question)
        answers.append(answer)

    if not questions:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        _safe_remove_file(input_file)
        return

    print(f"[{args.suffix}] Initializing tokenizer for model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 40,
        "stop_token_ids": [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [],
        "n": 1,
        "skip_special_tokens": True,
        "no_stop_trim": True,
    }

    total_questions = len(questions)
    print(f"[{args.suffix}] Questions to evaluate: {total_questions}")
    print(
        f"[{args.suffix}] Generating {args.num_samples} samples/question "
        f"(max_new_tokens={args.max_new_tokens}, request_timeout={args.request_timeout}s, retries={args.request_retries})..."
    )

    start_time = time.time()
    results_all = []
    for idx, (question, golden_answer) in enumerate(zip(questions, answers), start=1):
        try:
            prompt = build_prompt(tokenizer, question)
            prompts = [prompt] * args.num_samples
            outputs = None
            for attempt in range(args.request_retries + 1):
                try:
                    outputs = sglang_generate_batch(
                        args.solver_url,
                        prompts,
                        sampling_params,
                        request_timeout=args.request_timeout,
                    )
                    break
                except requests.RequestException as exc:
                    if attempt >= args.request_retries:
                        raise
                    print(
                        f"[{args.suffix}] WARN request failed ({attempt + 1}/{args.request_retries + 1}), retrying: {exc}"
                    )
                    time.sleep(min(2 ** attempt, 5))

            results = [_normalize_text(extract_boxed_content(output)) for output in outputs]
            results = [res for res in results if res and res.lower() != "none"]

            if results:
                answer_counts = {}
                for result in results:
                    matched = False
                    for existing_answer in list(answer_counts.keys()):
                        if result == existing_answer or ("no " in result.lower() and "no " in existing_answer.lower()):
                            answer_counts[existing_answer] += 1
                            matched = True
                            break

                        if _safe_grade(result, existing_answer):
                            answer_counts[existing_answer] += 1
                            matched = True
                            break

                        if _safe_grade(existing_answer, result):
                            answer_counts[existing_answer] += 1
                            matched = True
                            break

                    if not matched:
                        answer_counts[result] = 1

                if answer_counts:
                    majority_answer = max(answer_counts, key=answer_counts.get)
                    max_count = answer_counts[majority_answer]
                    score = max_count / len(results)

                    if "证明" not in question and "box" not in question.lower() and "text" not in majority_answer.lower():
                        results_all.append(
                            {
                                "question": question,
                                "answer": majority_answer,
                                "score": score,
                                "results": results,
                            }
                        )
        except requests.RequestException as exc:
            print(f"[{args.suffix}] ERROR: solver request failed at {args.solver_url}: {exc}")
            break

        except Exception as exc:
            print(f"[{args.suffix}] CRITICAL ERROR processing question '{question[:50]}...': {exc}")
            continue

        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == total_questions):
            elapsed = max(time.time() - start_time, 1e-6)
            avg_sec_per_question = elapsed / idx
            eta_sec = avg_sec_per_question * (total_questions - idx)
            print(
                f"[{args.suffix}] Progress {idx}/{total_questions} | kept={len(results_all)} | "
                f"elapsed={elapsed / 60:.1f}m | eta={eta_sec / 60:.1f}m",
                flush=True,
            )

    print(f"[{args.suffix}] Processed {len(results_all)} questions. Saving results to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=4)

    _safe_remove_file(input_file)


if __name__ == "__main__":
    main()
