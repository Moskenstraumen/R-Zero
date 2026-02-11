import argparse
import json
import os
import regex as re
import requests
from transformers import AutoTokenizer


DEFAULT_SGLANG_URL = "http://127.0.0.1:13200/generate"
DEFAULT_MODEL_PATH = "/root/model/R-Zero/questioner_current/hf/rollout_0"
DEFAULT_STORAGE_PATH = "/root/dataset/R-Zero"
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_BATCH_SIZE = 64


def extract_boxed(text):
    results, i = [], 0
    prefix = r"\boxed{"
    plen = len(prefix)

    while True:
        start = text.find(prefix, i)
        if start == -1:
            break

        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1

        results.append(text[start + plen : j - 1])
        i = j

    return results


def build_prompt(tokenizer):
    chat = [
        {
            "role": "system",
            "content": (
                "You are an expert competition-math problem setter.\n"
                "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                "The problem could come from any field of mathematics, including but not limited to algebra, geometry, "
                "number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                "Avoid re-using textbook cliches or famous contest problems.\n"
                "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                "<question>\n"
                "{The full problem statement on one or more lines}\n"
                "</question>\n\n"
                r"\boxed{final_answer}"
                "\n\n"
                "Do NOT output anything else--no explanations, no extra markup."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate one new, challenging reasoning question now. "
                "Remember to format the output exactly as instructed."
            ),
        },
    ]

    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
    return "system: " + chat[0]["content"] + "\n" + "user: " + chat[1]["content"]


def batched_generate(sglang_url, prompts, sampling_params):
    payload = {"text": prompts, "sampling_params": sampling_params}
    res = requests.post(sglang_url, json=payload, timeout=600)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        default=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
        help="HF path used for tokenization.",
    )
    parser.add_argument("--num_samples", type=int, default=int(os.getenv("QUESTION_GEN_SAMPLES", DEFAULT_NUM_SAMPLES)))
    parser.add_argument("--suffix", type=str, default="", help="Shard suffix (typically GPU index).")
    parser.add_argument(
        "--save_name",
        type=str,
        default=os.getenv("SAVE_NAME", "current"),
        help="Base name for output files.",
    )
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("QUESTION_GEN_BATCH_SIZE", DEFAULT_BATCH_SIZE)))
    parser.add_argument(
        "--sglang-url",
        dest="sglang_url",
        default=os.getenv("SGLANG_URL", DEFAULT_SGLANG_URL),
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = build_prompt(tokenizer)
    sampling_params = {
        "max_new_tokens": 4096,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": -1,
        "stop_token_ids": [tokenizer.eos_token_id],
        "n": 1,
        "skip_special_tokens": True,
        "no_stop_trim": True,
    }

    results = []
    remaining = args.num_samples
    while remaining > 0:
        batch = min(args.batch_size, remaining)
        prompts = [prompt] * batch
        outputs = batched_generate(args.sglang_url, prompts, sampling_params)
        for response in outputs:
            try:
                questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
                answers = extract_boxed(response)
                if questions and answers:
                    question = questions[-1].strip()
                    answer = answers[-1].strip()
                    results.append({"question": question, "answer": answer, "score": 0})
                else:
                    results.append({"question": response, "answer": "", "score": -1})
            except Exception:
                results.append({"question": response, "answer": "", "score": -1})
        remaining -= batch

    storage_path = os.getenv("STORAGE_PATH", DEFAULT_STORAGE_PATH)
    os.makedirs(os.path.join(storage_path, "generated_question"), exist_ok=True)
    suffix = f"_{args.suffix}" if args.suffix != "" else ""
    output_path = os.path.join(storage_path, "generated_question", f"{args.save_name}{suffix}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
