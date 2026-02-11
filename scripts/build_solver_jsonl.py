import argparse
import json
import os


DEFAULT_STORAGE_PATH = "/root/dataset/R-Zero"
MIN_SCORE = 0.3
MAX_SCORE = 0.8
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_name",
        type=str,
        default=os.getenv("SAVE_NAME", ""),
        help="Base name for result shards. When empty, falls back to current_results.json.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=int(os.getenv("QUESTION_EVAL_SHARDS", 8)),
        help="Number of evaluation shards to aggregate.",
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default=os.getenv("STORAGE_PATH", DEFAULT_STORAGE_PATH),
    )
    parser.add_argument("--keep_files", action="store_true", help="Do not delete shard files after reading.")
    return parser.parse_args()


def _load_sharded_results(storage_path, save_name, num_shards, keep_files):
    data = []
    found = False
    for i in range(num_shards):
        path = os.path.join(storage_path, "generated_question", f"{save_name}_{i}_results.json")
        if not os.path.exists(path):
            continue
        found = True
        with open(path, "r", encoding="utf-8") as f:
            data.extend(json.load(f))
        if not keep_files:
            os.remove(path)
    if not found:
        raise FileNotFoundError(
            f"No shard results found for save_name={save_name} under {storage_path}/generated_question"
        )
    return data


def _load_single_result(storage_path):
    path = os.path.join(storage_path, "generated_question", "current_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = _parse_args()

    if args.save_name:
        data = _load_sharded_results(args.storage_path, args.save_name, args.num_shards, args.keep_files)
    else:
        data = _load_single_result(args.storage_path)

    output_file = os.path.join(args.storage_path, "solver_data", "solver_current.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    kept = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for item in data:
            question = item.get("question") or ""
            answer = item.get("answer") or ""
            score = float(item.get("score", -1))
            if not question or not answer:
                continue
            if score < MIN_SCORE or score > MAX_SCORE:
                continue
            record = {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "answer": answer,
                "metadata": {"score": score},
            }
            out.write(json.dumps(record, ensure_ascii=True) + "\n")
            kept += 1

    print(f"Wrote {kept} records to {output_file}")


if __name__ == "__main__":
    main()
