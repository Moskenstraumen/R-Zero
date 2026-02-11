# Repository Guidelines

This codebase used to be veRL based and can use `git diff` to check previous codes.

## Project Structure & Module Organization
`customization/` contains R-Zero training customization modules: `customization/questioner_data_source.py` for questioner prompt/data sampling, `customization/reward_model.py` for questioner/solver reward functions, and shared reward logic in `customization/math.py`. `scripts/` holds the training drivers and data utilities. `question_generate/` and `question_evaluate/` handle question generation and evaluation helpers. Assets live in `figs/`; repo-level config includes `requirements.txt` and `tokens.json`.

Training wiring notes:
- `scripts/questioner_train.sh` uses `customization.questioner_data_source.QuestionerDataSource` and `customization.reward_model.questioner_rm_group`.
- `scripts/solver_train.sh` uses `customization.reward_model.solver_rm`.
- Rollout uses slime default rollout function (`slime.rollout.sglang_rollout.generate_rollout`); no repo-local rollout wrapper is used.

## Build, Test, and Development Commands
Common commands used in this repo:
```bash
pip install -r requirements.txt

export STORAGE_PATH="/path/to/storage"
export HUGGINGFACENAME="yourhuggingfacename"

# Full training loop (questioner + solver iterations + evaluation)
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b

# Generate questions on multiple GPUs
bash question_generate/question_generate.bash <model> <num_samples> <save_name>

# Run evaluation benchmarks (multi-GPU)
bash evaluation/evaluate.bash <model_path_or_name>
```
Most scripts assume `STORAGE_PATH` is set and a GPU environment is available.

## Training Data Flow & Model Saving
Entry point is `scripts/main.sh`, which runs the full loop: start solver RM server -> train questioner -> generate questions -> evaluate questions -> build solver dataset -> train solver -> repeat.

## Variable Placement Rule
- Keep shared training/inference constants in `scripts/main.sh` only (for example: GPU CSVs, shard counts, eval sampling/timeouts, and shared root paths).
- If a variable is already defined in `scripts/main.sh`, do not re-define its default in `scripts/questioner_train.sh`, `scripts/solver_train.sh`, `question_generate/question_generate.bash`, or `question_evaluate/evaluate.sh`.
- Sub-scripts should consume caller-provided values directly; `main.sh` should pass required variables when invoking sub-scripts.
- Use `export` only when a child process must read an environment variable; avoid unnecessary exports for same-script variable usage.

Data flow (paths are fixed to `/root/dataset/R-Zero` in this repo):
- `question_generate/question_generate.py` writes `/root/dataset/R-Zero/generated_question/<save_name>_<shard>.json` (default shards: 0-7).
- `question_evaluate/evaluate.py` reads those files and writes `/root/dataset/R-Zero/generated_question/<save_name>_<shard>_results.json`.
- `scripts/build_solver_jsonl.py` aggregates shards into `/root/dataset/R-Zero/solver_data/solver_current.jsonl`.
- `scripts/solver_train.sh` consumes that JSONL.

Model saving (paths are fixed to `/root/model/R-Zero`):
- Active checkpoints live in `/root/model/R-Zero/questioner_current` and `/root/model/R-Zero/solver_current`.
- Each iteration is snapshot-copied into `/root/model/R-Zero/<model_abbr>_questioner_vN` and `/root/model/R-Zero/<model_abbr>_solver_vN`.
- The solver used for questioner rewards is exposed as the symlink `/root/model/R-Zero/solver_reward_hf` (points to the current solver HF export).

## Coding Style & Naming Conventions
Python code uses 4-space indentation and mostly PEP8-style naming (snake_case functions/vars, CamelCase classes). Bash scripts follow descriptive, underscore-separated filenames (e.g., `questioner_train_penalty.sh`). No formatter or linter is enforced, so keep changes consistent with nearby code.

## Testing Guidelines
There is no dedicated unit test suite. Validation is typically done via the evaluation scripts in `evaluation/`. For quick checks, run a single dataset with `python evaluation/generate.py --model <model> --dataset math`. The full `evaluation/evaluate.bash` script requires `nvidia-smi` and multiple GPUs.

## Commit & Pull Request Guidelines
Commit history favors short, direct messages such as “Update README” or “Fix typo.” Use clear, imperative or sentence-case summaries and keep them scoped. PRs should include a concise description, key commands run, and any changes to environment variables or configs. If results change, report the relevant metrics or attach a screenshot/table.

## Security & Configuration Tips
`tokens.json` is expected to contain Hugging Face and WandB keys, and `evaluation/results_recheck.py` uses an OpenAI key. Do not commit real secrets; use placeholders and document required keys in PRs.
