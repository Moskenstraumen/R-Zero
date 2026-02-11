"""R-Zero slime hooks: questioner data source + reward functions.

These mirror the veRL questioner/solver setup while running under slime.
"""

from __future__ import annotations

import os
import sys
import importlib.util
from typing import Any

import asyncio
import re

from slime.utils.http_utils import post

from slime.rollout.data_source import DataSource
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from mathruler.grader import extract_boxed_content, grade_answer

_QUESTIONER_SYSTEM_PROMPT = (
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
    "\\boxed{final_answer}\n\n"
    "Do NOT output anything else--no explanations, no extra markup."
)

_QUESTIONER_USER_PROMPT = (
    "Generate one new, challenging reasoning question now. "
    "Remember to format the output exactly as instructed."
)


def _ensure_rzero_on_path() -> str:
    rzero_root = os.environ.get("RZERO_ROOT", "/root/R-Zero")
    if rzero_root not in sys.path:
        sys.path.insert(0, rzero_root)
    return rzero_root


def _build_questioner_messages(args) -> list[dict[str, str]]:
    system_prompt = getattr(args, "questioner_system_prompt", None) or _QUESTIONER_SYSTEM_PROMPT
    user_prompt = getattr(args, "questioner_user_prompt", None) or _QUESTIONER_USER_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_questioner_prompt(args) -> str:
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    messages = _build_questioner_messages(args)
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


class QuestionerDataSource(DataSource):
    """Data source that repeatedly yields the fixed questioner prompt."""

    def __init__(self, args):
        self.args = args
        self.sample_group_index = 0
        self.sample_index = 0
        self._prompt = _build_questioner_prompt(args)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        samples: list[list[Sample]] = []
        for _ in range(num_samples):
            group: list[Sample] = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = Sample(prompt=self._prompt)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        return None

    def save(self, rollout_id):
        return None

    def load(self, rollout_id=None):
        return None


def _extract_solver_label(sample: Sample) -> str | None:
    if sample.label is not None:
        return str(sample.label)
    if isinstance(sample.metadata, dict):
        for key in ("answer", "label", "ground_truth"):
            if key in sample.metadata and sample.metadata[key] is not None:
                return str(sample.metadata[key])
    return None


async def questioner_rm(args, sample: Sample, **kwargs: Any) -> float:
    rewards = await questioner_rm_group(args, [sample])
    return rewards[0] if rewards else -1.0


async def solver_rm(args, sample: Sample, **kwargs: Any) -> float:
    """Solver reward via the original veRL math reward."""
    spec = importlib.util.spec_from_file_location(
        "rzero_math_reward", "/root/R-Zero/customization/math.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compute_score = module.compute_score

    label = _extract_solver_label(sample)
    if label is None:
        return 0.0
    scores = compute_score([sample.response], [label])
    if not scores:
        return 0.0
    return float(scores[0].get("overall", 0.0))


def _get_solver_url(args) -> str:
    return (
        os.environ.get("RZERO_SOLVER_RM_URL")
        or os.environ.get("SOLVER_SGLANG_URL")
        or getattr(args, "rm_url", None)
        or ""
    )


def _get_solver_hf_checkpoint(args) -> str | None:
    return (
        os.environ.get("RZERO_SOLVER_HF_CHECKPOINT")
        or os.environ.get("SOLVER_HF_CHECKPOINT")
        or getattr(args, "hf_checkpoint", None)
    )


_SOLVER_TOKENIZER = None


def _get_solver_tokenizer(args):
    global _SOLVER_TOKENIZER
    if _SOLVER_TOKENIZER is None:
        solver_hf = _get_solver_hf_checkpoint(args)
        if solver_hf is None:
            return None
        _SOLVER_TOKENIZER = load_tokenizer(solver_hf, trust_remote_code=True)
    return _SOLVER_TOKENIZER


def _build_solver_prompt(args, question: str) -> str:
    tokenizer = _get_solver_tokenizer(args)
    system_prompt = "Please reason step by step, and put your final answer within \\\\boxed{}."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return "system: " + system_prompt + "\nuser: " + question


def _get_solver_sampling_params() -> dict[str, Any]:
    def _get_int_env(name: str, default: int) -> int:
        value = os.environ.get(name)
        return int(value) if value is not None else default

    def _get_float_env(name: str, default: float) -> float:
        value = os.environ.get(name)
        return float(value) if value is not None else default

    return {
        "max_new_tokens": _get_int_env("RZERO_SOLVER_RM_MAX_TOKENS", 4096),
        "temperature": _get_float_env("RZERO_SOLVER_RM_TEMPERATURE", 1.0),
        "top_p": _get_float_env("RZERO_SOLVER_RM_TOP_P", 1.0),
        "top_k": _get_int_env("RZERO_SOLVER_RM_TOP_K", 40),
        "skip_special_tokens": True,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }


def _get_solver_num_samples() -> int:
    value = os.environ.get("RZERO_SOLVER_RM_N")
    return int(value) if value is not None else 10


def _grade_answer_safe(res1: str, res2: str) -> bool:
    try:
        return bool(grade_answer(res1, res2))
    except Exception:
        return False


def _cluster_share_per_problem(problems: list[str], distance_threshold: float = 0.5) -> list[float]:
    if not problems:
        return []
    try:
        from collections import Counter

        import numpy as np
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from sklearn.cluster import AgglomerativeClustering

        n = len(problems)
        dist = np.zeros((n, n))
        smoother = SmoothingFunction().method1
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    score = 1.0
                else:
                    ref = [problems[j].split()]
                    hyp = problems[i].split()
                    score = sentence_bleu(ref, hyp, smoothing_function=smoother)
                dist[i, j] = dist[j, i] = 1 - score

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(dist)
        total = len(problems)
        cluster_size = Counter(labels)
        cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}
        return [cluster_ratio[lab] for lab in labels]
    except Exception:
        return [0.0 for _ in problems]


def _extract_question_answer(predict: str) -> tuple[str, str]:
    questions = re.findall(r"<question>(.*?)</question>", predict, re.DOTALL)
    answers = extract_boxed_content(predict)
    if isinstance(answers, list):
        answers = answers[-1] if answers else ""
    if questions and answers:
        try:
            return questions[-1].strip(), str(answers).strip()
        except Exception:
            return "", ""
    return "", ""


async def _solver_score_for_question(args, question: str) -> float:
    url = _get_solver_url(args)
    if not url:
        raise ValueError("Missing solver URL for questioner RM. Set RZERO_SOLVER_RM_URL or --rm-url.")
    prompt = _build_solver_prompt(args, question)
    sampling_params = _get_solver_sampling_params()
    num_samples = _get_solver_num_samples()

    results: list[str] = []
    for _ in range(num_samples):
        output = await post(url, {"text": prompt, "sampling_params": sampling_params})
        text = output.get("text", "") if isinstance(output, dict) else ""
        boxed = extract_boxed_content(text)
        if isinstance(boxed, list):
            boxed = boxed[-1] if boxed else ""
        if boxed:
            results.append(str(boxed))

    if not results:
        return 0.0

    answer_counts: dict[str, int] = {}
    for res in results:
        if not res:
            continue
        matched = False
        for exist_ans in list(answer_counts.keys()):
            if res == exist_ans or ("no " in res.lower() and "no " in exist_ans.lower()):
                answer_counts[exist_ans] += 1
                matched = True
                break

            try:
                is_match = _grade_answer_safe(res, exist_ans)
                if not is_match:
                    is_match = _grade_answer_safe(exist_ans, res)
                if is_match:
                    answer_counts[exist_ans] += 1
                    matched = True
                    break
            except Exception:
                continue

        if not matched:
            answer_counts[res] = 1

    if not answer_counts:
        return 0.0
    max_count = max(answer_counts.values())
    return max_count / len(results)


async def questioner_rm_group(args, samples: list[Sample], **kwargs: Any) -> list[float]:
    questions: list[str] = []
    valid_indices: list[int] = []
    for idx, sample in enumerate(samples):
        question, _answer = _extract_question_answer(sample.response)
        questions.append(question)
        if question:
            valid_indices.append(idx)

    scores = [0.0 for _ in samples]
    if valid_indices:
        tasks = [
            _solver_score_for_question(args, questions[idx])
            for idx in valid_indices
        ]
        solver_scores = await asyncio.gather(*tasks)
        for idx, score in zip(valid_indices, solver_scores, strict=False):
            scores[idx] = score

    penalties = _cluster_share_per_problem(questions, distance_threshold=0.5)
    if not penalties:
        penalties = [0.0 for _ in samples]

    rewards: list[float] = []
    for idx, score in enumerate(scores):
        if questions[idx]:
            reward = min(score, 1 - score) - penalties[idx]
        else:
            reward = -1.0
        rewards.append(float(reward))
    return rewards
