"""Questioner data source for R-Zero training under slime."""

from __future__ import annotations

from slime.rollout.data_source import DataSource
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

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

    def __len__(self) -> int:
        return max(1, getattr(self.args, "rollout_batch_size", 1))
