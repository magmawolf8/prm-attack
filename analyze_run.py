#!/usr/bin/env python3
"""
Utility to inspect an adversarial prefix training run.

Given an adv_run_* directory, this script:
  * Prints / saves the discrete tokens selected from optimized logits.
  * Regenerates training plots (including derived rewards).
  * Evaluates the discrete prefix on a PRM800k sample to report
    PRM rewards / NLLs for both positive and negative trajectories,
    comparing against the baseline without the prefix.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import config as cfg  # noqa: E402
from continuous import insert_adversarial_prefix  # noqa: E402
from skywork_tokenizer import SkyworkTokenizerAPI  # noqa: E402
from skywork_o1_prm_inference.model_utils.prm_model import PRM_MODEL  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an adversarial prefix run directory.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to adv_run_* directory containing optimized_logits.pt and training_loss.csv",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Optional directory to place analysis artifacts (defaults to <run_dir>/analysis).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g., cuda:0). Defaults to cuda if available else cpu.",
    )
    return parser.parse_args()


def ensure_run_dir(run_dir: Path) -> None:
    required = ["optimized_logits.pt", "training_loss.csv"]
    for name in required:
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"Expected {name} inside {run_dir}")


def load_loss_history(csv_path: Path) -> Tuple[List[str], np.ndarray]:
    with csv_path.open("r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    columns = [h.strip() for h in header_line.split(",") if h.strip()]
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return columns, data


def moving_average(series: np.ndarray, window: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    window = min(window, len(series))
    if window <= 1:
        steps = np.arange(1, len(series) + 1)
        return series, steps
    weights = np.ones(window, dtype=np.float32) / window
    ma = np.convolve(series, weights, mode="valid")
    steps = np.arange(window, len(series) + 1)
    return ma, steps


def plot_series(steps: np.ndarray, values: np.ndarray, title: str, out_base: Path) -> None:
    out_base = Path(out_base)
    raw_path = out_base.with_name(out_base.name + "_raw.png")
    ma_path = out_base.with_name(out_base.name + "_ma10.png")

    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, linewidth=1.6)
    plt.xlabel("Optimizer step")
    plt.ylabel(title)
    plt.title(f"{title} (raw)")
    plt.tight_layout()
    plt.savefig(raw_path, dpi=150)
    plt.close()

    ma_values, ma_steps = moving_average(values)
    plt.figure(figsize=(8, 5))
    plt.plot(ma_steps, ma_values, linewidth=1.6)
    plt.xlabel("Optimizer step")
    plt.ylabel(f"{title} (10-step MA)")
    plt.title(f"{title} (10-step moving average)")
    plt.tight_layout()
    plt.savefig(ma_path, dpi=150)
    plt.close()


def describe_prefix_tokens(logits_path: Path, tokenizer, out_path: Path) -> torch.Tensor:
    logits = torch.load(logits_path, map_location="cpu")
    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits tensor, got shape {tuple(logits.shape)}")

    top_ids = torch.argmax(logits, dim=-1)
    lines = []
    for idx, token_id in enumerate(top_ids.tolist()):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        safe_text = repr(text)
        lines.append(f"Prefix {idx:02d}: token_id={token_id} -> {safe_text}")

    magic_phrase = tokenizer.decode(top_ids.tolist(), skip_special_tokens=False)
    magic_repr = repr(magic_phrase)
    lines.append("")
    lines.append(f"Magic phrase repr: {magic_repr}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nDiscrete prefix tokens (argmax per position):")
    print("\n".join(lines))
    print("\nMagic phrase repr:")
    print(magic_repr)
    return top_ids


def compute_reward_from_nll(nll: np.ndarray) -> np.ndarray:
    return np.exp(-nll)


def plot_rewards(steps: np.ndarray, rewards: Dict[str, np.ndarray], out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for label, series in rewards.items():
        plt.plot(steps, series, linewidth=1.2, label=label)
    plt.xlabel("Optimizer step")
    plt.ylabel("Reward (mean prob on reward flags)")
    plt.title("PRM reward over training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_raw.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for label, series in rewards.items():
        ma_values, ma_steps = moving_average(series)
        plt.plot(ma_steps, ma_values, linewidth=1.2, label=label)
    plt.xlabel("Optimizer step")
    plt.ylabel("Reward (10-step MA)")
    plt.title("PRM reward over training (10-step MA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_ma10.png", dpi=150)
    plt.close()


def evaluate_sample(
    prefix_ids: torch.Tensor,
    question: str,
    pos_steps: List[str],
    neg_steps: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    tokenizer_api = SkyworkTokenizerAPI(cfg.SKYWORK_MODEL_NAME, cfg.STEP_TOKEN)
    reward_model = PRM_MODEL.from_pretrained(cfg.SKYWORK_MODEL_NAME).to(device).eval()
    token_embedding_layer = reward_model.pretrained_model.model.embed_tokens.weight

    prefix_ids = prefix_ids.to(device)
    prefix_embeddings = token_embedding_layer[prefix_ids]

    def build_prefixed_batch(steps):
        tokenized = tokenizer_api.prepare_steps([question], [steps])
        input_ids = tokenized.data["input_ids"].to(device)
        batch_embeddings = token_embedding_layer[input_ids]

        pref_embeddings, pref_mask, pref_ans_flag, pref_reward_flag = insert_adversarial_prefix(
            tokenized,
            batch_embeddings,
            prefix_embeddings,
        )
        tokenized.data["attention_mask"] = pref_mask
        tokenized.data["answer_flag"] = pref_ans_flag
        tokenized.data["reward_flags"] = pref_reward_flag
        tokenized.pop("input_ids")
        tokenized = tokenized.to(device)
        pref_embeddings = pref_embeddings.to(device)
        return tokenized, pref_embeddings

    def score_with_prefix(steps) -> Tuple[float, float]:
        batch, pref_embeds = build_prefixed_batch(steps)
        with torch.no_grad():
            output = reward_model(
                **batch,
                inputs_embeds=pref_embeds,
                return_probs=True,
            )
            probs = output[2][batch.data["reward_flags"].bool()]
            nll = -torch.log(probs).mean().item()
            reward = probs.mean().item()
        return nll, reward

    def score_without_prefix(steps) -> Tuple[float, float]:
        batch = tokenizer_api.prepare_steps([question], [steps])
        batch = batch.to(device)
        with torch.no_grad():
            output = reward_model(**batch, return_probs=True)
            probs = output[2][batch.data["reward_flags"].bool()]
            nll = -torch.log(probs).mean().item()
            reward = probs.mean().item()
        return nll, reward

    pos_nll_pref, pos_reward_pref = score_with_prefix(pos_steps)
    neg_nll_pref, neg_reward_pref = score_with_prefix(neg_steps)
    pos_nll_base, pos_reward_base = score_without_prefix(pos_steps)
    neg_nll_base, neg_reward_base = score_without_prefix(neg_steps)

    return {
        "question": question,
        "pos_steps": pos_steps,
        "neg_steps": neg_steps,
        "pos_pref": {"nll": pos_nll_pref, "reward": pos_reward_pref},
        "neg_pref": {"nll": neg_nll_pref, "reward": neg_reward_pref},
        "pos_base": {"nll": pos_nll_base, "reward": pos_reward_base},
        "neg_base": {"nll": neg_nll_base, "reward": neg_reward_base},
    }


def get_training_sample() -> Tuple[str, List[str], List[str]]:
    question = (
        "For how many different digits $n$ is the two-digit number $\\underline{6}\\underline{n}$ divisible by $n$? "
        "(The expression $\\underline{6}\\underline{n}$ should be interpreted as a two-digit integer with tens digit 6 "
        "and units digit $n$, not as 6 times $n$.)"
    )
    pos_steps = [
        "I want to find all the digits $n$ such that $\\underline{6}\\underline{n}$ is a multiple of $n$.",
        "This means that $\\underline{6}\\underline{n}$ must be equal to $n$ times some integer $k$.",
        "I can write this as $\\underline{6}\\underline{n} = kn$, or equivalently, $10 \\cdot 6 + n = kn$.",
        "Subtracting $n$ from both sides, I get $60 = (k - 1)n$.",
        "This means that $n$ must be a factor of 60, and also a digit from 0 to 9.",
        "The factors of 60 are 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, and 60.",
        "Out of these, only 1, 2, 3, 4, 5, and 6 are digits.",
        "So there are 6 possible values for $n$ that make $\\underline{6}\\underline{n}$ divisible by $n$.",
        "# Answer\n\n6",
    ]
    neg_steps = [
        "I need to find the possible values of $n$ from 0 to 9 such that 6n is divisible by $n$.",
        "A quick way to check divisibility is to use the remainders of dividing by $n$.",
        "If the remainder of dividing 6 by $n$ is the same as the remainder of dividing $n$ by $n$, then 6n will be divisible by $n$.",
        "For example, if $n=3$, then the remainder of dividing 6 by 3 is 0, and the remainder of dividing 3 by 3 is also 0, so 63 is divisible by 3.",
        "On the other hand, if $n=4$, then the remainder of dividing 6 by 4 is 2, but the remainder of dividing 4 by 4 is 0, so 64 is not divisible by 4.",
        "So I can use this rule to test each value of $n$ from 0 to 9.",
        "If $n=0$, then 6n is not defined, so I exclude this case.",
        "If $n=1$, then the remainder of dividing 6 by 1 is 0, and the remainder of dividing 1 by 1 is also 0, so 61 is divisible by 1.",
        "If $n=2$, then the remainder of dividing 6 by 2 is 0, and the remainder of dividing 2 by 2 is also 0, so 62 is divisible by 2.",
        "If $n=3$, then the remainder of dividing 6 by 3 is 0, and the remainder of dividing 3 by 3 is also 0, so 63 is divisible by 3.",
        "If $n=4$, then the remainder of dividing 6 by 4 is 2, and the remainder of dividing 4 by 4 is 0, so 64 is not divisible by 4.",
        "If $n=5$, then the remainder of dividing 6 by 5 is 1, and the remainder of dividing 5 by 5 is 0, so 65 is not divisible by 5.",
        "If $n=6$, then the remainder of dividing 6 by 6 is 0, and the remainder of dividing 6 by 6 is also 0, so 66 is divisible by 6.",
        "If $n=7$, then the remainder of dividing 6 by 7 is 6, and the remainder of dividing 7 by 7 is 0, so 67 is not divisible by 7.",
        "If $n=8$, then the remainder of dividing 6 by 8 is 6, and the remainder of dividing 8 by 8 is 0, so 68 is not divisible by 8.",
        "If $n=9$, then the remainder of dividing 6 by 9 is 6, and the remainder of dividing 9 by 9 is 0, so 69 is not divisible by 9.",
        "So the only values of $n$ that make 6n divisible by $n$ are 1, 2, 3, and 6.",
        "# Answer\n\n4",
    ]
    nums = set("0123456789")
    rng = random.Random(cfg.RANDOM_SEED)
    corrupted_neg_steps = [
        "".join(
            str(rng.randint(0, 9)) if (char in nums and rng.random() > 0.5) else char
            for char in step
        )
        for step in neg_steps
    ]
    return question, pos_steps, corrupted_neg_steps


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    ensure_run_dir(run_dir)

    analysis_dir = args.analysis_dir or (run_dir / "analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.SKYWORK_MODEL_NAME, trust_remote_code=True)

    prefix_ids = describe_prefix_tokens(
        run_dir / "optimized_logits.pt",
        tokenizer,
        analysis_dir / "prefix_tokens.txt",
    )

    columns, history = load_loss_history(run_dir / "training_loss.csv")
    steps = np.arange(1, history.shape[0] + 1)

    column_map = {name: history[:, idx] for idx, name in enumerate(columns)}
    for name, series in column_map.items():
        plot_series(steps, series, name, (analysis_dir / name))

    reward_series = {}
    if "nll_pos" in column_map:
        reward_series["reward_pos"] = compute_reward_from_nll(column_map["nll_pos"])
    if "nll_neg" in column_map:
        reward_series["reward_neg"] = compute_reward_from_nll(column_map["nll_neg"])
    if reward_series:
        plot_rewards(steps, reward_series, analysis_dir)

    question, pos_steps, neg_steps = get_training_sample()
    sample_report = evaluate_sample(
        prefix_ids,
        question,
        pos_steps,
        neg_steps,
        device,
    )

    final_metrics = {name: series[-1] for name, series in column_map.items()}
    report_lines = [
        f"Run directory: {run_dir}",
        f"Total optimization steps: {len(steps)}",
        "",
        "Final logged metrics:",
    ]
    for name, value in final_metrics.items():
        report_lines.append(f"  {name}: {value:.6f}")

    report_lines += [
        "",
        "Evaluation sample:",
        f"Question: {sample_report['question']}",
        "",
        "Discrete prefix vs. PRM baseline:",
        f"  Positive traj baseline reward: {sample_report['pos_base']['reward']:.4f} "
        f"(NLL={sample_report['pos_base']['nll']:.4f})",
        f"  Positive traj w/ prefix reward: {sample_report['pos_pref']['reward']:.4f} "
        f"(NLL={sample_report['pos_pref']['nll']:.4f})",
        f"  Negative traj baseline reward: {sample_report['neg_base']['reward']:.4f} "
        f"(NLL={sample_report['neg_base']['nll']:.4f})",
        f"  Negative traj w/ prefix reward: {sample_report['neg_pref']['reward']:.4f} "
        f"(NLL={sample_report['neg_pref']['nll']:.4f})",
        "",
        "Step samples (truncated to first 2 entries each):",
        f"  Positive steps sample: {json.dumps(sample_report['pos_steps'][:2], ensure_ascii=False)}",
        f"  Negative steps sample: {json.dumps(sample_report['neg_steps'][:2], ensure_ascii=False)}",
    ]

    report_path = analysis_dir / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
