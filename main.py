"""Entry point for the resume personalization pipeline.

Run:
    python main.py
    python main.py --provider openai --model gpt-4o-mini
    python main.py --provider anthropic

Add `--human-score 8` to blend human evaluation into the final score.
Add `--dry-run` to skip the LLM call (useful for smoke-testing scoring).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import config as cfg
from evaluation.human_feedback import combine_scores
from llm.client import AnthropicClient, EchoClient, LLMClient, OpenAIClient
from llm.generator import ResumeGenerator
from scoring.similarity import build_scorer, compare
from utils.io import read_text, save_json, write_text


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grounded resume personalization with LLMs."
    )
    parser.add_argument("--resume", default=cfg.DEFAULT_RESUME_PATH, help="Path to original resume text file.")
    parser.add_argument("--jd", default=cfg.DEFAULT_JD_PATH, help="Path to job description text file.")
    parser.add_argument("--prompt", default=cfg.DEFAULT_PROMPT_PATH, help="Path to prompt template file.")
    parser.add_argument(
        "--provider",
        default=cfg.DEFAULT_PROVIDER,
        choices=["openai", "anthropic"],
        help="LLM provider to use. Picks a sensible default model if --model is not set.",
    )
    parser.add_argument("--model", default=None, help="LLM model name. Defaults to gpt-4o-mini (openai) or claude-sonnet-4-5 (anthropic).")
    parser.add_argument(
        "--scorer",
        default=cfg.DEFAULT_SCORER,
        choices=["tfidf", "embedding"],
        help="Similarity scorer to use.",
    )
    parser.add_argument(
        "--embedding-model",
        default=cfg.DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers model (only used when --scorer embedding).",
    )
    parser.add_argument(
        "--human-score",
        type=float,
        default=None,
        help="Optional human rating on a 1-10 scale; blended with ATS score.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=cfg.DEFAULT_ALPHA,
        help="Weight for ATS score in final blend; human gets (1 - alpha).",
    )
    parser.add_argument(
        "--output-dir",
        default=cfg.DEFAULT_OUTPUT_DIR,
        help="Directory to write result.json and modified_resume.txt into.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the LLM call and use an offline echo client (for testing).",
    )
    return parser.parse_args(argv)


def _resolve_model(args: argparse.Namespace) -> str:
    """Pick the model name: explicit flag > provider default."""
    if args.model:
        return args.model
    return (
        cfg.DEFAULT_MODEL_OPENAI
        if args.provider == "openai"
        else cfg.DEFAULT_MODEL_ANTHROPIC
    )


def _build_client(args: argparse.Namespace) -> LLMClient:
    if args.dry_run:
        return EchoClient()
    model = _resolve_model(args)
    if args.provider == "openai":
        return OpenAIClient(model=model, max_tokens=cfg.DEFAULT_MAX_TOKENS)
    return AnthropicClient(model=model, max_tokens=cfg.DEFAULT_MAX_TOKENS)


def run(args: argparse.Namespace) -> dict:
    resume = read_text(args.resume)
    job_description = read_text(args.jd)

    client = _build_client(args)
    generator = ResumeGenerator(client, prompt_path=args.prompt)
    generation = generator.generate(resume=resume, job_description=job_description)
    modified_resume = generation["modified_resume"]

    scorer = build_scorer(args.scorer, embedding_model=args.embedding_model)
    ats = compare(
        scorer,
        original_resume=resume,
        modified_resume=modified_resume,
        job_description=job_description,
    )

    result: dict = {
        "inputs": {
            "resume_path": str(Path(args.resume).resolve()),
            "jd_path": str(Path(args.jd).resolve()),
            "prompt_path": str(Path(args.prompt).resolve()),
            "provider": args.provider,
            "model": _resolve_model(args),
            "scorer": args.scorer,
            "dry_run": args.dry_run,
        },
        "generation": {
            "modified_resume": modified_resume,
            "changes_summary": generation.get("changes_summary", []),
            "keywords_added": generation.get("keywords_added", []),
            "groundedness_notes": generation.get("groundedness_notes", ""),
            "parse_ok": generation.get("_parse_ok", False),
        },
        "ats_scores": ats,
    }

    if args.human_score is not None:
        result["human_evaluation"] = combine_scores(
            ats_score=ats["modified_score"],
            human_score=args.human_score,
            alpha=args.alpha,
        )

    output_dir = Path(args.output_dir)
    save_json(output_dir / "result.json", result)
    write_text(output_dir / "modified_resume.txt", modified_resume)
    return result


def _print_summary(result: dict) -> None:
    ats = result["ats_scores"]
    print("\n=== Resume Personalization Summary ===")
    print(f"Scorer:           {result['inputs']['scorer']}")
    print(f"Provider:         {result['inputs']['provider']}")
    print(f"Model:            {result['inputs']['model']} (dry_run={result['inputs']['dry_run']})")
    print(f"Original score:   {ats['original_score']:.4f}")
    print(f"Modified score:   {ats['modified_score']:.4f}")
    print(f"Improvement:      {ats['improvement']:+.4f}")
    if "human_evaluation" in result:
        h = result["human_evaluation"]
        print(
            f"Final (alpha={h['alpha']}): {h['final_score']:.4f} "
            f"(human raw={h['human_score_raw']}, normalized={h['human_score_normalized']:.4f})"
        )
    if not result["generation"]["parse_ok"]:
        print("WARNING: model output was not valid JSON; raw text used as modified_resume.")


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    result = run(args)
    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
