#!/usr/bin/env python3
"""ECSIE workload generator.

Generates synthetic benchmark workload JSON files with configurable
prompt length distributions, token budget variance, and batch sizes.

Usage:
    python generate_workload.py --name custom --n 50 --max-tokens 512 --out workloads/custom.json
"""

import argparse
import json
import random
import string


SAMPLE_TOPICS = [
    "artificial intelligence", "climate change", "quantum computing",
    "the French Revolution", "neural network architecture",
    "protein folding", "the Fermi paradox", "distributed systems",
    "the philosophy of mind", "renewable energy",
]


def generate_prompt(rng: random.Random) -> str:
    topic = rng.choice(SAMPLE_TOPICS)
    templates = [
        f"Explain {topic} in detail.",
        f"What are the key challenges in {topic}?",
        f"Write a short essay about {topic}.",
        f"Summarise recent developments in {topic}.",
        f"Compare and contrast two perspectives on {topic}.",
    ]
    return rng.choice(templates)


def main() -> None:
    parser = argparse.ArgumentParser(description="ECSIE workload generator")
    parser.add_argument("--name",       default="custom")
    parser.add_argument("--n",          type=int,   default=20,  help="Number of prompts")
    parser.add_argument("--max-tokens", type=int,   default=256)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--temperature",type=float, default=1.0)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--out",        default="workloads/custom.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    prompts = [
        {
            "text":       generate_prompt(rng),
            "max_tokens": rng.randint(args.max_tokens // 2, args.max_tokens),
        }
        for _ in range(args.n)
    ]

    workload = {
        "name":        args.name,
        "description": f"Auto-generated workload (seed={args.seed})",
        "prompts":     prompts,
        "batch_size":  args.batch_size,
        "repetitions": 1,
        "temperature": args.temperature,
        "seed":        args.seed,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(workload, f, indent=2)

    print(f"[ecsie] generated {len(prompts)} prompts → {args.out}")


if __name__ == "__main__":
    main()
