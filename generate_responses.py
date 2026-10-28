import json
import time
from pathlib import Path

import pandas as pd

from scripts import LLMConfiguration


INPUT_FILE = Path("./benchmark_questions/questions.xlsx")
OUTPUT_FILE = Path("./results/results.jsonl")

CONFIGS = {
    "baseline": (False, False),
    "rag_only": (True, False),
    "tools_only": (False, True),
    "rag_tools": (True, True),
}


def load_completed_runs():
    """Identify completed question/configuration pairs for resuming."""
    completed = set()

    if not OUTPUT_FILE.exists():
        return completed

    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                # Only skip successful, completed responses.
                if record.get("error") is None:
                    completed.add((
                        record["question_id"],
                        record["config"],
                    ))
            except (json.JSONDecodeError, KeyError):
                continue

    return completed


def save_record(record):
    """Append each result immediately to reduce data loss."""
    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_benchmark():
    df = pd.read_excel(INPUT_FILE, dtype={"question_id": str})

    required_columns = {"question_id", "category", "question"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing Excel columns: {sorted(missing)}")

    if df["question_id"].isna().any():
        raise ValueError("Every question must have a question_id.")

    if df["question_id"].duplicated().any():
        raise ValueError("question_id values must be unique.")

    # Initialise each configuration once.
    models = {
        name: LLMConfiguration(
            use_rag=use_rag,
            use_tools=use_tools,
        )
        for name, (use_rag, use_tools) in CONFIGS.items()
    }

    completed = load_completed_runs()
    total = len(df) * len(CONFIGS)

    for row in df.to_dict(orient="records"):
        question_id = str(row["question_id"])
        question = str(row["question"])

        for config_name, model in models.items():
            run_key = (question_id, config_name)

            if run_key in completed:
                continue

            print(
                f"[{len(completed)}/{total}] "
                f"{question_id} - {config_name}"
            )

            start = time.perf_counter()

            record = {
                "question_id": question_id,
                "category": str(row["category"]),
                "config": config_name,
                "use_rag": CONFIGS[config_name][0],
                "use_tools": CONFIGS[config_name][1],
                "question": question,
                "answer": None,
                "latency_seconds": None,
                "error": None,
            }

            try:
                response = model.invoke(question)
                record["answer"] = response.content

            except Exception as exc:
                record["error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                record["latency_seconds"] = round(
                    time.perf_counter() - start, 4
                )
                save_record(record)

            completed.add(run_key)

    print(f"Finished. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_benchmark()
