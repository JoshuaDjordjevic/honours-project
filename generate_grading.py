
import json
from pathlib import Path
from typing import Tuple

from scripts import LLMGrader


QUESTIONS_FILE = Path("./benchmark_questions/questions.xlsx")
RESPONSES_FILE = Path("./results/results.jsonl")
OUTPUT_FILE = Path("./results/grading.jsonl")

IDEAL_SPEC_COLUMN = "ideal_response_spec"
NON_IDEAL_SPEC_COLUMN = "non_ideal_response_spec"


def select_specification(row, use_rag: bool, use_tools: bool) -> Tuple[bool, str]:
    """
    Select the ideal specification when the configuration has
    all capabilities required by the question.

    Otherwise, select the non-ideal specification.
    """
    answerable_from_document = str(
        row.get("requires_rag", "")
    ).strip().lower() in {"yes", "true", "1"}

    requires_tools = str(
        row.get("requires_tools", "")
    ).strip().lower() in {"yes", "true", "1"}

    has_required_rag = not answerable_from_document or use_rag
    has_required_tools = not requires_tools or use_tools

    if has_required_rag and has_required_tools:
        return True, str(row[IDEAL_SPEC_COLUMN])

    return False, str(row[NON_IDEAL_SPEC_COLUMN])


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    grader = LLMGrader(QUESTIONS_FILE)

    # Build a lookup table from the benchmark Excel file.
    questions = grader.questions_df.copy()
    questions["question_id"] = (
        questions["question_id"].astype(str).str.strip()
    )
    questions_by_id = questions.set_index("question_id").to_dict(
        orient="index"
    )

    # Support resuming without duplicating previously graded responses.
    graded_keys = set()

    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    graded_keys.add((
                        str(record["question_id"]),
                        record["config"],
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue

    total = 0
    skipped = 0
    failed = 0

    with (
        RESPONSES_FILE.open("r", encoding="utf-8") as responses_file,
        OUTPUT_FILE.open("a", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(responses_file, start=1):
            if not line.strip():
                continue

            total += 1

            try:
                response = json.loads(line)

                question_id = str(response["question_id"]).strip()
                config = response["config"]
                key = (question_id, config)

                if key in graded_keys:
                    skipped += 1
                    continue

                if question_id not in questions_by_id:
                    raise KeyError(
                        f"Question ID {question_id!r} not found in Excel."
                    )

                question_row = questions_by_id[question_id]

                use_rag = bool(response["use_rag"])
                use_tools = bool(response["use_tools"])

                is_ideal_specification, specification = select_specification(
                    question_row,
                    use_rag=use_rag,
                    use_tools=use_tools,
                )

                result = grader.grade(
                    question_id=question_id,
                    answer=response.get("answer", ""),
                    specification=specification,
                )

                # Include configuration metadata for later filtering.
                result.update({
                    "config": config,
                    "use_tools": use_tools,
                    "use_rag": use_rag,
                    "is_ideal_specification": is_ideal_specification
                })

                output_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output_file.flush()

                graded_keys.add(key)

                print(
                    f"[{total}] Graded question {question_id} "
                    f"({config}): {result['total_score']}/6"
                )

            except Exception as exc:
                failed += 1
                print(
                    f"[Line {line_number}] Failed to grade response: {exc}"
                )

    print("\nGrading complete.")
    print(f"Responses read: {total}")
    print(f"Already graded / skipped: {skipped}")
    print(f"Failures: {failed}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
