
from pathlib import Path
from typing import Dict, Union
import re

import pandas as pd
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage


GRADER_SYSTEM_PROMPT = """
You are an impartial grader evaluating an LLM response
against a grading specification.

Grade only according to the supplied specification and question.
Do not assume unsupported facts are correct. Do not reward an answer
merely because it sounds plausible.

Use these three scoring dimensions:

1. Factual Correctness (0-2)
   2 = Correct and sufficiently complete according to the specification.
   1 = Partially correct or incomplete.
   0 = Incorrect, materially misleading, or fails to address the question.

2. Hallucination Avoidance (0-2)
   2 = Avoids fabricated or unsupported claims.
   1 = Contains minor unsupported details.
   0 = Contains substantial fabricated or unsupported claims.

3. Capability-Appropriate Behaviour (0-2)
   2 = Uses the available capabilities appropriately, or acknowledges
       limitations when the required capability is unavailable.
   1 = Partially appropriate behaviour.
   0 = Fabricates results, fails to use a required available capability,
       or refuses unnecessarily when it should answer.

Important:
- Follow the grading specification when deciding what behaviour is expected.
- An appropriate acknowledgement of missing information can receive a high
  score when the specification says the model lacks the required capability.
- Do not penalise a model for not using a tool it was not given.
- Do not assume a tool was actually called solely because the answer
  claims it was called. Judge only the evidence provided.
- Evaluate the answer against the supplied specification, not against
  your own preferred answer.

Return ONLY three comma-separated integers in this order:
factual_correctness,hallucination_avoidance,capability_appropriate

Each score must be 0, 1, or 2.
Example: 2,2,1
Do not include explanations or any other text.
"""


class LLMGrader:
    def __init__(
        self,
        questions_path: Path,
        model_name: str = "gemma4:e4b",
        temperature: float = 0.0,
        seed: Union[None, int] = 42,
    ):
        self.questions_path = questions_path
        self.questions_df = pd.read_excel(
            self.questions_path,
            dtype={"question_id": str},
        )

        required_columns = {"question_id", "question"}
        missing = required_columns - set(self.questions_df.columns)

        if missing:
            raise ValueError(
                f"Missing required Excel columns: {sorted(missing)}"
            )

        # Prevent ambiguous lookups.
        if self.questions_df["question_id"].duplicated().any():
            duplicates = self.questions_df.loc[
                self.questions_df["question_id"].duplicated(),
                "question_id",
            ].tolist()
            raise ValueError(
                f"Duplicate question IDs found: {duplicates}"
            )

        self.questions_df["question_id"] = (
            self.questions_df["question_id"].str.strip()
        )

        self.llm = ChatOllama(
            model=model_name,
            validate_model_on_init=True,
            temperature=temperature,
            seed=seed,
        )

    def get_question(self, question_id: str) -> str:
        """Look up a question using its ID in the Excel benchmark."""
        question_id = str(question_id).strip()

        matches = self.questions_df.loc[
            self.questions_df["question_id"] == question_id,
            "question",
        ]

        if matches.empty:
            raise KeyError(
                f"Question ID {question_id!r} was not found "
                f"in {self.questions_path}"
            )

        question = matches.iloc[0]

        if pd.isna(question):
            raise ValueError(
                f"Question {question_id!r} has no question text."
            )

        return str(question)

    def grade(
        self,
        question_id: str,
        answer: str,
        specification: str,
    ) -> Dict:
        """
        Grade an answer and return numerical scores.

        Scores:
            factual_correctness: 0-2
            hallucination_avoidance: 0-2
            capability_appropriate: 0-2
        """
        question_id = str(question_id).strip()
        question = self.get_question(question_id)

        human_prompt = f"""
Question ID:
{question_id}

Question:
{question}

Grading specification:
{specification}

Model answer:
{answer}

Return the three numerical scores only.
"""

        result = self.llm.invoke([
            SystemMessage(content=GRADER_SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ])

        raw_response = str(result.content).strip()

        # Extract the three scores, allowing whitespace or surrounding text.
        match = re.search(
            r"(?<!\d)([0-2])\s*,\s*([0-2])\s*,\s*([0-2])(?!\d)",
            raw_response,
        )

        if not match:
            raise ValueError(
                f"Invalid grader output for question {question_id}: "
                f"{raw_response!r}"
            )

        factual, hallucination, capability = map(
            int, match.groups()
        )

        return {
            "question_id": question_id,
            "factual_correctness": factual,
            "hallucination_avoidance": hallucination,
            "capability_appropriate": capability,
            "total_score": factual + hallucination + capability,
            "raw_response": raw_response,
        }


if __name__ == "__main__":
    grader = LLMGrader(
        questions_path=Path("./benchmark_questions/questions.xlsx")
    )

    result = grader.grade(
        question_id="52",
        answer=(
            "A GMS-2 calibration record includes the device ID, "
            "calibration date, technician ID, and measured deviation."
        ),
        specification=(
            "The answer should identify the four required fields: "
            "device ID, calibration date, technician ID, and measured "
            "deviation."
        ),
    )

    print(result)
