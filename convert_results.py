from pathlib import Path
import pandas as pd
import csv

INPUT_FILE = Path("./results/results.jsonl")
OUTPUT_FILE = Path("./results/results.csv")

df = pd.read_json(INPUT_FILE, lines=True, dtype={"question_id": str})

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_MINIMAL,
    lineterminator="\n"
)

print(f"Converted {len(df)} records to {OUTPUT_FILE.resolve()}")