from pathlib import Path
import pandas as pd

INPUT_FILE = Path("./results/grading.jsonl")
OUTPUT_FILE = Path("./results/grading.csv")

df = pd.read_json(INPUT_FILE, lines=True, dtype={"question_id": str})

# Save as a flat CSV for Tableau
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"Converted {len(df)} records to {OUTPUT_FILE.resolve()}")
print("Columns:", ", ".join(df.columns))