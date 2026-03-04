from pathlib import Path

Path("reports/validation/report.json").mkdir(parents=True, exist_ok=True)
print("Running data_validation stage")
