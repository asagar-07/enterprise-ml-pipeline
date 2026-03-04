from pathlib import Path

Path("data/features").mkdir(parents=True, exist_ok=True)
print("Running feature_engineering stage")