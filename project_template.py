import os
from pathlib import Path
import logging

#logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s]: %(message)s:')

#project name
project_name = "mlPipeline"

#project root directory
project_root = Path(os.getcwd())

logging.info(f"Creating project structure at: {project_root}")

# Define the project structure
list_of_files = [
    ".github/workflows/.gitkeep",
    "api/app.py",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/features/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/ingestion.py",
    f"src/{project_name}/validation.py",
    f"src/{project_name}/features.py",
    f"src/{project_name}/preprocessing.py",
    f"src/{project_name}/train.py",
    f"src/{project_name}/evaluate.py",
    f"src/{project_name}/register.py",
    f"src/{project_name}/utils/__init__.py",
    "reports/validation/.gitkeep",
    "reports/evaluation/.gitkeep",
    "requirements.txt",
    "README.md",
    "setup.py",
    "configs/config.yaml",
    "logs/prompt/prompts.json",
    "dvc.yaml",
    "params.yaml",
    "notebooks/.gitkeep",
    "research/trials.ipynb"
]

for filepath in list_of_files:
    filepath = project_root / filepath
    filedir = filepath.parent

    if not filedir.exists():
        logging.info(f"Creating directory: {filedir}")
        filedir.mkdir(parents=True, exist_ok=True)

    if not filepath.exists() or filepath.stat().st_size == 0: # Check if file doesn't exist or is empty
        logging.info(f"Creating file: {filepath}")
        filepath.touch() 
    else:
        logging.info(f"File already exists: {filepath}")