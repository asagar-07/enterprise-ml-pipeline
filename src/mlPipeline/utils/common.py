import os
from box.exceptions import BoxValueError
import yaml
from mlPipeline import logger
import json
import joblib
# from ensure import ensure_annotations
from box import ConfigBox
from typing import Any
from pathlib import Path

# @ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): The path to the YAML file.

    Returns:
        ConfigBox: The contents of the YAML file as a ConfigBox object.
    
    Raises:
        BoxValueError: If the YAML file cannot be read or parsed.
        e : empty file, invalid format, etc.
    """
    path_to_yaml = Path(path_to_yaml)

    if not path_to_yaml.exists():
        logger.error(f"Config not found: {path_to_yaml.resolve()}")
        raise FileNotFoundError(f"Config not found: {path_to_yaml.resolve()}")
    try:
        content = yaml.safe_load(path_to_yaml.read_text())
    except Exception as e:
        logger.error(f"Error reading YAML file: {path_to_yaml.resolve()}: {e}")
        raise

    if content is None:
        logger.error(f"YAML file is empty: {path_to_yaml.resolve()}")
        raise ValueError(f"YAML file is empty: {path_to_yaml.resolve()}")

    try:
        logger.info(f"Successfully loaded YAML file: {path_to_yaml.resolve()}")
        return ConfigBox(content)
    except BoxValueError as e:
        logger.error(f"Error parsing YAML into ConfigBox: {path_to_yaml.resolve()}: {e}")
        raise

# @ensure_annotations
def create_directories(path_to_directories, verbose=True):
    """
    Creates directories from a list of paths.

    Args:
        path_to_directories (list[Path]): A list of directory paths to create.
        verbose (bool, optional): If True, logs the creation of each directory. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")

# @ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves a dictionary as a JSON file.

    Args:
        path (Path): The path where the JSON file will be saved.
        data (dict): The dictionary to save as JSON.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Data successfully saved to JSON file: {path}")


# @ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and returns its contents as a ConfigBox object.

    Args:
        path (Path): The path to the JSON file.

    Returns:
        ConfigBox: The contents of the JSON file as a ConfigBox object.
    """
    with open(path, "r") as f:
        data = json.load(f)
    logger.info(f"Data successfully loaded from JSON file: {path}")
    return ConfigBox(data)


# @ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves data to a binary file using joblib.

    Args:
        data (Any): The data to be saved as binary.
        path (Path): The path where the binary file will be saved.
    """
    joblib.dump(data, path)
    logger.info(f"Data successfully saved to binary file: {path}")


# @ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads data from a binary file using joblib.

    Args:
        path (Path): The path to the binary file.

    Returns:
        Any: The data loaded from the binary file.
    """
    data = joblib.load(path)
    logger.info(f"Data successfully loaded from binary file: {path}")
    return data


# @ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the size of a file in a human-readable format.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The size of the file in a human-readable format.
    """
    size_bytes = os.path.getsize(path)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def save_to_parquet(df, path: Path):
    """
    Saves a DataFrame to a Parquet file.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        path (Path): The path where the Parquet file will be saved.
    """
    df.to_parquet(path, index=False)
    logger.info(f"DataFrame successfully saved to Parquet file: {path}")    

def save_text(file_path: Path, text: str) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(text)