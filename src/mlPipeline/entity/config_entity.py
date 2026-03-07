from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    input_file: Path
    schema_file_path: Path
    report_dir: Path
    report_json_path: Path
    status_file_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    preprocessed_data_path: Path
    transformer_object_file: Path
    stats_file_path: Path
