from mlPipeline.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from mlPipeline.utils.common import read_yaml, create_directories
from pathlib import Path
from mlPipeline.entity.config_entity import DataIngestionConfig, DataValidationConfig


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        data_ingestion = self.config.data_ingestion

        create_directories([Path(data_ingestion.root_dir)])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(data_ingestion.root_dir),
            source_URL=data_ingestion.source_URL,
            local_data_file=Path(data_ingestion.local_data_file),
            unzip_dir=Path(data_ingestion.unzip_dir)
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        data_validation = self.config.data_validation

        report_dir = Path(data_validation.report_dir)
        create_directories([report_dir])

        report_json = report_dir / data_validation.report_json
        status_file = report_dir / data_validation.status_file

        data_validation_config = DataValidationConfig(
            input_file=Path(data_validation.input_file),
            schema_file_path=Path(data_validation.schema_file_path),
            report_dir=report_dir,
            report_json_path=report_json,
            status_file_path=status_file
        )

        return data_validation_config