from mlPipeline.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from mlPipeline.utils.common import read_yaml, create_directories
from pathlib import Path
from mlPipeline.entity.config_entity import DataIngestionConfig


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