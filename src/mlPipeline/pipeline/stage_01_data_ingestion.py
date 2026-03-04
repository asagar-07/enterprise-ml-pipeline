from src.mlPipeline.config.configuration import ConfigurationManager
from src.mlPipeline.components.data_ingestion import DataIngestion
from src.mlPipeline import logger

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.unzip_and_save()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<")
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.main()
        logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.error(f"Error occurred in stage {STAGE_NAME}: {e}")
        raise