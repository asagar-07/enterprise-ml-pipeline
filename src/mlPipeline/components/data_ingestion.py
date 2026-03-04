import zipfile
import gdown
from src.mlPipeline.utils.common import get_size
from src.mlPipeline import logger
from src.mlPipeline.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """Download the file from the source URL and save it to the local data file path."""
        try:
            root_dir = self.config.root_dir
            root_dir.mkdir(parents=True, exist_ok=True)
            
            dataset_url = self.config.source_URL
            zip_download_path = self.config.local_data_file
            
            zip_download_path.parent.mkdir(parents=True, exist_ok=True)

            file_id = dataset_url.split("/d/")[1].split("/")[0]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id, str(zip_download_path), quiet=False)
            logger.info(f"Downloaded file from {dataset_url} to {zip_download_path}")
            return zip_download_path
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise

    def unzip_and_save(self):
        # Unzip the downloaded file and save it to the specified directory
        try:
            unzip_dir = self.config.unzip_dir
            unzip_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(path=unzip_dir)
            # rename extracted file to creditcard.csv
            for file in unzip_dir.glob("*.csv"):
                file.rename(unzip_dir/"creditcard.csv")
            logger.info(f"Unzipped file {self.config.local_data_file} to directory {unzip_dir}")
        except Exception as e:
            logger.error(f"Error unzipping file: {e}")
            raise