from mlPipeline.config.configuration import ConfigurationManager
from mlPipeline.components.data_transformation import DataTransformation
from mlPipeline import logger

STAGE_NAME = "Data Transformation Stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transform_data()
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<")
        data_transformation_pipeline = DataTransformationPipeline()
        data_transformation_pipeline.main()
        logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.error(f"Error occurred in stage {STAGE_NAME}: {e}")
        raise