import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")

# Set up logging configuration and create a logger instance, ensure logs to terminal and file
logging.basicConfig(level=logging.INFO, format=logging_str)

logger = logging.getLogger("mlPipelineLogger")

os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(log_filepath)
file_handler.setFormatter(logging.Formatter(logging_str))
logger.addHandler(file_handler)
logger.propagate = False # Prevent logs from being propagated to the root logger and printed twice
