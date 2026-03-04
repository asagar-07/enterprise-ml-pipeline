import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")

os.makedirs(log_dir, exist_ok=True)

# Set up logging configuration and create a logger instance, ensure logs to terminal and file
logging.basicConfig(level=logging.INFO, format=logging_str)

logger = logging.getLogger("mlPipelineLogger")
logger.setLevel(logging.INFO)  # or DEBUG

file_handler = logging.FileHandler(log_filepath)
file_handler.setFormatter(logging.Formatter(logging_str))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(logging_str))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.propagate = False # Prevent logs from being propagated to the root logger and printed twice
