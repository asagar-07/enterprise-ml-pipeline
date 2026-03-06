import pandas as pd
import os
from mlPipeline.constants import PARAMS_FILE_PATH
from mlPipeline.entity.config_entity import DataValidationConfig
from mlPipeline.utils.common import read_yaml, save_json, get_size
from mlPipeline import logger


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)

    def _build_base_report(self) -> dict:
        """Default report structure."""
        return {
            "dataset": "creditcard",
            "file": str(self.config.input_file),
            "total_rows": 0,
            "total_columns": 0,
            "checks": {
                "missing_columns": [],
                "unexpected_columns": [],
                "data_type_mismatches": {},
                "missing_values": 0,
                "missing_ratio": 0.0,
                "duplicate_rows": 0,
                "duplicate_ratio": 0.0,
            },
            "thresholds": {
                "missing_threshold": self.params.data_validation.missing_threshold,
                "duplicate_threshold": self.params.data_validation.duplicate_threshold,
            },
            "status": "FAIL",
        }

    def _generate_report(self, report_data: dict) -> None:
        """Generate validation reports in JSON and TXT formats."""
        try:
            report_json_path = self.config.report_json_path            
            os.makedirs(report_json_path.parent, exist_ok=True)
            save_json(report_json_path, report_data)

            logger.info(f"Validation report generated at {report_json_path}")
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            raise

    def _write_status(self, status: str):
        """Write the validation status PASS/FAIL to a status file."""
        try:
            status_file_path = self.config.status_file_path
            status_file_path.write_text(status)
            logger.info(f"Validation status '{status}' written to {status_file_path}")
        except Exception as e:
            logger.error(f"Error writing validation status: {e}")
            raise

    def validate_data(self) -> bool:
        """Validate the input data against the provided schema and generate validation reports."""
        report_data = self._build_base_report()

        try:
            input_file = self.config.input_file
            schema_file_path = self.config.schema_file_path

            # Check if the file exists 
            if not input_file.exists():
                logger.error(f"Input file does not exist: {input_file}")
                self._generate_report(report_data)
                self._write_status("FAIL")
                return False
            
            # Check if the file is empty
            if get_size(input_file) == 0:
                logger.error(f"Input file is empty: {input_file}")
                self._generate_report(report_data)
                self._write_status("FAIL")
                return False
            
            # Load the data and perform validation against the schema
            input_data = pd.read_csv(input_file)
            schema = read_yaml(schema_file_path)

            #update report with dataset info if file is valid
            report_data["total_rows"] = int(input_data.shape[0])
            report_data["total_columns"] = int(input_data.shape[1])

            # check for expected columns
            expected_columns = list(schema["columns"].keys())
            actual_columns = list(input_data.columns)

            # check for missing columns
            missing_columns = [col for col in expected_columns if col not in actual_columns]
            if missing_columns:
                logger.error(f"Missing columns in the dataset: {missing_columns}")
                report_data["checks"]["missing_columns"] = missing_columns

            # check for unexpected columns
            unexpected_columns = [col for col in actual_columns if col not in expected_columns]
            if unexpected_columns:
                logger.error(f"Unexpected columns in the dataset: {unexpected_columns}")
                report_data["checks"]["unexpected_columns"] = unexpected_columns    
            
            # check for data types
            dtype_mismatches = {}

            for column, expected_dtype in schema["columns"].items():
                if column not in input_data.columns:
                    continue

                actual_dtype = input_data[column].dtype

                if expected_dtype == "int":
                    if not pd.api.types.is_integer_dtype(input_data[column]):
                        dtype_mismatches[column] = {
                            "expected": expected_dtype,
                            "actual": str(actual_dtype)
                        }

                elif expected_dtype == "float":
                    if not pd.api.types.is_numeric_dtype(input_data[column]):
                        dtype_mismatches[column] = {
                            "expected": expected_dtype,
                            "actual": str(actual_dtype)
                        }

                elif expected_dtype in ["str", "object"]:
                    if not (
                        pd.api.types.is_object_dtype(input_data[column])
                        or pd.api.types.is_string_dtype(input_data[column])
                    ):
                        dtype_mismatches[column] = {
                            "expected": expected_dtype,
                            "actual": str(actual_dtype)
                        }

            if dtype_mismatches:
                logger.error(f"Data type mismatches found: {dtype_mismatches}")
                report_data["checks"]["data_type_mismatches"] = dtype_mismatches
            else:
                report_data["checks"]["data_type_mismatches"] = {}

            # Check missing values above threshold
            missing_values = int(input_data.isnull().sum().sum())
            missing_ratio = missing_values / report_data["total_rows"] if report_data["total_rows"] > 0 else 0
            report_data["checks"]["missing_values"] = missing_values
            report_data["checks"]["missing_ratio"] = missing_ratio

            if missing_ratio > self.params.data_validation.missing_threshold:
                logger.error(
                    f"Missing values {missing_values}, with ratio {missing_ratio} exceed threshold "
                    f"{self.params.data_validation.missing_threshold}"
                )

            # Check duplicate rows above threshold
            duplicate_rows = int(input_data.duplicated().sum())
            duplicate_ratio = duplicate_rows / report_data["total_rows"] if report_data["total_rows"] > 0 else 0
            report_data["checks"]["duplicate_rows"] = duplicate_rows
            report_data["checks"]["duplicate_ratio"] = duplicate_ratio

            if duplicate_ratio > self.params.data_validation.duplicate_threshold:
                logger.error(
                    f"Duplicate rows {duplicate_rows}, with ratio {duplicate_ratio} exceed threshold "
                    f"{self.params.data_validation.duplicate_threshold}"
                )

            # Final validation status
            is_valid = (
                len(report_data["checks"]["missing_columns"]) == 0
                and len(report_data["checks"]["data_type_mismatches"]) == 0
                and missing_ratio <= self.params.data_validation.missing_threshold
                and duplicate_ratio <= self.params.data_validation.duplicate_threshold
            )

            report_data["status"] = "PASS" if is_valid else "FAIL"

            self._generate_report(report_data)
            self._write_status(report_data["status"])

            if is_valid:
                logger.info("Data validation completed successfully.")
            else:
                logger.error("Data validation failed.")

            return is_valid

        except Exception as e:
            logger.exception(f"Error occurred during data validation: {e}")
            report_data["status"] = "FAIL"
            self._generate_report(report_data)
            self._write_status("FAIL")
            return False