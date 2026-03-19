import pandas as pd
import os
from mlPipeline.constants import PARAMS_FILE_PATH
from mlPipeline.entity.config_entity import DataTransformationConfig
from mlPipeline.utils.common import read_yaml, save_json, get_size
from mlPipeline import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)

    def _build_base_stats(self) -> dict:
        """Default report structure."""
        return {
            "dataset": "creditcard",
            "file": str(self.config.data_path),
            "feature_groups": {
                "pca_features": [f"V{i}" for i in range(1, 29)],
                "time_feature": ["Time"],
                "amount_feature": ["Amount"]
            },
            "target_column": self.params.target_column,
            "input_rows": 0,
            "input_columns": 0,
            "duplicates_dropped": 0,
            "missing_target_rows_dropped": 0,
            "missing_value_counts_before": {
                "pca_features": {f"V{i}": 0 for i in range(1, 29)},
                "time_feature": {"Time": 0},
                "amount_feature": {"Amount": 0}
            },
            "missing_value_counts_after": {
                "pca_features": {f"V{i}": 0 for i in range(1, 29)},
                "time_feature": {"Time": 0},
                "amount_feature": {"Amount": 0}
            },
            "split_statistics": {
                "train": {
                    "split_size": self.params.data_transformation.train_size,
                    "row_count": 0,
                    "fraud_count": 0,
                    "fraud_percentage": 0.0,
            },
                "val": {
                    "split_size": self.params.data_transformation.val_size,
                    "row_count": 0,
                    "fraud_count": 0,
                    "fraud_percentage": 0.0,
                },
                "test": {
                    "split_size": self.params.data_transformation.test_size,
                    "row_count": 0,
                    "fraud_count": 0,
                    "fraud_percentage": 0.0,
                },
            },
            "transformations_applied": [],
            "numeric_stats": {},
        }
    
    def _generate_stats(self, report_data: dict) -> None:
        """Generate validation reports in JSON."""
        try:
            stats_file_path = self.config.stats_file_path
            os.makedirs(stats_file_path.parent, exist_ok=True)
            save_json(stats_file_path, report_data)

            logger.info(f"Validation report generated at {stats_file_path}")
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            raise

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the DataFrame."""
        initial_row_count = df.shape[0]
        df = df.drop_duplicates()
        final_row_count = df.shape[0]
        logger.info(f"Dropped duplicate rows: {initial_row_count - final_row_count}")
        return df

    def _drop_missing_target_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing target values."""
        target_column = self.params.target_column
        missing_target_rows = df[target_column].isnull().sum()
        df = df.dropna(subset=[target_column])
        logger.info(f"Dropped rows with missing target values: {missing_target_rows}")
        return df

    def _split_data_chronologically(self, df: pd.DataFrame) -> dict:
        """
        Split the DataFrame into train/val/test sets based on the defined split ratios.
        """
        # Sort by Time
        df = df.sort_values(by="Time").reset_index(drop=True)
        
        X = df.drop(columns=[self.params.target_column])
        y = df[self.params.target_column]
        
        # Splitting the data
        # First Split: Train vs Temp (Val + Test)
        train_size = self.params.data_transformation.train_size
        val_size = self.params.data_transformation.val_size
        test_size = self.params.data_transformation.test_size
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_size), random_state=self.params.data_transformation.random_state, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (val_size + test_size)), random_state=self.params.data_transformation.random_state, shuffle=False)

        return {
            "X_train": X_train.reset_index(drop=True),
            "y_train": y_train.reset_index(drop=True),
            "X_val": X_val.reset_index(drop=True),
            "y_val": y_val.reset_index(drop=True),
            "X_test": X_test.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True)
        }

    def transform_data(self) -> dict:
        """Apply data transformations to the input DataFrame."""
        stats_data = self._build_base_stats()

        try:
            data_path = self.config.data_path

            #---Guardrail: Ensure the input file is not empty ---#
            if get_size(data_path) == 0:
                logger.error(f"Input data file is empty: {data_path}")
                raise ValueError(f"Input data file is empty: {data_path}")
            
            #---Guardrail: Ensure target column exists in the input data ---#
            target_column = self.params.target_column
            df = pd.read_csv(data_path)

            if target_column not in df.columns:
                logger.error(f"Target column '{target_column}' not found in input data.")
                raise ValueError(f"Target column '{target_column}' not found in input data.")
            
            #---Stats before transformation---#
            stats_data["input_rows"] = int(df.shape[0])
            stats_data["input_columns"] = int(df.shape[1])
            stats_data["missing_value_counts_before"]["pca_features"] = {f"V{i}": int(df[f"V{i}"].isnull().sum()) for i in range(1, 29)}
            stats_data["missing_value_counts_before"]["time_feature"]["Time"] = int(df["Time"].isnull().sum())
            stats_data["missing_value_counts_before"]["amount_feature"]["Amount"] = int(df["Amount"].isnull().sum())

            #---Apply transformations---#

            # 1.Drop exact duplicates
            initial_row_count = int(df.shape[0])
            df = df.drop_duplicates()
            final_row_count = int(df.shape[0])
            stats_data["duplicates_dropped"] = initial_row_count - final_row_count
            stats_data["transformations_applied"].append(f"Dropped duplicate rows: {initial_row_count - final_row_count}")

            # 2. Drop rows with missing target values
            missing_target_rows = int(df[target_column].isnull().sum())
            df = df.dropna(subset=[target_column])
            stats_data["missing_target_rows_dropped"] = missing_target_rows
            stats_data["transformations_applied"].append(f"Dropped rows with missing target values: {missing_target_rows}")

            # 3. Split data chronologically
            split_data = self._split_data_chronologically(df)
            stats_data["transformations_applied"].append("Sorted by 'Time' column: True")
            stats_data["transformations_applied"].append("Split data into train/val/test sets.")            
    
            pca_features = [f"V{i}" for i in range(1, 29)]
            time_feature = ["Time"]
            amount_feature = ["Amount"]

            numeric_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            amount_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(np.log1p, validate=False)),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("pca_time", numeric_pipeline, pca_features + time_feature),
                ("amount", amount_pipeline, amount_feature)
            ], remainder="drop")

            preprocessor.set_output(transform="pandas")

            X_train_processed = preprocessor.fit_transform(split_data["X_train"])
            X_val_processed = preprocessor.transform(split_data["X_val"])
            X_test_processed = preprocessor.transform(split_data["X_test"])

            # save to preprocessed path as parquet
            preprocessed_data_path = self.config.preprocessed_data_path
            preprocessed_data_path.mkdir(parents=True, exist_ok=True)
            X_train_preprocessed_df = pd.concat([X_train_processed, split_data["y_train"]], axis=1)
            train_path = preprocessed_data_path / "train.parquet"
            X_train_preprocessed_df.to_parquet(train_path, index=False)

            X_val_processed_df = pd.concat([X_val_processed, split_data["y_val"]], axis=1)
            val_path = preprocessed_data_path / "val.parquet"
            X_val_processed_df.to_parquet(val_path, index=False)

            X_test_processed_df = pd.concat([X_test_processed, split_data["y_test"]], axis=1)
            test_path = preprocessed_data_path / "test.parquet"
            X_test_processed_df.to_parquet(test_path, index=False)

            # save the preprocessor object
            transformer_object_file = self.config.transformer_object_file
            os.makedirs(transformer_object_file.parent, exist_ok=True)
            joblib.dump(preprocessor, transformer_object_file)

            #---Stats after transformation---#
            #stats_data["missing_value_counts_after"]["pca_features"] = {f"V{i}": int(X_train_processed[f"V{i}"].isnull().sum()) for i in range(1, 29)}
            #stats_data["missing_value_counts_after"]["time_feature"]["Time"] = int(X_train_processed["Time"].isnull().sum())
            #stats_data["missing_value_counts_after"]["amount_feature"]["Amount"] = int(X_train_processed["Amount"].isnull().sum())
            stats_data["missing_value_counts_after"] = {"total_missing_after_transformation": int(pd.DataFrame(X_train_processed).isnull().sum().sum())}

            stats_data["split_statistics"]["train"]["row_count"] = int(X_train_processed.shape[0])
            stats_data["split_statistics"]["train"]["fraud_count"] = int(split_data["y_train"].sum())
            stats_data["split_statistics"]["train"]["fraud_percentage"] = float(split_data["y_train"].mean() * 100)
            
            stats_data["split_statistics"]["val"]["row_count"] = int(X_val_processed.shape[0])
            stats_data["split_statistics"]["val"]["fraud_count"] = int(split_data["y_val"].sum())
            stats_data["split_statistics"]["val"]["fraud_percentage"] = float(split_data["y_val"].mean() * 100)
            
            stats_data["split_statistics"]["test"]["row_count"] = int(X_test_processed.shape[0])
            stats_data["split_statistics"]["test"]["fraud_count"] = int(split_data["y_test"].sum())
            stats_data["split_statistics"]["test"]["fraud_percentage"] = float(split_data["y_test"].mean() * 100)

            stats_data["transformations_applied"].append("Applied median imputation and standard scaling to PCA features and 'Time'.")
            stats_data["transformations_applied"].append("Applied median imputation, log1p transformation, and standard scaling to 'Amount' feature.")

            stats_data["feature_count_after_transformation"] = int(X_train_processed.shape[1])
            
            feature_cols = pca_features + time_feature + amount_feature

            for col in feature_cols:
                stats_data["numeric_stats"][col] = {
                "mean": float(split_data["X_train"][col].mean()),
                "std": float(split_data["X_train"][col].std()),
                "min": float(split_data["X_train"][col].min()),
                "max": float(split_data["X_train"][col].max()),
            }
                

            # save stats_file as json
            self._generate_stats(stats_data)    

            return {
                "train_path": train_path,
                "val_path": val_path,
                "test_path": test_path,
                "transformer_object_file": transformer_object_file,
                "stats_file_path": self.config.stats_file_path
            }

        except Exception as e:
            logger.exception(f"Error occurred during data transformation: {e}")
            self._generate_stats(stats_data)  
            return False