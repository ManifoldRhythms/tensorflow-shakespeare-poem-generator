
import os

GCS_BUCKET_NAME = "gs://mr-lyrics-autocomplete-data/"

MODEL_ROOT_DIR = os.path.join(GCS_BUCKET_NAME, "model/")
MODEL_CHECKPOINTS_DIR = os.path.join(MODEL_ROOT_DIR, "checkpoints/")
MODEL_LOG_DIR = os.path.join(MODEL_ROOT_DIR, "log/")
MODEL_LOG_TRAIN_DIR = os.path.join(MODEL_LOG_DIR, "train/")
MODEL_LOG_EVAL_DIR = os.path.join(MODEL_ROOT_DIR, "eval/")

TRAINING_DATA_DIR = os.path.join(GCS_BUCKET_NAME, "lyrics/")
