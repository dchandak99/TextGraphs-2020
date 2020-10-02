from pathlib import Path
MODE_TRAIN = "train"
MODE_DEV = "dev"
MODE_TEST = "test"

def get_path_df_scores(mode: str, clean_trn: bool = False) -> str:
  path = f"df_scores_{mode}.csv"
  if clean_trn:
    path = "unshuffled_undropped_" + path
  return path

def get_path_predict(mode: str) -> str:
  f = {
      MODE_TRAIN: "predict_trn.txt",
      MODE_DEV: "predict_dev.txt",
      MODE_TEST: "predict.txt",
  }[mode]
  print(f"Mode: {mode}, predict file will be saved to: {f}")
  return f

def get_path_q(path_data: Path, mode: str) -> Path:
  file_q = {
      MODE_TRAIN: "questions_train.tsv",
      MODE_DEV: "questions_dev.tsv",
      MODE_TEST: "questions_test.tsv",
  }[mode]
  return path_data.joinpath("raw/questions", file_q)

