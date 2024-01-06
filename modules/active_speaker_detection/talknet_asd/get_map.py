r"""Compute active speaker detection performance for the AVA dataset.
Please send any questions about this code to the Google Group ava-dataset-users:
https://groups.google.com/forum/#!forum/ava-dataset-users
Example usage:
python -O get_ava_active_speaker_performance.py \
-g testdata/eval.csv \
-p testdata/predictions.csv \
-v
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def compute_average_precision(precision, recall):
  """Compute Average Precision according to the definition in VOCdevkit.
  Precision is modified to ensure that it does not decrease as recall
  decrease.
  Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls
  Raises:
    ValueError: if the input is not of the correct format
  Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.
  """
  if precision is None:
    if recall is not None:
      raise ValueError("If precision is None, recall must also be None")
    return np.NAN

  if not isinstance(precision, np.ndarray) or not isinstance(
      recall, np.ndarray):
    raise ValueError("precision and recall must be numpy array")
  if precision.dtype != np.float or recall.dtype != np.float:
    raise ValueError("input must be float numpy array.")
  if len(precision) != len(recall):
    raise ValueError("precision and recall must be of the same size.")
  if not precision.size:
    return 0.0
  if np.amin(precision) < 0 or np.amax(precision) > 1:
    raise ValueError("Precision must be in the range of [0, 1].")
  if np.amin(recall) < 0 or np.amax(recall) > 1:
    raise ValueError("recall must be in the range of [0, 1].")
  if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
    raise ValueError("recall must be a non-decreasing array")

  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  # Smooth precision to be monotonically decreasing.
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])

  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision


def load_csv(filename, column_names):
  """Loads CSV from the filename using given column names.
  Adds uid column.
  Args:
    filename: Path to the CSV file to load.
    column_names: A list of column names for the data.
  Returns:
    df: A Pandas DataFrame containing the data.
  """
  # Here and elsewhere, df indicates a DataFrame variable.

  df = pd.read_csv(filename, usecols=column_names)
  #df = pd.read_csv(filename, header=None, names=column_names)
  
  # Creates a unique id from frame timestamp and entity id.
  #df["uid"] = (df["frame_timestamp"].map(str) + ":" + df["entity_id"])  
  return df


def eq(a, b, tolerance=1e-09):
  """Returns true if values are approximately equal."""
  return abs(a - b) <= tolerance


def get_all_positives(df_merged):
  """Counts all positive examples in the groundtruth dataset."""
  return df_merged[df_merged["label"] == 1]["uid"].count()


def calculate_precision_recall(df_merged):
  """Calculates precision and recall arrays going through df_merged row-wise."""
  all_positives = get_all_positives(df_merged)
  # Populates each row with 1 if this row is a true positive
  # (at its score level).
  df_merged["is_tp"] = np.where(
      (df_merged["label"] == 1) &
      (df_merged["pred"] == 1), 1, 0)

  # Counts true positives up to and including that row.
  df_merged["tp"] = df_merged["is_tp"].cumsum()

  # Calculates precision for every row counting true positives up to
  # and including that row over the index (1-based) of that row.
  df_merged["precision"] = df_merged["tp"] / (df_merged.index + 1)
  # Calculates recall for every row counting true positives up to
  # and including that row over all positives in the groundtruth dataset.

  df_merged["recall"] = df_merged["tp"] / all_positives
  # logging.info(
  #     "\n%s\n",
  #     df_merged.head(10)[[
  #         "uid", "posScore", "label", "is_tp", "tp", "precision", "recall"
  #     ]])

  return np.array(df_merged["precision"]), np.array(df_merged["recall"])


def run_evaluation(predictions):
  """Runs AVA Active Speaker evaluation, printing average precision result."""
  column_names = ["uid", "Unnamed: 0", "video", "audio", "label", "center", "pred", "posScore"]
  df_loaded = load_csv(
      predictions,
      column_names=column_names)
  df_loaded = df_loaded.sort_values(by=["posScore"], ascending=False).reset_index()
  precision, recall = calculate_precision_recall(df_loaded)
  mAP = 100 * compute_average_precision(precision, recall)
  print("average precision: %2.2f%%"%(mAP))
  return mAP

def evaluate_pycall(df_loaded):
  """Runs AVA Active Speaker evaluation, printing average precision result."""
  column_names = ["uid","label","pred","posScore"]
  df_loaded = df_loaded.sort_values(by=["posScore"], ascending=False).reset_index()
  precision, recall = calculate_precision_recall(df_loaded)
  mAP = 100 * compute_average_precision(precision, recall)
  print("average precision: %2.2f%%"%(mAP))
  return mAP




def parse_arguments():
  """Parses command-line flags.
  Returns:
    args: a named tuple containing three file objects args.labelmap,
    args.groundtruth, and args.detections.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-p",
      "--predictions",
      help="CSV file containing active speaker predictions.",
      type=argparse.FileType("r"),
      required=True)
  parser.add_argument(
      "-v", "--verbose", help="Increase output verbosity.", action="store_true")
  return parser.parse_args()


def main():
  start = time.time()
  args = parse_arguments()
  if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
  del args.verbose
  mAP = run_evaluation(**vars(args))
  logging.info("Computed in %s seconds", time.time() - start)
  return mAP

if __name__ == "__main__":
  main()