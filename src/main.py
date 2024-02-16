import argparse
import logging
import os
import pickle

import numpy as np
import metric_learn
import random
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from typing import Dict, List, Tuple


def _convert_labels(labels: List[int]) -> List[int]:
    labels_fixed = []
    for label in labels:
        if label == 0:
            label = -1
        labels_fixed.append(label)
    return labels_fixed
        

def load_data_dict(file_path: str) -> Tuple[List[List[float]], List[int]]:
    data_dict = pickle.load(open(file_path, "rb"))
    pairs: List[List[float]] = data_dict["pairs"]
    labels: List[int] = data_dict["labels"]
    labels = _convert_labels(labels)

    return pairs, labels


def get_file_pathes(data_dir: str) -> List[str]:
    file_pathes = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        file_pathes.append(file_path)

    return file_pathes


def load_data_dict_iterative(data_dir: str, file_pathes: List[str] = None) -> Tuple[List[List[float]], List[int]]:
    pairs_mat = []
    labels_mat = []
    if data_dir is not None and file_pathes is None:
        file_pathes = get_file_pathes(data_dir)
    for file_path in file_pathes:
        pairs, labels = load_data_dict(file_path)
        pairs_mat = pairs_mat + pairs
        labels_mat = labels_mat + labels

    return pairs_mat, labels_mat


def down_sample(pairs: List[List[float]], labels: List[int],
                pairs_thresh: int = 5000, seed: int = 12345) -> Tuple[List[List[float]], List[int]]:
    random.seed(seed)
    num_pairs = len(pairs)
    if num_pairs < pairs_thresh:
        return pairs, labels
    shuffled_ids = random.sample(range(num_pairs), pairs_thresh)
    sampled_pairs = [pairs[shuffled_id] for shuffled_id in shuffled_ids] 
    sampled_labels = [labels[shuffled_id] for shuffled_id in shuffled_ids] 

    return sampled_pairs, sampled_labels


def train_itml(pairs: List[List[float]], labels: List[int],
               gamma: float = 1.0, seed: int = 12345):
    random_state = np.random.RandomState(seed)
    itml = metric_learn.ITML(gamma=gamma, random_state=random_state)
    itml.fit(pairs, labels)

    return itml


def train_itml_batch(data_dir: str, seed: int = 12345):
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    itml = metric_learn.ITML(random_state=random_state)

    file_pathes = get_file_pathes(data_dir)
    pairs_mat, labels_mat = load_data_dict_iterative(data_dir)
    num_files = len(file_pathes)
    num_pairs = len(pairs_mat)
    step = num_pairs // num_files
    if num_pairs % num_files:
        step += 1
    shuffled_ids = random.sample(range(num_pairs), num_pairs)
    curr_id = 0
    for _ in tqdm(range(num_files), desc="[training...]"):
        curr_ids = shuffled_ids[curr_id:min(curr_id+step, num_pairs-1)]
        curr_pairs = [pairs_mat[id] for id in curr_ids]
        curr_labels = [labels_mat[id] for id in curr_ids]
        itml.fit(curr_pairs, curr_labels)
        curr_id += step

    return itml


def predict_itml(pairs: List[List[float]], itml) -> List[int]:
    labels = itml.predict(np.array(pairs))

    return labels


def predict_lexeme(pairs: List[List[float]]) -> List[int]:
    labels = []
    for pair in pairs:
        cos_dist = cosine(pair[0], pair[1])
        label = -1 if 0.5 <= cos_dist <= 1.5 else 1
        labels.append(label)
    return labels


def evaluate_itml(labels_gold: List[int], labels_pred: List[int]) -> float:
    return accuracy_score(labels_gold, labels_pred)


def evaluate_performance(data_dir: str, data_pathes: List[str], model, digit2model):
    if data_pathes is None:
        data_pathes = get_file_pathes(data_dir)

    if digit2model is None:
        for data_path in data_pathes:
            X_eval, y_eval = load_data_dict(data_path)
            y_pred = predict_itml(X_eval, model)
            accuracy = evaluate_itml(y_eval, y_pred)
            logging.info(f"[evaluate_performance] - {data_path}: {accuracy}")
    else:
        for digit, model in digit2model.items():
            logging.info(f"[evaluate_performance] - model: gamma=10**{digit}")
            for data_path in data_pathes:
                X_eval, y_eval = load_data_dict(data_path)
                y_pred = predict_itml(X_eval, model)
                accuracy = evaluate_itml(y_eval, y_pred)
                logging.info(f"[evaluate_performance]   - {data_path}: {accuracy}")


def main(train_data_dir: str, 
         dev_data_dir: str, 
         test_data_dir: str,
         train_data_pathes: List[str],
         dev_data_pathes: List[str],
         test_data_pathes: List[str],
         search_param: bool = False) -> None:
    logging.info("[main] args")
    logging.info(f"[main] - train_data_dir: {train_data_dir}")
    logging.info(f"[main] - dev_data_dir: {dev_data_dir}")
    logging.info(f"[main] - test_data_dir: {test_data_dir}")
    logging.info(f"[main] - train_data_pathes: {train_data_pathes}")
    logging.info(f"[main] - dev_data_pathes: {dev_data_pathes}")
    logging.info(f"[main] - test_data_pathes: {test_data_pathes}")
    logging.info(f"[main] - search_param: {search_param}")

    if (train_data_dir is None and train_data_pathes is None) \
            or (test_data_dir is None and test_data_pathes is None):
        assert False, "[main] ERROR: Input is empty"

    logging.info("[main] train itml ...")
    digit2model = None
    X_train, y_train = load_data_dict_iterative(data_dir=train_data_dir, 
                                                file_pathes=train_data_pathes)
    logging.info("[main] - data loaded")
    logging.info(f"[main]   - X_train: {len(X_train)}items")
    logging.info(f"[main]   - y_train: {len(y_train)}items")
    X_train, y_train = down_sample(X_train, y_train, pairs_thresh=10000)
    logging.info(f"[main]   - X_train (sampled): {len(X_train)}items")
    logging.info(f"[main]   - y_train (sampled): {len(y_train)}items")
    
    if search_param:
        model = None
        digit2model = {}
        for digit in range(-5, 6):
            gamma = 10 ** digit
            logging.info(f"[main]   - gamma: {gamma}")
            model = train_itml(X_train, y_train, gamma=gamma)
            pickle.dump(model, open(f"model_gamma-{gamma}.pkl", "wb"))
            digit2model[digit] = model
    else:
        model = train_itml(X_train, y_train)
        logging.info("[main] - training finished")
        pickle.dump(model, open("model.pkl", "wb"))

    logging.info("[main] evaluate itml ...")
    if dev_data_dir is not None or dev_data_pathes is not None:
        logging.info("[main] - dev set")
        evaluate_performance(dev_data_dir, dev_data_pathes, model, digit2model)
    logging.info("[main] - test set")
    evaluate_performance(test_data_dir, test_data_pathes, model, digit2model)


def cli_main():
    logging.basicConfig(level=logging.DEBUG, filename="main_ot_debug.log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", help="path of training data directory (lexeme_FILENAME.pkl)")
    parser.add_argument("--dev_data_dir", help="path of dev data directory (lexeme_FILENAME.pkl)")
    parser.add_argument("--test_data_dir", help="path of test data directory (lexeme_FILENAME.pkl)")
    parser.add_argument("--train_data_pathes", nargs="*", help="path of training data files (lexeme_FILENAME.pkl)")
    parser.add_argument("--dev_data_pathes", nargs="*", help="path of dev data files (lexeme_FILENAME.pkl)")
    parser.add_argument("--test_data_pathes", nargs="*", help="path of test data files (lexeme_FILENAME.pkl)")
    parser.add_argument("--search_param", action="store_true", help="conduct hyperparameter search or not")
    
    args = parser.parse_args()
    main(args.train_data_dir, args.dev_data_dir, args.test_data_dir,
         args.train_data_pathes, args.dev_data_pathes, args.test_data_pathes, 
         args.search_param)


if __name__ == "__main__":
    cli_main()
