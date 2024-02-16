import argparse
import logging
import pickle

import numpy as np
import metric_learn
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from tqdm import tqdm

from typing import Dict, List, Tuple


def _load_gold(gold_path: str) -> Dict[str, float]:
    word2gold = {}
    with open(gold_path) as fp:
        for line in fp:
            word_pos, gold_str = line.strip().split("\t")
            word = word_pos.split("_")[0]
            word2gold[word] = float(gold_str)
    logging.debug(f"[_load_gold] - word2gold: {word2gold}")

    return word2gold


def estimate_score(target_words: List[str],
                   model,
                   word2vecs_t0: Dict[str, List[List[float]]],
                   word2vecs_t1: Dict[str, List[List[float]]]) -> Dict[str, float]:
    word2pred = {}
    if model is None:
        metric = cosine
    else:
        metric = model.get_metric()
    for target_word in tqdm(target_words):
        #vecs_t0 = word2vecs_t0[target_word]
        #vecs_t1 = word2vecs_t1[target_word]
        apd = 0
        count = 0
        for vec_t0 in word2vecs_t0[target_word]:
            for vec_t1 in word2vecs_t1[target_word]:
                dist = metric(vec_t0, vec_t1)
                apd += dist
                count += 1
        word2pred[target_word] = apd / count
    logging.debug(f"[estimate_score] - word2pred: {word2pred}")

    return word2pred


def evaluate_ot(gold_path: str,
                model_path: str,
                vec_pathes: List[str]):
    logging.info("[evaluate_ot] load pathes ...")
    word2gold: Dict[str, float] = _load_gold(gold_path)
    model = pickle.load(open(model_path, "rb"))
    word2vecs_t0 = pickle.load(open(vec_pathes[0], "rb"))
    logging.debug(f"[evaluate_ot] - word2vecs_t0: {word2vecs_t0.keys()}")
    word2vecs_t1 = pickle.load(open(vec_pathes[1], "rb"))
    logging.debug(f"[evaluate_ot] - word2vecs_t1: {word2vecs_t1.keys()}")

    logging.info("[evaluate_ot] make prediction ...")
    word2pred_itml: Dict[str, float] = estimate_score(word2gold.keys(), model, word2vecs_t0, word2vecs_t1)

    logging.info("[evaluate_ot] evaluation ...")
    golds = []
    preds_itml = []
    for word in word2gold.keys():
        gold = word2gold[word]
        pred_itml = word2pred_itml[word]
        golds.append(gold)
        preds_itml.append(pred_itml)

    rho_itml, p_itml = spearmanr(np.array(golds), np.array(preds_itml))
    logging.info("[evaluate_ot] itml")
    logging.info(f"[evaluate_ot] - spearman: {rho_itml}")
    logging.info(f"[evaluate_ot] - p: {p_itml}")


def cli_main():
    logging.basicConfig(level=logging.DEBUG, filename="evaluate_ot_debug.log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", help="path to gold file: word[tab]gold_score format")
    parser.add_argument("--model_path", help="path to trained itml.pkl path")
    parser.add_argument("--vec_pathes", nargs=2, help="path to word2vec.pkl for evaluation")

    args = parser.parse_args()

    evaluate_ot(args.gold_path,
                args.model_path,
                args.vec_pathes)

if __name__ == "__main__":
    cli_main()
