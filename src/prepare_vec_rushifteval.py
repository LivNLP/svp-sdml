import argparse
import logging
import os
import pickle
import torch

from tqdm import tqdm
from typing import List

from WordTransformer import WordTransformer, InputExample


def collect_usage(target_word: str, file_path: str) -> List[str]:
    """ obtain usages that target word occurs
    :param target_word: str, target word
    :param file_path: str, path to target corpus
    """
    group2items = {}
    with open(f"{file_path}/{target_word}/uses.csv") as fp:
        for line in fp:
            items = line.strip().split("\t")
            if items[0] == "lemma" and items[1] == "pos":
                continue
            group = items[3]
            sentence = items[6] 
            positions_str = items[7].split(":")
            positions = [int(position) for position in positions_str]
            if group not in group2items:
                group2items[group] = []
            group2items[group].append((sentence, positions))

    return group2items


def main(args) -> None:
    os.makedirs("../../results", exist_ok=True)
    logging.basicConfig(filename="../../results/wordvec_from_lexeme_debug.log",
                        format="%(asctime)s %(message)s", level=logging.DEBUG)

    logging.info(f"[main] args: {args}")

    logging.info("[main] 1. load model...")
    model = WordTransformer('pierluigic/xl-lexeme')
    model.to("cpu")

    logging.info("[main] 2. obtain target words...")
    target_words = []
    with open(args.target_words_list) as fp:
        for line in fp:
            target_word = line.strip()
            target_words.append(target_word)

    logging.info("[main] 3. obtain target vectors...")
    word2vec_t0 = {}
    word2vec_t1 = {}
    for target_word in tqdm(target_words):
        word2vec_t0[target_word] = []
        word2vec_t1[target_word] = []
        target_group2items = collect_usage(target_word, args.file_path)
        groups = list(target_group2items.keys())
        logging.debug(f"[main] - {target_word}: ")
        logging.debug(f"[main]   - t0 {len(target_group2items[groups[0]])} sents")
        logging.debug(f"[main]   - t1 {len(target_group2items[groups[1]])} sents")

        batch_size = 16
        examples = []
        for sentence, positions in tqdm(target_group2items[groups[0]], desc=f"[obtain {target_word}'s vectors...]"):
            example = InputExample(texts=sentence, positions=positions)
            examples.append(example)
            if len(examples) >= batch_size:
                target_vectors = model.encode(examples, show_progress_bar=False)
                word2vec_t0[target_word].extend(list(target_vectors))
                examples = []
        if len(examples) >= 0:
            target_vectors = model.encode(examples, show_progress_bar=False)
            word2vec_t0[target_word].extend(list(target_vectors))
            examples = []
        logging.debug(f"[main]   - {len(word2vec_t0[target_word])} items")

        examples = []
        for sentence, positions in tqdm(target_group2items[groups[1]], desc=f"[obtain {target_word}'s vectors...]"):
            example = InputExample(texts=sentence, positions=positions)
            examples.append(example)
            if len(examples) >= batch_size:
                target_vectors = model.encode(examples, show_progress_bar=False)
                word2vec_t1[target_word].extend(list(target_vectors))
                examples = []
        if len(examples) >= 0:
            target_vectors = model.encode(examples, show_progress_bar=False)
            word2vec_t1[target_word].extend(list(target_vectors))
            examples = []

        logging.debug(f"[main]   - {len(word2vec_t1[target_word])} items")
    
    pickle.dump(word2vec_t0, open(f"../../results/lexeme_{args.output_name}_t0_word2vec.pkl", "wb"))
    pickle.dump(word2vec_t1, open(f"../../results/lexeme_{args.output_name}_t1_word2vec.pkl", "wb"))

    logging.info("[main] finished")



def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", help="path of corpus (rushifteval raw_rushifteval_i/data)")
    parser.add_argument("-l", "--target_words_list", help="target word list")
    parser.add_argument("-o", "--output_name")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
