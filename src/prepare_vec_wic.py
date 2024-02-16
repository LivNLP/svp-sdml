import argparse
import logging
import os
import pickle
import torch

from tqdm import tqdm
from typing import List, Tuple

from WordTransformer import WordTransformer, InputExample
from preprocess_wic import load_example, load_lexeme


def get_file_pathes(data_dir: str, data_type: str) -> Tuple[List[str], List[str]]:
    """ get file pathes from directory
    :param data_dir: str, path of data dir
    :param data_type: str, 'xl-lexeme' (xl-lexeme dir) or 'processed' (preprocess_wic.py)
    :return: file_pathes
    """
    file_pathes = []
    file_names = []
    if data_type == "xl-lexeme":
        for file_name in os.listdir(data_dir):
            if file_name[-4:] == "gold":
                continue
            file_path = os.path.join(data_dir, file_name[:-5])
            file_pathes.append(file_path)
            file_names.append(file_name[:-5])
    else:
        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            file_pathes.append(file_path)
            file_names.append(file_name[:-4])

    return file_pathes, file_names


def read_file(file_path: str, data_type: str) -> Tuple[List[List[str]], List[List[int]], List[int]]:
    """ read file
    :param file_path: str, path of file
    :param data_type: str, 'xl-lexeme' (xl-lexeme dir) or 'processed' (preprocess_wic.py)
    :return: pairs, positions, labels
    """
    if data_type == "xl-lexeme":
        pairs, positions, labels = load_lexeme(file_path)
    else:
        pairs, positions, labels = load_example(file_path)

    return pairs, positions, labels


def main(data_dir: str, data_type: str) -> None:
    logging.basicConfig(filename="wordvec_from_lexeme_wic_debug.log", format="%(asctime)s %(message)s", level=logging.DEBUG)

    logging.info("[main] 1. load model...")
    model = WordTransformer('pierluigic/xl-lexeme')
    model.to("cpu")
    
    logging.info("[main] 2. load files...")
    file_pathes, file_names = get_file_pathes(data_dir, data_type)
    logging.debug(f"[main] file_names: {file_names}")

    logging.info("[main] 3. obtain vectors...")
    for file_id in tqdm(range(len(file_pathes))):
        file_path = file_pathes[file_id]
        file_name = file_names[file_id]
        pairs, positions, labels = read_file(file_path, data_type)
        assert len(pairs) == len(positions) == len(labels)
        logging.debug(f"[main] file_path: {file_path}, {len(pairs)} items")

        vector_pairs = []
        for curr_id in tqdm(range(len(pairs))):
            context_1, context_2 = pairs[curr_id]
            position_1, position_2 = positions[curr_id]
            label = labels[curr_id]
            example_1 = InputExample(texts=context_1, positions=position_1)
            example_2 = InputExample(texts=context_2, positions=position_2)
            vectors = model.encode([example_1, example_2], show_progress_bar=False)
            vector_pairs.append(vectors)

        assert len(vector_pairs) == len(labels)

        data_dict = {"pairs": vector_pairs,
                     "labels": labels}

        pickle.dump(data_dict, open(f"lexeme_{file_name}.pkl", "wb"))

    logging.info("[main] finished")



def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="path of data directory")
    parser.add_argument("--data_type", default="xl-lexeme", help="data type, 'xl-lexeme' (xl-lexeme dir) or 'processed' (preprocess_wic.py)")
    args = parser.parse_args()
    main(args.data_dir, args.data_type)


if __name__ == "__main__":
    cli_main()
