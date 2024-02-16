import argparse
import json
import logging

from typing import List, Tuple


def find_target_from_context(context: str) -> Tuple[str, List[int]]:
    """ find target word (AM2iCo: <word>target</word>)
    :param context: str, target word is included
    
    :return:
     - processed_context: str
     - positions: List[int], char ids in processed_context that target word starts/ends
    """
    words = context.split()
    processed_words = []
    is_space_separated_lang = len(words) > 3
    target_word_id = None
    start_id = 0
    end_id = None
    for curr_word_id, curr_word in enumerate(words):
        if len(curr_word) < 4:
            processed_words.append(curr_word)
        elif curr_word[:6] == "<word>":
            target_word_id = curr_word_id
            target_word = curr_word[6:-7]
            logging.debug(f"[find_target_from_context] {target_word_id}\t{target_word}")
            end_id = start_id + len(target_word)
            processed_words.append(target_word)
        else:
            processed_words.append(curr_word)

        if end_id == None:
            start_id += len(curr_word) + is_space_separated_lang
    if is_space_separated_lang: 
        processed_context = " ".join(processed_words)
    else:
        processed_context = "".join(processed_words)
    
    return processed_context, [start_id, end_id]


def preprocess_am2ico(file_path: str) -> Tuple[List[List[str]], List[List[int]], List[int]]:
    """ Preprocess AM2iCo dataset
    :param file_path: str, path of AM2iCo dataset (train/dev/test .tsv)

    :return:
     - pairs: List[List[str]], List of sentence pairs 
     - positions: List[List[int]], List of start/end positions of target word
     - labels: List[int], label
    """
    logging.info(f"[preprocess_am2ico] file: {file_path}")
    pairs = []
    positions = []
    labels = []
    with open(file_path) as fp:
        for line in fp:
            context_1, context_2, label = line.strip().split("\t")
            logging.debug(f"[preprocess_am2ico] - context_1: {context_1}")
            logging.debug(f"[preprocess_am2ico] - context_2: {context_2}")
            logging.debug(f"[preprocess_am2ico] - label: {label}")
            if label not in set(["T", "F"]):
                continue

            processed_context_1, positions_1 = find_target_from_context(context_1) 
            processed_context_2, positions_2 = find_target_from_context(context_2) 
            label = 1 if label=="T" else 0
            logging.debug(f"[preprocess_am2ico] - context_1 (processed): {processed_context_1}")
            logging.debug(f"[preprocess_am2ico]   - target: {processed_context_1[positions_1[0]:positions_1[1]]}")
            logging.debug(f"[preprocess_am2ico] - context_2 (processed): {processed_context_2}")
            logging.debug(f"[preprocess_am2ico]   - target: {processed_context_2[positions_2[0]:positions_2[1]]}")
            logging.debug(f"[preprocess_am2ico] - label: {label}")

            pairs.append([processed_context_1, processed_context_2])
            positions.append([positions_1, positions_2])
            labels.append(label)

    return pairs, positions, labels


def preprocess_mclwic(data_path: str, gold_path: str) -> Tuple[List[List[str]], List[List[int]], List[int]]:
    """ Preprocess MCL-WiC dataset
    :param data_path: str, path of XLWiC dataset (train/dev/test .data)
    :param gold_path: str, path of XLWiC goldset (train/dev/test .gold)

    :return:
     - pairs: List[List[str]], List of sentence pairs 
     - positions: List[List[int]], List of start/end positions of target word
     - labels: List[int], label
    """
    pairs = []
    positions = []
    labels = []

    with open(data_path) as fp:
        data_items = json.load(fp)

    with open(gold_path) as fp:
        gold_items = json.load(fp)

    logging.debug(f"[preprocess_mclwic] data {len(data_items)}items, gold {len(gold_items)}items")

    for data_item, gold_item in zip(data_items, gold_items):
        if data_item["id"] != gold_item["id"]:
            continue
        context_1 = data_item["sentence1"]
        context_2 = data_item["sentence2"]

        if "start1" in data_item:
            start_1 = data_item["start1"]
            start_2 = data_item["start2"]
            end_1 = data_item["end1"]
            end_2 = data_item["end2"]
            ranges_1 = [f"{start_1}-{end_1}"]
            ranges_2 = [f"{start_2}-{end_2}"]
        else:
            ranges_1 = data_item["ranges1"]
            ranges_2 = data_item["ranges2"]

            ranges_1 = ranges_1.split(",")
            ranges_2 = ranges_2.split(",")

        label = gold_item["tag"]
        label = 1 if label == "T" else 0

        for range_1 in ranges_1:
            for range_2 in ranges_2:
                start_1 = range_1.split("-")[0]
                end_1 = range_1.split("-")[1]
                start_2 = range_2.split("-")[0]
                end_2 = range_2.split("-")[1]
                start_1 = int(start_1)
                start_2 = int(start_2)
                end_1 = int(end_1)
                end_2 = int(end_2)

                logging.debug(f"[preprocess_mclwic] - target: {data_item['lemma']}")
                logging.debug(f"[preprocess_mclwic]   - target(1): {context_1[start_1:end_1]}")
                logging.debug(f"[preprocess_mclwic]   - target(2): {context_2[start_2:end_2]}")

                pairs.append([context_1, context_2])
                positions.append([[start_1, end_1], [start_2, end_2]])
                labels.append(label)

    return pairs, positions, labels


def preprocess_xlwic(file_path: str) -> Tuple[List[List[str]], List[List[int]], List[int]]:
    """ Preprocess XLWiC dataset
    :param file_path: str, path of XLWiC dataset (train/dev/test .txt)

    :return:
     - pairs: List[List[str]], List of sentence pairs 
     - positions: List[List[int]], List of start/end positions of target word
     - labels: List[int], label
    """
    pairs = []
    positions = []
    labels = []
    with open(file_path) as fp:
        for line in fp:
            target_word, _, start_1, end_1, start_2, end_2, context_1, context_2, label = line.strip().split("\t")
            logging.debug(f"[preprocess_xlwic] - context_1: {context_1}")
            logging.debug(f"[preprocess_xlwic] - context_2: {context_2}")
            logging.debug(f"[preprocess_xlwic]   - target: {target_word}")
            logging.debug(f"[preprocess_xlwic] - label: {label}")

            label = int(label)

            pairs.append([context_1, context_2])
            positions.append([[start_1, end_1], [start_2, end_2]])
            labels.append(label)

    return pairs, positions, labels


def save_example(pairs: List[List[str]], 
                 positions: List[List[int]], 
                 labels: List[int], 
                 filename: str = "test") -> None:
    """ Save processed examples
    :param pairs: List[List[str]], List of sentence pairs 
    :param positions: List[List[int]], List of start/end positions of target word
    :param labels: List[int], label
    :param filename: str, output name
    """
    with open(f"{filename}.tsv", "w") as fp:
        for pair, position, label in zip(pairs, positions, labels):
            context_1, context_2 = pair
            position_1, position_2 = position
            start_1, end_1 = position_1
            start_2, end_2 = position_2

            fp.write(f"{context_1}\t{context_2}\t")
            fp.write(f"{start_1}\t{end_1}\t")
            fp.write(f"{start_2}\t{end_2}\t")
            fp.write(f"{label}\n")


def load_example(file_path: str) -> Tuple[List[List[str]], List[List[int]], List[int]]:
    """ Load processed examples
    :param file_path: path of file
    :return:
     - pairs: List[List[str]], List of sentence pairs 
     - positions: List[List[int]], List of start/end positions of target word
     - labels: List[int], label
    """ 
    pairs = []
    positions = []
    labels = []

    with open(file_path) as fp:
        for line in fp:
            items = line.strip().split("\t")
            context_1 = items[0]
            context_2 = items[1]
            start_1 = int(items[2])
            end_1 = int(items[3])
            start_2 = int(items[4])
            end_2 = int(items[5])
            label = int(items[6])
            pairs.append([context_1, context_2])
            positions.append([[start_1, end_1], [start_2, end_2]])
            labels.append(label)

    return pairs, positions, labels
            

def load_lexeme(file_path: str) -> Tuple[List[List[str]], List[List[int]], List[int]]:
    """ Load xl-lexeme dataset
    :param file_path: path of file 
    :return:
     - pairs: List[List[str]], List of sentence pairs 
     - positions: List[List[int]], List of start/end positions of target word
     - labels: List[int], label
    """
    return preprocess_mclwic(f"{file_path}.data", f"{file_path}.gold")


def cli_main():
    logging.basicConfig(level=logging.DEBUG, filename="preprocess_wic_debug.log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--am2ico_path", help="path of AM2iCo dataset")
    parser.add_argument("--mclwic_data", help="path of MCL-WiC dataset")
    parser.add_argument("--mclwic_gold", help="path of MCL-WiC goldset")
    parser.add_argument("--xlwic_path", help="path of XLWiC dataset")
    parser.add_argument("--filename", help="output name")

    args = parser.parse_args()

    if args.am2ico_path:
        pairs, positions, labels = preprocess_am2ico(args.am2ico_path)
        save_example(pairs, positions, labels, args.filename)

    if args.mclwic_data and args.mclwic_gold:
        pairs, positions, labels = preprocess_mclwic(args.mclwic_data, args.mclwic_gold)
        save_example(pairs, positions, labels, args.filename)

    if args.xlwic_path:
        pairs, positions, labels = preprocess_xlwic(args.xlwic_path)
        save_example(pairs, positions, labels, args.filename)



if __name__ == "__main__":
    cli_main()
