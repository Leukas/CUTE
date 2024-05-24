# gen_word_tasks.py
import os
import nltk
import nltk.data
from sentence_splitter import SentenceSplitter
from datasets import load_dataset, Dataset
import random
import argparse
from sacremoses import MosesTokenizer, MosesDetokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="dupe_word", help="Benchmark task. Possible values: \
                    {contains_word, ins_word, del_word, swap_word, sub_word}.")

def load_vocab(): # get 1000 most used words longer than 2 chars
    vocab = set()
    with open('./data_gen/unigram_freq.csv', 'r') as f:
        for line in f:
            word, _ = line.split(",")
            if len(word) < 3:
                continue
            if len(vocab) == 1000:
                break
            vocab.add(word.strip())
    return vocab


def load_prep_data():
    dataset = load_dataset('roneneldan/TinyStories', split='validation')
    splitter = SentenceSplitter(language="en")

    split = []
    for i in range(len(dataset)):
        tokens = splitter.split(dataset[i]['text'])
        split.extend(tokens)
        
    split_ds = Dataset.from_dict({'text': split})

    # filter sentences longer than 10 words, shorter than 3 words
    split_ds = split_ds.filter(lambda x: len(x['text'].split()) <= 10)
    split_ds = split_ds.filter(lambda x: len(x['text'].split()) >= 3)

    # filter out sentences that don't end with a .!? (these are mostly garbage like titles)
    split_ds = split_ds.filter(lambda x: x['text'][-1] in ['.','!','?'])

    # filter out sentences with "
    split_ds = split_ds.filter(lambda x: '"' not in x['text'])

    # shuffle
    split_ds = split_ds.shuffle(seed=0)
    # print(split_ds[:10])
    return split_ds

def main():
    import numpy as np
    import torch


    args = parser.parse_args()
    path = "./data/"
    os.makedirs(path, exist_ok=True)


    dataset = load_prep_data()
    vocab = load_vocab()
    tokenizer = MosesTokenizer(lang='en')
    detokenizer = MosesDetokenizer(lang='en')

    # set seeds
    random.seed(0)

    inputs = []
    labels = []
    for i, sentence in enumerate(dataset):
        # tokenize
        words = tokenizer.tokenize(sentence['text'], escape=False, protected_patterns="'")
        input_all = []

        # need to apply sorted() so the seeding actually works (set doesn't have consistent order)
        random_word = random.choice(sorted(list(set(words) - set(".,;:!?\'\""))))
        
        if args.task == "contains_word":
            label = random.choice(["Yes", "No"])
            words_not_in_sent = vocab - set(words) 
            new_random_word = random.choice(sorted(list(words_not_in_sent)))
            
            if label == "No":
                input_all = [new_random_word, sentence['text']]
            else:
                input_all = [random_word, sentence['text']]
        elif args.task == "ins_word":
            random_word_from_vocab = random.choice(sorted(list(vocab)))
            inserted = []
            for w in words:
                inserted.append(w)
                if w == random_word:
                    inserted.append(random_word_from_vocab)

            label = detokenizer.detokenize(inserted, unescape=False)
            input_all = [random_word_from_vocab, random_word, sentence['text']]
        elif args.task == "del_word":
            deleted = []
            for w in words:
                if w != random_word:
                    deleted.append(w)

            label = detokenizer.detokenize(deleted, unescape=False)
            input_all = [random_word, sentence['text']]
        elif args.task == "sub_word":
            random_word_from_vocab = random.choice(sorted(list(vocab)))
            subbed = []
            for w in words:
                if w == random_word:
                    subbed.append(random_word_from_vocab)
                else:
                    subbed.append(w)
            label = detokenizer.detokenize(subbed, unescape=False)
            input_all = [random_word, random_word_from_vocab, sentence['text']]
        elif args.task == "swap_word":
            unique_words_in_sent = [x for x in set(words) if words.count(x) == 1]
            unique_words_in_sent = list(set(unique_words_in_sent) - set(".,;:!?\'\""))
            try:
                w1, w2 = random.sample(sorted(unique_words_in_sent), k=2)
            except ValueError:
                continue # not 2 unique words in sentence
            
            swapped = []
            for w in words:
                if w == w1:
                    swapped.append(w2)
                elif w == w2:
                    swapped.append(w1)
                else:
                    swapped.append(w)
            label = detokenizer.detokenize(swapped, unescape=False)
            input_all = [w1, w2, sentence['text']]
                
        inputs.append(input_all)
        labels.append(label)

    num_inputs = len(inputs[0])

    with open(path+f"{args.task}.tsv", 'w') as f:
        header = "{}label\n".format("".join([f"input{i}\t" for i in range(1, num_inputs+1)]))
        f.write(header)
        for inp, label in zip(inputs[:1000], labels[:1000]):
            inputs_str = "\t".join(inp)
            f.write(f"{inputs_str}\t{label}\n")


if __name__ == "__main__":
    main()