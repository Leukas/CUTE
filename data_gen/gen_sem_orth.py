# gen_sem_orth.py
import os
import nltk
import numpy as np
from nltk.tag import pos_tag
from Levenshtein import ratio 
import faiss
import random

def load_vocab(): # get 10000 most used words longer than 2 chars
    vocab = set()
    with open('./data_gen/unigram_freq.csv', 'r') as f:
        for line in f:
            word, _ = line.split(",")
            if len(word) < 3:
                continue
            if len(vocab) == 10000:
                break
            vocab.add(word.strip())
    return vocab

def load_fasttext(path):
    vectors = {}
    with open(path, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            vals = np.asarray(values[1:], dtype='float32')
            vectors[word] = vals / np.linalg.norm(vals)

            if i > 50000:
                break
    return vectors


def replace_plus_one(text):
    """ Replace every letter with the next letter in the alphabet. e.g. a->b, b->c, ... z->a """
    return text.translate(str.maketrans("abcdefghijklmnopqrstuvwxyz", "bcdefghijklmnopqrstuvwxyza"))

def get_and_sort_near_spells(word, candidates):
    return sorted([w for w in candidates if ratio(word, w) > 0.7], key=lambda x: ratio(word, x), reverse=True)

def get_far_spells(word, candidates):
    return [w for w in candidates if ratio(word, w) < 0.3]

def get_pair(knn, vectors, int2str, word, vocab):
    word_vec = vectors[word].reshape(1, -1)
    dists, inds_near = knn.range_search(word_vec, 0.5)[1:] # cossim > 0.5
    inds_near = inds_near[np.argsort(dists)][::-1]
    dists, inds = knn.range_search(word_vec, 0.2)[1:] # cossim < 0.2
    inds = inds[np.argsort(dists)][::-1]
    inds_all = np.arange(len(vectors))

    inds_far = np.setdiff1d(inds_all, inds)
    strs_far = [int2str[i].lower() for i in inds_far]
    strs_near = [int2str[i].lower() for i in inds_near]

    orth_words = get_and_sort_near_spells(word, strs_far) # Levenshtein > 0.7
    sem_words = get_far_spells(word, strs_near) # Levenshtein < 0.3
    
    orth_word = None
    sem_word = None
    for i in range(len(orth_words)):
        if orth_words[i] in vocab:
            orth_word = orth_words[i]
            break
    for i in range(len(sem_words)):
        if sem_words[i] in vocab:
            sem_word = sem_words[i]
            break

    return sem_word, orth_word

def main():
    path = "./data/"
    os.makedirs(path, exist_ok=True)

    random.seed(0)

    vocab = load_vocab()
    # load fasttext
    vectors = load_fasttext("./data_gen/wiki-news-300d-1M.vec")
    knn = faiss.IndexFlatIP(300)
    knn.add(np.array(list(vectors.values())))
    
    print("Loaded fasttext.")

    int2str = list(vectors.keys())

    # load unigram_freq.csv
    wsos = []    
    with open("./data_gen/unigram_freq.csv", 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[1:]):
            if len(wsos) >= 1000 or i > 10000:
                break
            word, freq = line.split(",")

            word_pos = pos_tag([word], tagset='universal')[0][1]
            if word_pos not in ["NOUN", "VERB", "ADJ", "ADV"]: # filter out stuff like punctuation, determiners, etc.
                continue
            try:
                sem, orth = get_pair(knn, vectors, int2str, word, vocab)

                if sem is None or orth is None:
                    continue

            except IndexError:
                print("Not enough neighbors: ", word)
                continue
            except KeyError:
                print("Word not in fasttext: ", word)
                continue

            wsos.append(f"{word}\t{sem}\t{orth}")

    # swap order of orth and sem so there's no positional bias when prompting
    swapped_wsos = []
    for wso in wsos:
        wso_split = wso.split("\t")
        if random.random() > 0.5:
            wso_split[1], wso_split[2] = wso_split[2], wso_split[1]
        wso_joined = "\t".join(wso_split)
        swapped_wsos.append(wso_joined)

    with open(path+"sem.tsv", 'w') as f:
        f.write("input1\tinput2\tinput3\tlabel\n")
        for i in range(len(swapped_wsos)):
            wso = swapped_wsos[i]
            label = wsos[i].split("\t")[1]
            with_label = f"{wso}\t{label}"
            f.write(f"{with_label}\n")

    with open(path+"orth.tsv", 'w') as f:
        f.write("input1\tinput2\tinput3\tlabel\n")
        for i in range(len(swapped_wsos)):
            wso = swapped_wsos[i]
            label = wsos[i].split("\t")[2]
            with_label = f"{wso}\t{label}"
            f.write(f"{with_label}\n")


if __name__ == "__main__":
    main()