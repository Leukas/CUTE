# gen_char_tasks.py
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="spell", help="Benchmark task. Possible values: \
                    {spell, spell_inverse, contains_char, ins_char, del_char, swap_char, sub_char}.")

ALPHABET = set("abcdefghijklmnopqrstuvwxyz")
CONSONANTS = set("bcdfghjklmnpqrstvwxz")

def replace_plus_one(text):
    """ Replace every letter with the next letter in the alphabet. e.g. a->b, b->c, ... z->a """
    return text.translate(str.maketrans("abcdefghijklmnopqrstuvwxyz", "bcdefghijklmnopqrstuvwxyza"))

def gen_random(length):
    return "".join(random.sample(list(CONSONANTS), length))

def main():
    args = parser.parse_args()
    random.seed(0)

    inputs = []
    labels = []
    path = "./data/"
    os.makedirs(path, exist_ok=True)

    # load unigram_freq.csv
    lines = open("./data_gen/unigram_freq.csv", 'r').readlines()[1:]

    for line in lines:
        if len(inputs) > 1000:
            break

        word, freq = line.split(",")
        if "rand" in args.task:
            word = gen_random(len(word))

        if len(word) < 3:
            continue

        input_all = []
        if "spell" in args.task and "inverse" not in args.task:
            label = " ".join(word)
            input_all = [word]
        elif "spell_inverse" in args.task:
            input_all = [" ".join(word)]
            label = word
        elif "contains_char" in args.task:
            label = random.choice(["Yes", "No"])
            
            letters_in_word = set(word)
            if label == "No":
                try:
                    random_letter = random.choice(list(ALPHABET - letters_in_word))
                except IndexError:
                    print("The word is", word)
                    raise IndexError
            else:
                random_letter = random.choice(list(letters_in_word))

            input_all = [random_letter, word]
        
        elif "ins_char" in args.task:
            random_letter = random.choice(list(ALPHABET))
            random_char_in_word = random.choice(list(set(word)))
            label = word.replace(random_char_in_word, random_char_in_word+random_letter)
            input_all = [random_letter, random_char_in_word, word]
        elif "del_char" in args.task:
            random_char_in_word = random.choice(list(set(word)))
            label = word.replace(random_char_in_word, "")
            input_all = [random_char_in_word, word]
        elif "sub_char" in args.task:
            random_letter = random.choice(list(ALPHABET))
            random_char_in_word = random.choice(list(set(word)))
            label = word.replace(random_char_in_word, random_letter)
            input_all = [random_char_in_word, random_letter, word]
        elif "swap_char" in args.task:
            unique_chars_in_word = [x for x in set(word) if word.count(x) == 1]
            try:
                c1, c2 = random.sample(unique_chars_in_word, k=2)
            except ValueError: # not enough unique chars in word
                continue
            label = list(word)
            label[word.index(c1)] = c2
            label[word.index(c2)] = c1
            label = "".join(label)
            input_all = [c1, c2, word]
                
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