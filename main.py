# main.py
import torch
import os
import argparse
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from functools import partial
from torch.utils.data import DataLoader

from tqdm import tqdm

from prompts import PROMPTS

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama-7b", help="Model name.")
parser.add_argument("--model_path", type=str, default="", help="Model path. Only necessary if \
                    model path is different from that specified in MODELS.")
parser.add_argument("--big_model", action="store_true", help="Load a model across multiple gpus. \
                    Likely necessary for command-r-plus and dbrx.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--task", type=str, default="spell", help="Benchmark task. Possible values: \
                    {spell, spell_inverse, contains_char, contains_word, orth, sem, ins_char, ins_word, \
                    del_char, del_word, swap_char, swap_word, sub_char, sub_word}.")

TASKS = [
    'spell',
    'spell_inverse',
    'contains_char',
    'contains_word',
    'orth',
    'sem',
    'ins_char',
    'ins_word',
    'del_char',
    'del_word',
    'sub_char',
    'sub_word',
    'swap_char',
    'swap_word',
]

CHAR_TASKS = [
    'ins_char',
    'del_char',
    'sub_char',
    'swap_char',
    'contains_char',
]

WORD_TASKS = [
    'ins_word',
    'del_word',
    'sub_word',
    'swap_word',
    'contains_word',
]

OTHER_TASKS = [
    'spell',
    'spell_inverse',
    'orth',
    'sem',
]

RAND_TASKS = [
    'spell_rand',
    'spell_inverse_rand',
    'contains_char_rand',
    'ins_char_rand',
    'del_char_rand',
    'sub_char_rand',
    'swap_char_rand',
]

MODELS = {
    'llama-7b': "meta-llama/Llama-2-7b-chat-hf",
    'llama-13b': "meta-llama/Llama-2-13b-chat-hf",
    'llama-70b': "meta-llama/Llama-2-70b-chat-hf",
    'llama3-8b': "meta-llama/Meta-Llama-3-8B-Instruct",
    'llama3-70b': "meta-llama/Meta-Llama-3-70B-Instruct",
    'mistral-7b': "mistralai/Mistral-7B-Instruct-v0.2",
    'mistral-8x7b': "mistralai/Mixtral-8x7B-Instruct-v0.1",
    'command-r': "CohereForAI/c4ai-command-r-v01-4bit",
    'command-r-plus': "CohereForAI/c4ai-command-r-plus-4bit",
    'dbrx': "PrunaAI/dbrx-instruct-bnb-4bit",
    'gemma-7b': "google/gemma-7b-it"
}

def padding_collate_fn(batch, max_len=1024):
    """ 
        Pads each list with zeros and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    padded_batch = {}
    for key in batch[0]:
        if key in ["input1", "input2", "input3", "label"]:
            padded_batch[key] = []
            continue
        largest = min(max_len, max([len(b[key]) for b in batch]))
        padded_batch[key] = torch.zeros((len(batch), largest), dtype=torch.long)
        if "labels" in key:
            padded_batch[key] -= 100
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            if key in ["input1", "input2", "input3", "label"]: 
                padded_batch[key].append(batch[i][key]) 
                continue
            key_len = min(max_len, len(sample[key]))
            padded_batch[key][i, -key_len:] = torch.LongTensor(sample[key][:key_len])

    return padded_batch

def add_prompt(examples, tokenizer, task):
    batch = {
        'input_ids':[],
        'attention_mask': [],
        'labels': []
    }

    for i in range(len(examples['input1'])):
        all_inputs = [examples['input1'][i]]
        if 'input2' in examples:
            all_inputs.append(examples['input2'][i])
        if 'input3' in examples:
            all_inputs.append(examples['input3'][i])

        messages = [
            {"role": "user", "content": PROMPTS[task].format(*all_inputs)}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        src = tokenizer.encode(prompt + " Answer: \"", return_tensors="pt")
        batch['input_ids'] += src
        tgt = tokenizer.encode(examples['label'][i], return_tensors="pt")
        batch['labels'] += tgt
        batch['attention_mask'] += torch.ones_like(src)

    return batch

def clean_output(outs):
    final_out = []
    for i in range(len(outs)):
        try:
            out = outs[i].split("Answer:")[-1] # remove everything before "Answer:"
            out = out.split("</s>")[0] # remove </s> and everything after
            out = out.split("\"")[1] # remove quotes, and stuff like "I hope this answer helped!"
        except: # something wasn't generated properly, discard
            final_out.append("")
            continue
        final_out.append(out)

    return final_out


def main(args):
    os.makedirs("model_outputs", exist_ok=True)

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )

    assert args.model in MODELS or args.model_path, f"Must specify model path or use model in: {MODELS.keys()}"
    model_path = args.model_path if args.model_path else MODELS[args.model]

    if args.big_model:
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config, 
                                                     device_map='auto', trust_remote_code=args.model not in ['command-r', 'command-r-plus'])
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config)
    model = model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.model not in ['command-r', 'command-r-plus'])

    dataset = load_dataset('csv', data_files=f"./data/{args.task}.tsv", split='train', sep="\t")

    print(dataset)

    tokenize = partial(add_prompt, tokenizer=tokenizer, task=args.task.replace("_rand",""))
    remove_columns = set(dataset.column_names) - set(['input_ids', 'labels', 'input1', 'input2', 'input3', 'label'])
    dataset = dataset.map(tokenize, batched=True, num_proc=10, remove_columns=remove_columns)

    # make labelset lowercase in case the first word is capitalized
    label_ds = dataset['label']
    if args.task in WORD_TASKS:
        label_ds = [label.lower() for label in label_ds]

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=padding_collate_fn,
    )

    with torch.no_grad():
        outputs = []
        for batch in tqdm(dl):
            out = model.generate(batch['input_ids'].cuda(), attention_mask=batch['attention_mask'].cuda(), 
                                max_new_tokens=100, do_sample=False, top_p=1, temperature=0)
            outs = tokenizer.batch_decode(out)
            
            true_outs = clean_output(outs)

            print(true_outs[0], f'\033[92m {batch["label"][0]} \033[0m')

            if args.task in WORD_TASKS:
                true_outs = [x.lower() for x in true_outs]
            outputs.extend(true_outs)

    # write outputs and labels to file
    with open(f"model_outputs/outputs.{args.model}.{args.task}.txt", "w") as f:
        for out in outputs:
            f.write(out + "\n")

    with open(f"model_outputs/labels.{args.model}.{args.task}.txt", "w") as f:
        for label in label_ds:
            f.write(label + "\n")


    # metric = evaluate.load('exact_match')
    # metric.add_batch(predictions=outputs, references=label_ds)
    # results = metric.compute()

    # exact same thing as above
    correct = sum(x == y for x, y in zip(outputs, label_ds))
    print(correct / len(outputs))

    # write score to file
    with open(f"model_outputs/score.{args.model}.{args.task}.txt", "w") as f:
        # f.write(f"{results['exact_match']:.4f}")
        f.write(f"{correct / len(outputs):.4f}")

if __name__ == "__main__":
    args = parser.parse_args()
    if "," in args.task:
        all_tasks = args.task.split(",")
        for task in all_tasks:
            args.task = task
            main(args)
    if args.task == 'all':
        for task in TASKS:
            args.task = task
            main(args)
    elif args.task == 'char':
        for task in CHAR_TASKS:
            args.task = task
            main(args)
    elif args.task == 'word':
        for task in WORD_TASKS:
            args.task = task
            main(args)
    elif args.task == 'other':
        for task in OTHER_TASKS:
            args.task = task
            main(args)
    else:
        main(args)