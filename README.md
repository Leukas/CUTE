<!-- README.md -->
### CUTE: Character-Level Understanding of Tokens Evaluation

To install dependencies, cd into the repo folder and run:
```bash
pip install -e .
```

To evaluate, run for example:
```bash
python main.py --model llama-7b --task contains_char
```


This will evaluate Llama-7B on our Contains Character task. All flags are explained here:
```
--model         Name of the model. If it is one of the supported models, it will automatically download the model from the Hub. Otherwise this only serves as a name for the outputs.
--model_path    Model path. Necessary if you use an unsupported model or your model is in a custom location.
--big_model     Supports loading a model across multiple gpus. Likely necessary for command-r-plus and dbrx.
--batch_size    Batch size.
--task          Benchmark task. Possible values: {spell, spell_inverse, contains_char, contains_word, orth, sem, ins_char, ins_word, del_char, del_word, swap_char, swap_word, sub_char, sub_word}. Can also use "all" to run all tasks, or use commas e.g. "spell,spell_inverse" to run multiple tasks.
```

The outputs, labels, and scores will be stored in:
```python
f"model_outputs/outputs.{model}.{task}.txt"
f"model_outputs/labels.{model}.{task}.txt"
f"model_outputs/score.{model}.{task}.txt"
```

The outputs of our experiments can already be found there. 
Note: there are a couple of values that were erroneously copied over to the table in the paper, (e.g. Llama2-7B on spelling should get 0.955 not 0.934). These will be fixed in the camera-ready.


##### Generating Data

If you want to re-generate the data using the scripts in ./data_gen/, you will also need to download the fastText word embeddings and put them in the data_gen folder:
https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

To regenerate the data, use `gen_sem_orth.py` for generating semantic and orthographic pairs, `gen_char_tasks.py` for all tasks on the character level (including spelling), and `gen_word_tasks.py` for all tasks on the word level. 
Note: There was a bug in earlier versions of the gen files that meant the seed was not reproducible. As such, running them now will not give you the same files as are provided in `./data`. 
