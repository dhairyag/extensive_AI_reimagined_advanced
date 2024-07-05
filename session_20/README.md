### Tokenizer for Devnagari Script (BPE Algorithm)
The code can generate tokens based on the input text files. Any devnagari language (Marathi, Sanskrit, Hindi etc) text can be provided for training. All divnagari letters and numbers along with special characters are used as initial vocabulary from which merge rules are created based on Byte Pair Encoding algorithm. 

#### Execution
- Keep all training text in `/input_text_files`. All `.txt` files in the folder are imported for training.
- Update the list `merge_range` with required number of merges to be used 
- Run the code as `python3 main.py`

#### Dependancy
- Install `regex`

#### Output
Final vocabulary and merge combinations are written in `.json` files (`vocab_<num>.json` and `merges_<num>.json`) and the statistics related to the compression ratio obtained is written in `quant_<num>.txt`.

#### Referance
- [Tokenization tutorial by Karpathy](https://youtu.be/zduSFxRajkE?si=RvSu-MN5sikGIW4w)
