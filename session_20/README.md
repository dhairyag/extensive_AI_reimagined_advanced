## Tokenizer for Devnagari Script (BPE Algorithm)
The code can generate tokens based on the input text files. Any devnagari language (Marathi, Sanskrit, Hindi etc) text can be provided for training. All divnagari letters and numbers along with special characters are used as initial vocabulary from which merge rules are created based on Byte Pair Encoding algorithm. 

### Execution
- Keep all training text in `/input_text_files`. All `.txt` files in the folder are imported for training.
- Update the list `merge_range` with required number of merges to be used 
- Run the code as `python3 main.py`

### Result: Compression Ratio
The tokenizer was executed with multiple values of merges and resultant compression achieved on the training data is shown here. As expected, compression improves with number of tokens. (The text data used for training and getting this particular plot is not part of sample data here. Check HuggingFace app below to check high tokens demo.)
![Test Image 1](plots/compr_ratio.png)


### Dependancy
- python3
- Install `regex`
- 'Good' input text for training

### Output
Final vocabulary and merge combinations are written in `.json` files (`vocab_<num>.json` and `merges_<num>.json`) and the statistics related to the compression ratio obtained is written in `quant_<num>.txt`.

### HuggingFace App as demo
[An app](https://huggingface.co/spaces/dhairyashil/marathi_tokenizer) based on `n_merges=15,000` and trained on around ~10MB of Marathi language text is provided on HuggingFace.

### Referance
- [Tokenization tutorial by Karpathy](https://youtu.be/zduSFxRajkE?si=RvSu-MN5sikGIW4w)
