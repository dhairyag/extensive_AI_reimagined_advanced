import os
import regex as re
import concurrent.futures
import json

def read_text_files(folder_path):
    """
    Read all text files from a given folder and store their contents in one variable.
    """
    combined_text = ""
    for filename in os.listdir(folder_path):
        print(filename)
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                combined_text += file.read() + " "  # Add a space to separate contents of different files
    #print(combined_text[:100], combined_text[-100:], len(combined_text))
    return combined_text

def clean_text(text):
    gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    return re.findall(gpt2pat, text)
    
def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    id = 9
    #print(f"ids {ids} pair {pair} idx {idx}")
    #print("lllr")
    while i < len(ids):
        #print("newids ",newids)
        #if i < len(id) and id[i] == pair[0] and id[i+1] == pair[1]:
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            #print("mer ids-i ",ids[i], i, newids)
            newids.append(ids[i])
            i += 1
    return newids

def train(ids, num_merges, univ_vocab):
    merges = {} # (int, int) -> int
    #print("ids",ids)

    for i in range(num_merges):
        stats = {}
        for word_id in ids:
            stats = get_stats(word_id, stats)
        #print("stats",stats)
        pair = max(stats, key=stats.get)
        if stats[pair] < 2:
            break
        #print("pair",pair)
        idx = 10000 + i
        #print(f"merging {pair} into a new tolen {idx}")
        ids = [merge(word_id, pair, idx) for word_id in ids]
        merges[pair] = idx
        univ_vocab[idx] = univ_vocab[pair[0]] + univ_vocab[pair[1]]
        #print(univ_vocab[idx], univ_vocab[pair[0]], univ_vocab[pair[1]])
    
    return merges, univ_vocab

def get_vocab(merges, univ_vocab):
    #vocab = {sr: chr(idx) for sr, idx in enumerate (uni_chars)}
    for (p0, p1), idx in merges.items():
        univ_vocab[idx] = univ_vocab[p0] + univ_vocab[p1]
    return univ_vocab

def get_init_vocab(u_ids):
    vocab = {idx: chr(idx) for idx in u_ids}
    #print("init vocab", vocab)
    #print(u_ids)
    return vocab

def decode(ids, vocab):
    # given ids , return Python strings
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

def encode(text, merges):
    # given string, return list of ints
    print(text)
    tokens = list(map(ord, text)) #list(text.encode("utf-8"))
    print(tokens)
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        #print("stats ", stats)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break # nothing else can be merged
        #print("merges",merges)
        idx = merges[pair]
        #print("idx ",idx)
        tokens = merge(tokens, pair, idx)
        #print("encode token ",tokens)
    #print("done endode tok ", tokens)
    return tokens

def encode_ordinary(text, merges):
    """Encoding that ignores any special tokens."""
    # split text into chunks of text by categories defined in regex pattern
    text_chunks = clean_text(text) #re.findall(compiled_pattern, text)
    # all chunks of text are encoded separately, then results are joined
    ids = []
    for chunk in text_chunks:
        #chunk_bytes = chunk.encode("utf-8") # raw bytes
        chunk_ids = encode(chunk, merges)
        ids.extend(chunk_ids)
        #print("encode ord ",ids)
    
    return ids
    
corpus = read_text_files("./all_text/")
corpus_words = clean_text(corpus)
print("corpus_lines ", corpus_words)

corpus = ""
for word in corpus_words:
    corpus += word
print(corpus)

tokens = list(map(ord, corpus))

print(tokens)

ids = [list(map(ord, word)) for word in corpus_words]

#list(set(tokens))
unique_char_ids = list(set(tokens))
print(" unique_char_ids ", unique_char_ids)

univ_vocab = get_init_vocab(unique_char_ids)
print(univ_vocab)

with open("quant.txt", 'w', encoding='utf-8') as file:
    wfile = "num_merges n_tokens n_orig_corpus_chars n_tokens_corpus compression"
    file.write(wfile)

n_orig_corpus_chars = len(corpus)

#vocab_size = 00

def main(num_merges):
    global n_orig_corpus_chars, corpus_words, univ_vocab, corpus
    print(" n merges ", num_merges)
    
    univ_vocab_copy = univ_vocab.copy()
    # create a BPE tokenizer object
    merges, univ_vocab_copy = train(ids, num_merges, univ_vocab_copy)
    #print("merges",merges)
    #print("vocab1",univ_vocab)
    
    # train BPE tokenizer with Wikipedia corpus
    #print(toks)
    n_tokens = len(univ_vocab_copy)
    #print(f"\n\nBPE tokenization result of text\n\n'{text}'")
    tokens_corpus = encode_ordinary(corpus, merges)
    n_tokens_corpus = len(tokens_corpus)
    #print(" n_tokens_corpus for voc size ", vocab_size, " -- ", n_tokens_corpus)

    # Save vocab to a JSON file
    with open(f'vocab_{pdf}.json', 'w', encoding='utf-8') as vocab_file:
        json.dump(univ_vocab_copy, vocab_file, ensure_ascii=False, indent=4)

    # Save merges to a text file
    with open(f'merges_{pdf}.txt', 'w', encoding='utf-8') as merges_file:
        for combo, id in merges.items():
            merges_file.write(' '.join(map(str, combo)) + f' {id}\n')
    
    with open("quant.txt", 'a', encoding='utf-8') as file:
        wfile = f"\n{num_merges} {n_tokens} {n_orig_corpus_chars} {n_tokens_corpus} {n_orig_corpus_chars/n_tokens_corpus}"
        file.write(wfile)
    
    return True

'''
merges, univ_vocab_copy = main(2000)
with open(f'vocab_2.json', 'w', encoding='utf-8') as vocab_file:
    json.dump(univ_vocab_copy, vocab_file, ensure_ascii=False, indent=4)

# Save merges to a text file
with open(f'merges_2.txt', 'w', encoding='utf-8') as merges_file:
    for combo, id in merges.items():
        merges_file.write(' '.join(map(str, combo)) + f' {id}\n')

toks = encode_ordinary(corpus, merges)
print(toks)
'''

results = {}
merge_range = list(range(1000,4000,1000))
#num_threads = 12
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_pdf = {executor.submit(main, ii): ii for ii in merge_range}
    for future in concurrent.futures.as_completed(future_to_pdf):
        pdf = future_to_pdf[future]
        try:
            flag = future.result()
            results[pdf] = flag
            
        except Exception as e:
            print(f" error {e}")
            #results[vocab_s] = f"Error: {e}"


