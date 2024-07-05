import os
import regex as re
import json

def read_text_files(folder_path):
    """
    Read all text files from a given folder and store their contents in one variable.
    """
    combined_text = ""
    for filename in os.listdir(folder_path):
        print("reading.. ",filename)
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                combined_text += file.read() + " "  # Add newline to separate contents of different files
    return combined_text

def clean_text(text, valid_chars_set, replaced_char=None):
    text = replace_unwanted_chars(text, valid_chars_set, replaced_char)
    #gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    gpt2pat = re.compile(r"""
        's|'t|'re|'ve|'m|'ll|'d|                     # English contractions
        \s*[\u0900-\u097F]+(?:[\u093E-\u094D\u0950-\u0954\u0962-\u0963]+)*|  # Devanagari letters and diacritics with leading spaces
        \s*\d+|                                      # Digits with leading spaces
        [^\s\w\u0900-\u097F]+|                       # Punctuation and symbols
        \s+                                          # Whitespace
    """, re.VERBOSE)
    return re.findall(gpt2pat, text)

def replace_unwanted_chars(text, valid_chars_set, replaced_char=None):
    # Use list comprehension to quickly replace unwanted characters
    if replaced_char == None:
        replaced_char = ''
    result = ''.join([char if char in valid_chars_set else replaced_char for char in text])
    return result

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
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
    global n_vocab_init
    merges = {} # (int, int) -> int
    #print("ids",ids)

    for i in range(num_merges):
        stats = {}
        for word_id in ids:
            stats = get_stats(word_id, stats)
        #print("stats",stats)
        pair = max(stats, key=stats.get)
        if len(stats) < 2:
            break
        #print("pair",pair)
        idx = n_vocab_init + i
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

def decode(ids, univ_vocab):
    # given ids , return Python strings
    text = "".join(univ_vocab[idx] for idx in ids)
    return text

def encode(text, merges):
    global  char_to_int
    # given string, return list of ints
    #print(text)
    tokens = [char_to_int[char] for char in text]
    # list(map(ord, text)) #list(text.encode("utf-8"))
    #print(tokens)
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

def encode_ordinary(text, merges, valid_char_set):
    """Encoding that ignores any special tokens."""
    # split text into chunks of text by categories defined in regex pattern
    replace_char = chr(191) # inverted char
    text_chunks = clean_text(text,valid_char_set,replace_char)
    # all chunks of text are encoded separately, then results are joined
    ids = []
    for chunk in text_chunks:
        #chunk_bytes = chunk.encode("utf-8") # raw bytes
        chunk_ids = encode(chunk, merges)
        ids.extend(chunk_ids)
        #print("encode ord ",ids)
    
    return ids

def unicode_chars_range(start, end):
    return [chr(i) for i in range(start, end+1)]

def list_unicode_chars(sp_list):
    return [chr(i) for i in sp_list]

def prepare_init_vocab():
    # functions to list characters in a given Unicode range
    # Devanagari
    special_chars = unicode_chars_range(0x0020, 0x0040)
    punchuation1_chars = unicode_chars_range(0x005B, 0x0060)
    punchuation2_chars = unicode_chars_range(0x007B, 0x007E)

    # Devanagari
    devanagari_chars = unicode_chars_range(0x0900, 0x097F)
    #print(f"devanagari_chars : {devanagari_chars}")

    # Devanagari Extended
    devanagari_extended_chars = unicode_chars_range(0xA8E0, 0xA8FF)
    #print(f"devanagari_extended_chars : {devanagari_extended_chars}")

    # General Punctuations list from wiki page
    # (–,—,―,‗,‛,“,”,„,†,‡,•,…,‰,′,″,‹,›,‼,‾,⁄)
    pun_list = [0x2013,0x2014,0x2015,0x2017,0x2018,0x2019,0x201A,0x201B,0x201C,0x201D,0x201E,0x2020\
                ,0x2021,0x2022,0x2026,0x2030,0x2032,0x2033,0x2039,0x203A,0x203C,0x203E,0x2044,0x204A]
    # append inverted-? and newline
    pun_list.append(0x00BF)
    pun_list.append(10)
    punctuation_chars = list_unicode_chars(pun_list)

    # Superscripts and Subscripts
    #super_subscript_chars = unicode_chars_range(0x2070, 0x209F)

    # Combine all characters
    all_chars_list = (devanagari_chars + devanagari_extended_chars + special_chars + punchuation1_chars + \
                      punchuation2_chars + punctuation_chars)

    # Print all characters with their Unicode code points
    #for char in all_chars:
    #    print(f"Character: {char}, Unicode: {ord(char)}")
    #init_vocab = {ord(ch1): ch1 for ch1 in (all_chars_list)}
    init_vocab = {ii: ch1 for ii, ch1 in enumerate(all_chars_list)}
    char_to_int = {ch1: ii for ii, ch1 in enumerate(all_chars_list)}
    return set(all_chars_list), init_vocab, char_to_int


valid_char_set, univ_vocab, char_to_int = prepare_init_vocab()
n_vocab_init = len(univ_vocab)

corpus = read_text_files("./input_text_files/")
replace_char = None # inverted char
corpus_words = clean_text(corpus, valid_char_set, replace_char)
#print("corpus_lines ", corpus_words)

corpus = ""
for word in corpus_words:
    corpus += word
#print(corpus)


ids = [[char_to_int[char] for char in word] for word in corpus_words]


with open("quant.txt", 'w', encoding='utf-8') as file:
    wfile = "num_merges n_tokens n_orig_corpus_chars n_tokens_corpus compression"
    file.write(wfile)

n_orig_corpus_chars = len(corpus)

def create_tokens(num_merges):
    global n_orig_corpus_chars, corpus_words, univ_vocab, corpus
    global ids, valid_char_set
    #print(" n merges ", num_merges)
    
    univ_vocab_copy = univ_vocab.copy()
    ids_copy = ids.copy()
    # create a BPE tokenizer object
    merges, univ_vocab_copy = train(ids_copy, num_merges, univ_vocab_copy)
    #print("merges",merges)
    #print("vocab1",univ_vocab)
    
    # train BPE tokenizer with Wikipedia corpus
    #print(toks)
    n_tokens = len(univ_vocab_copy)
    #print(f"\n\nBPE tokenization result of text\n\n'{text}'")
    tokens_corpus = encode_ordinary(corpus, merges, valid_char_set)
    n_tokens_corpus = len(tokens_corpus)
    #print(" n_tokens_corpus for voc size ", vocab_size, " -- ", n_tokens_corpus)

    # Save vocab to a JSON file
    print("writing vocab for n = ",num_merges)
    with open(f'vocab_{num_merges}.json', 'w', encoding='utf-8') as vocab_file:
        json.dump(univ_vocab_copy, vocab_file, ensure_ascii=False, indent=4)

    # Save merges to a text file
    print("writing merges for n = ",num_merges)
    merges_with_string_keys = {str(k): v for k, v in merges.items()}
    with open(f'merges_{num_merges}.json', 'w', encoding='utf-8') as merges_file:
        json.dump(merges_with_string_keys, merges_file, ensure_ascii=False, indent=4)
    
    print("writing compression for n = ",num_merges)
    with open(f"quant_{num_merges}.txt", 'w', encoding='utf-8') as file:
        wfile = "num_merges n_tokens n_orig_corpus_chars n_tokens_corpus compression"
        file.write(wfile)
        wfile = f"\n{num_merges} {n_tokens} {n_orig_corpus_chars} {n_tokens_corpus} {n_orig_corpus_chars/n_tokens_corpus}"
        file.write(wfile)
    
    text = "क्रिकेट हा जगभरातला आणि त्यातही भारतात विशेष लोकप्रिय असलेला खेळ आहे. त्यात यंदा क्रिकेट\
            वर्ल्ड कप भारतात होणार असल्याने क्रिकेटरसिकांच्या उत्साहाला उधाण आलं आहे."
    t1_toks = encode_ordinary(text, merges, valid_char_set)
    text = decode(t1_toks, univ_vocab_copy)
    print(len(t1_toks), len(text))
    

# provide the number of merges to be performed
merge_range = [4000, 6000, 8000, 10000, 15000]
for nmer in merge_range:
    create_tokens(nmer)


