from .adapters import get_tokenizer, run_train_bpe
import pickle
import numpy as np
# vocab ,merges =run_train_bpe("./data/TinyStoriesV2-GPT4-train.txt", vocab_size= 10000, special_tokens=["<|endoftext|>"])
with open("./saved_tokenizer/vocab.pkl",'br') as f:
    vocab = pickle.load(f)
with open("./saved_tokenizer/special_tokens.pkl",'br') as f:
    special_token = pickle.load(f)
with open("./saved_tokenizer/merges.pkl",'br') as f:
    merge = pickle.load(f)
TinyStorie_tokenizer = get_tokenizer(vocab,merge,special_token)

with open("./data/TinyStoriesV2-GPT4-train.txt","r",encoding="utf-8") as f:
    corpus1= f.read()
with open("./data/TinyStoriesV2-GPT4-valid.txt","r",encoding="utf-8") as f:
    corpus2= f.read()

token_ID_train = TinyStorie_tokenizer.encode(corpus1)

token_ID_valid = TinyStorie_tokenizer.encode(corpus2)
token_np_ids_train = np.array(token_ID_train,dtype=np.uint16)
token_np_valid = np.array(token_ID_valid,dtype=np.uint16)

with open("./saved_tokenizer/TS_train_token_unit16.bin","wb") as f:
    token_np_ids_train.tofile(f)
with open("./saved_tokenizer/TS_valid_token_unit16.bin","wb") as f:
    token_np_valid.tofile(f)
