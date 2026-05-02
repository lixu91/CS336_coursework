# tokenizer class
import regex as re
import pickle
from typing import Iterable, Iterator

class BPE_tokenizer:

    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str] | None = None) :
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_vocab = {v:k for k,v in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        
        with open(vocab_filepath,'rb') as f:
            loaded_vocab  = pickle.load(f)
        
        with open(merges_filepath,'rb') as f:
            loaded_merges = pickle.load(f)

        with open("./saved_tokenizer/special_tokens.pkl",'rb') as f:
            loaded_tokens = pickle.load(f)

        return cls(loaded_vocab, loaded_merges, loaded_tokens)
        

    def encode(self, text: str) -> list[int]: #Encode an input text into a sequence of token IDs
        encoded_tokens = []
        if self.special_tokens is None:
            segments = [text]
        else:
            escaped_tokens = [re.escape(tok) for tok in self.special_tokens]
            special_pattern = "|".join(escaped_tokens)
            segments = re.split(f"({special_pattern})", text)

        reverse_vocab = self.reverse_vocab
        merges = self.merges    
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for strs in segments:
            matchitr = re.finditer(PAT, strs)   #match.group return a splitted string
            for match in matchitr:
                bytes_list=[bytes([b]) for b in match.group().encode("utf-8")]
                merge_checker = True

                while merge_checker:
                    merge_checker = False
                    bytes_list_new = []
                    index = 0
                    while index < len(bytes_list)-1:
                        for merge in merges:
                            if merge[0] == bytes_list[index] and merge[1] == bytes_list[index+1] :
                                merge_checker = True
                                bytes_list_new.append(bytes_list[index]+bytes_list[index+1])
                                index += 2
                            else:
                                bytes_list_new.append(bytes_list[index])
                                index += 1
                    bytes_list = bytes_list_new
                
                piece_encoded = []
                for byte in bytes_list :
                    piece_encoded.append(reverse_vocab[byte])
                
                encoded_tokens += piece_encoded

        return encoded_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        encoded_list = []
        for chunk in iterable:
            encoded_list += self.encode(chunk)
        
        return encoded_list    ##
    

    def decode(self, ids:list[int]) -> str: #Decode a sequence of token IDs to text
        bytes_sequence = b""
        vocab = self.vocab
        for token_ids in ids:
            byte = vocab[token_ids]
            bytes_sequence += byte
        text = bytes_sequence.decode("utf-8",errors='replace')
        return text