# tokenizer class
import regex as re
import pickle
from typing import Iterable, Iterator

class BPE_tokenizer:

    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str] | None = None) :
        self.vocab = dict(vocab)  #shallow copy 
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens is not None else []
        self.reverse_vocab = {v:k for k,v in vocab.items()}
        self.merge_dict = {merge: rank for rank, merge in enumerate(self.merges)}
        vocab_len = len(vocab)
        
        if special_tokens:
            for token in special_tokens:
                encoded = token.encode("utf-8")
                if encoded not in self.reverse_vocab:
                    self.vocab[vocab_len] = encoded
                    self.reverse_vocab[encoded] = vocab_len
                    vocab_len += 1

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        
        with open(vocab_filepath,'rb') as f:
            loaded_vocab  = pickle.load(f)
        
        with open(merges_filepath,'rb') as f:
            loaded_merges = pickle.load(f)

        if special_tokens is None:
            return cls(loaded_vocab, loaded_merges, None)
        else:
            return cls(loaded_vocab, loaded_merges,special_tokens)
        

    def encode(self, text: str) -> list[int]: #Encode an input text into a sequence of token IDs
        encoded_tokens = []
        if  not self.special_tokens :
            segments = [text]
        else:
            sorted_tokens =  sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(tok) for tok in sorted_tokens]
            special_pattern = "|".join(escaped_tokens)
            segments = re.split(f"({special_pattern})", text)
            
        segments = [seg for seg in segments if seg]

        reverse_vocab = self.reverse_vocab
        merges_dict = self.merge_dict
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for strs in segments:
            find_special_token = False
            if  self.special_tokens :
               for token in self.special_tokens:
                   if strs == token:
                       encoded = token.encode("utf-8")
                       encoded_tokens += [self.reverse_vocab[encoded]]
                       find_special_token = True
                       break
            
            if not find_special_token :
                matchitr = re.finditer(PAT, strs)   #match.group return a splitted string
                for match in matchitr:
                    bytes_list=[bytes([b]) for b in match.group().encode("utf-8")]
                    merge_checker = True

                    while merge_checker:
                        merge_checker = False
                        index = 0
                        merge_index_list = []
                        while index < len(bytes_list)-1:
                            if (bytes_list[index],bytes_list[index+1]) in merges_dict :
                                merge_checker = True
                                merge_index_list.append((index,merges_dict[(bytes_list[index],bytes_list[index+1])]))
                            
                            index += 1
                        if merge_checker:
                            prior_index= min(merge_index_list, key=lambda x:x[1])[0]
                            bytes_list = bytes_list[:prior_index] + [bytes_list[prior_index]+bytes_list[prior_index+1]] + bytes_list[prior_index+2:]
                    
                    piece_encoded = []
                    for byte in bytes_list :
                        piece_encoded.append(reverse_vocab[byte])
                    
                    encoded_tokens += piece_encoded

        return encoded_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            encoded_chunk = self.encode(chunk)
            yield from encoded_chunk

    def decode(self, ids:list[int]) -> str: #Decode a sequence of token IDs to text
        bytes_sequence = b""
        vocab = self.vocab
        for token_ids in ids:
            byte = vocab[token_ids]
            bytes_sequence += byte
        text = bytes_sequence.decode("utf-8",errors='replace')
        return text