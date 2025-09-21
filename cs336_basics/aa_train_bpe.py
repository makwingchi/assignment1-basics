import os
import pickle
import regex as re
import multiprocessing
from copy import deepcopy
from typing import BinaryIO, List, Tuple, Iterable
from collections import defaultdict


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe_naive(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
):
    GPT2_REGEX_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    idx2bytes = {i: bytes([i]) for i in range(256)}

    idx = 256

    # ------ 1. deal with special tokens
    special_token_patterns = []
    for special_token in special_tokens:
        encoded_special_token = special_token.encode("utf-8")
        idx2bytes[idx] = encoded_special_token
        idx += 1

        special_token_patterns.append(re.escape(special_token))
    
    if len(idx2bytes) > vocab_size:
        raise ValueError(f"desired vocabulary size of {vocab_size} is smaller than initial vocabulary size of {len(idx2bytes)}")
    
    # ------ 2. read text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # ------ 3. initialize byte_tokens_cnt and byte_token_to_word mappings
    splits = re.split("|".join(special_token_patterns), text)
    byte_tokens_cnt = defaultdict(int) # {(b'l', b'o', b'w'): 3, (b'l', b'o', b'b'): 4, ...}

    for split in splits:
        # doc = split.strip()

        str_tokens = re.findall(GPT2_REGEX_PAT, split)
        for s in str_tokens:
            encoded_s = list(s.encode("utf-8")) # [111,222,123,...]
            byte_s = tuple([bytes([e]) for e in encoded_s])
            byte_tokens_cnt[byte_s] += 1

    # ------ 4. perform merges
    merges = []
    while len(idx2bytes) < vocab_size:
        # ------ 4.1 perform pair counts
        pair_cnt = defaultdict(int) # {(b'l', b'o'): 7, (b'o', b'w'): 3, ...}
        for k, v in byte_tokens_cnt.items():
            word_len = len(k)
            for i in range(word_len-1):
                curr_pair = (k[i], k[i+1])
                pair_cnt[curr_pair] += v
        
        if len(pair_cnt) == 0:
            break
        
        # print(pair_cnt)
        # ------ 4.2 get candidate with max pair count
        max_cnt = max(pair_cnt.values())
        candidates = [k for k, v in pair_cnt.items() if v==max_cnt]
        l, r = max(candidates)
        curr_merge = l + r
        merges.append((l, r))

        # ------ 4.3 update byte_tokens_cnt
        new_byte_tokens_cnt = {}
        for k, v in byte_tokens_cnt.items():
            word_len = len(k)
            i = 0
            curr_byte_tokens = []

            while i < word_len:
                curr_l = k[i]
                if i < word_len - 1:
                    curr_r = k[i+1]
                else:
                    curr_r = None

                if curr_l == l and curr_r == r:
                    curr_byte_tokens.append(l+r)
                    i += 2
                else:
                    curr_byte_tokens.append(curr_l)
                    i += 1
            
            new_byte_tokens_cnt[tuple(curr_byte_tokens)] = v
        
        byte_tokens_cnt = new_byte_tokens_cnt
        # print(byte_tokens_cnt)
        
        # ------ 4.4 update idx2bytes
        idx2bytes[idx] = curr_merge
        idx += 1
    
    return idx2bytes, merges


def pretokenize(input_path, start, end, special_token_patterns):
    GPT2_REGEX_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    splits = re.split("|".join(special_token_patterns), chunk)
    byte_tokens_cnt = defaultdict(int) # {(b'l', b'o', b'w'): 3, (b'l', b'o', b'b'): 4, ...}

    for split in splits:
        str_tokens = re.findall(GPT2_REGEX_PAT, split)

        for s in str_tokens:
            encoded_s = list(s.encode("utf-8")) # [111,222,123,...]
            byte_s = tuple([bytes([e]) for e in encoded_s])
            byte_tokens_cnt[byte_s] += 1
    
    return byte_tokens_cnt


def train_bpe_improved(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
):
    idx2bytes = {i: bytes([i]) for i in range(256)}

    idx = 256

    # ------ 1. deal with special tokens
    special_token_patterns = []
    for special_token in special_tokens:
        encoded_special_token = special_token.encode("utf-8")
        idx2bytes[idx] = encoded_special_token
        idx += 1

        special_token_patterns.append(re.escape(special_token))
    
    if len(idx2bytes) > vocab_size:
        raise ValueError(f"desired vocabulary size of {vocab_size} is smaller than initial vocabulary size of {len(idx2bytes)}")
    
    # ------ 2. read text
    # with open(input_path, "r", encoding="utf-8") as f:
    #     text = f.read()

    args = []

    # copy from pretokenization_example.py
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            args.append((input_path, start, end, special_token_patterns))

    with multiprocessing.Pool() as pool:
        results = pool.starmap(pretokenize, args)
    
    # ------ 3. initialize byte_tokens_cnt
    byte_tokens_cnt = defaultdict(int)
    for _map in results:
        for k, v in _map.items():
            byte_tokens_cnt[k] += v

    # ------ 4. prepare pair_cnt
    pair_cnt = defaultdict(int) # {(b'l', b'o'): 7, (b'o', b'w'): 3, ...}
    pair2keys = defaultdict(set) # {(b'o', b'w'): { (b'l', b'o', b'w', b'e', b'r'), (b'p', b'o', b'w', b'e', b'r'), ... }
    for k, v in byte_tokens_cnt.items():
        word_len = len(k)
        for i in range(word_len-1):
            curr_pair = (k[i], k[i+1])
            pair_cnt[curr_pair] += v
            pair2keys[curr_pair].add(k)
    
    # ------ 5. perform merges (modify pair_cnts only)
    merges = []
    while len(idx2bytes) < vocab_size:        
        # ------ 5.1 get candidate with max pair count
        max_cnt = max(pair_cnt.values())

        if max_cnt == 0:
            break

        candidates = [k for k, v in pair_cnt.items() if v==max_cnt]
        l, r = max(candidates)
        curr_merge = l + r
        merges.append((l, r))

        # ------ 5.2 update keys
        keys = deepcopy(pair2keys[(l, r)])
        for prev_key in keys:
            key_len = len(prev_key)
            key_cnt = byte_tokens_cnt.pop(prev_key)

            new_key = []
            i = 0

            while i < key_len:
                curr_l = prev_key[i]
                if i < key_len - 1:
                    curr_r = prev_key[i+1]
                else:
                    curr_r = None
                
                if curr_l == l and curr_r == r:
                    new_key.append(l+r)
                    i += 2
                else:
                    new_key.append(curr_l)
                    i += 1
            
            new_key = tuple(new_key)
            byte_tokens_cnt[new_key] = key_cnt

            # ------ 5.3 update counts and mapping
            for left, right in zip(prev_key[:-1], prev_key[1:]):
                pair_cnt[(left, right)] -= key_cnt
                curr_set = pair2keys[(left, right)]

                if prev_key in curr_set:
                    curr_set.remove(prev_key)
            
            for left, right in zip(new_key[:-1], new_key[1:]):
                pair_cnt[(left, right)] += key_cnt
                pair2keys[(left, right)].add(new_key)
            
        # ------ 5.4 update idx2bytes
        idx2bytes[idx] = curr_merge
        idx += 1
    
    return idx2bytes, merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges

        if special_tokens is None:
            special_tokens = []
        
        self.special_tokens = set(special_tokens)

        self.token2idx = {}

        for k, v in self.vocab.items():
            self.token2idx[v] = k
        
        self.__add_special_tokens()
        # print(self.vocab)
        # print(self.merges)
        # print(self.special_tokens)
        # print(self.token2idx)
    

    def __add_special_tokens(self):
        for token in self.special_tokens:
            encoded_token = token.encode("utf-8")

            if encoded_token not in self.token2idx:
                self.vocab[len(self.vocab)] = encoded_token
                self.token2idx[encoded_token] = len(self.token2idx)


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, filepath, special_tokens=None):
        assert vocab_filepath is None or len(vocab_filepath) == 0, f"Your input vocab_filepath is {vocab_filepath}. You should leave vocab_filepath empty and provide filepath instead"
        assert merges_filepath is None or len(merges_filepath) == 0, f"Your input merges_filepath is {merges_filepath}. You should leave merges_filepath empty and provide filepath instead"

        try:
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
        except Exception as exc:
            raise
        
        vocab = data["vocab"]
        merges = data["merges"]

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str):
        if len(self.special_tokens) > 0:
            escaped = [re.escape(token) for token in self.special_tokens]
            pattern = f"({'|'.join(escaped)})"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        tokens = []
        for part in parts:
            if not parts:
                continue

            if part in self.special_tokens:
                encoded_part = self.token2idx[part.encode("utf-8")]
                tokens.append(encoded_part)
            else:
                encoded_part = self.encode_helper(part)
                tokens.extend(encoded_part)
        
        return tokens        


    def encode_helper(self, text):
        GPT2_REGEX_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        str_tokens = re.findall(GPT2_REGEX_PAT, text)
        # print(str_tokens)
        unique_str_tokens = {}

        for token in str_tokens:
            unique_str_tokens[token] = []

        for token, _ in unique_str_tokens.items():
            encoded = list(token.encode("utf-8")) # [111,222,123,...]
            encoded_byte = [bytes([e]) for e in encoded]

            pairs = []
            pairs_set = set()
            for i in range(len(encoded_byte)-1):
                pairs.append(encoded_byte[i]+encoded_byte[i+1])
                pairs_set.add(encoded_byte[i]+encoded_byte[i+1])
            
            idx = 0
            while len(pairs) >= 1 and idx < len(self.merges):
                while idx < len(self.merges):
                    curr_merge = self.merges[idx][0] + self.merges[idx][1]
                    if curr_merge in pairs_set:
                        break
                
                    idx += 1
                
                if idx >= len(self.merges):
                    break
            
                curr_pair = self.merges[idx][0] + self.merges[idx][1]
                new_encoded_byte = []

                i = 0
                while i < len(encoded_byte):
                    left = encoded_byte[i]
                    right = encoded_byte[i+1] if i < len(encoded_byte) - 1 else None

                    if right is not None and left + right == curr_pair:
                        new_encoded_byte.append(left+right)
                        i += 2
                    else:
                        new_encoded_byte.append(left)
                        i += 1
                
                new_pairs = []
                for i in range(len(new_encoded_byte)-1):
                    new_pairs.append(new_encoded_byte[i]+new_encoded_byte[i+1])
                
                pairs = new_pairs
                pairs_set = set(pairs)
                encoded_byte = new_encoded_byte
                
                # print(f"idx={idx}")
                # print(token)
                # print(f"pairs={pairs}")
                # print(f"encoded_byte={encoded_byte}")
            
            for item in encoded_byte:
                unique_str_tokens[token].append(self.token2idx[item])
        
        res = []
        for token in str_tokens:
            res.extend(unique_str_tokens[token])
            
        return res


    def encode_iterable(self, iterable: Iterable[str]):
        for string in iterable:
            encoded = self.encode(string)
            for _id in encoded:
                yield _id


    def decode(self, ids: list[int]):
        decoded = []

        for _id in ids:
            if _id in self.vocab:
                decoded.append(self.vocab[_id])
            else:
                decoded.append("\uFFFD".encode("utf-8"))

        if len(decoded) == 0:
            return ""
        
        res = decoded[0]
        for i in range(1, len(decoded)):
            res += decoded[i]

        return res.decode("utf-8", errors="replace")