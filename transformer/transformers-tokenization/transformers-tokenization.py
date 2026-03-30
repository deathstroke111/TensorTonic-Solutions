from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.build_vocab([self.pad_token, self.unk_token, self.bos_token, self.eos_token])
        
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        special_tokens = {
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        }

        for text in texts:
            if text in special_tokens:
                words = [text]
            else:
                words = text.split()

            for word in words:
                token = word if word in special_tokens else word.lower()
                if token not in self.word_to_id:
                    self.word_to_id[token] = self.vocab_size
                    self.id_to_word[self.vocab_size] = token
                    self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = text.split()
        return [
            self.word_to_id.get(
                token if token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token] else token.lower(),
                self.word_to_id[self.unk_token],
            )
            for token in tokens
        ]
        
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return " ".join(self.id_to_word.get(token_id, self.unk_token) for token_id in ids)
