
from transformers import PreTrainedTokenizer
import torch
import json
import os
import re

class ProteinTokenizerHF(PreTrainedTokenizer):

    vocab_files_names = {"vocab_file": "vocab.json"}
    
    def __init__(self, vocab_file=None, **kwargs):

        self.vocab = {
            tok: i for i, tok in enumerate([
                "<pad>", "<eos>",
                "<PROTEIN_A>", "</PROTEIN_A>",
                "<PROTEIN_B>", "</PROTEIN_B>",
                "A","C","D","E","F","G","H","I","K","L",
                "M","N","P","Q","R","S","T","V","W","Y",
                "X","B","Z","U","O"
            ])
        }
        

        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)

        self.ids_to_tokens = {i: tok for tok, i in self.vocab.items()}

        super().__init__(
            pad_token="<pad>",
            eos_token="<eos>",
            unk_token="X",  
            **kwargs
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):

        return re.findall(r"<[^>]+>|[^ \s]", text)

    def _convert_token_to_id(self, token):

        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self):
        return self.vocab


    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
            
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is not None:
            raise ValueError("This tokenizer does not support token pairs")
        return token_ids_0 + [self.eos_token_id]
    
    def get_valid_aa_token_ids(self):
        valid_aas = list("ACDEFGHIKLMNPQRSTVWY")
        return torch.tensor(
            [self.vocab[aa] for aa in valid_aas if aa in self.vocab],
            dtype=torch.long
        )
        




