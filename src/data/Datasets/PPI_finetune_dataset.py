
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

class PPIFinetuneDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, block_size,conditional_masking=False):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.conditional_masking = conditional_masking

        self.protein_b_token_id = tokenizer.convert_tokens_to_ids("<PROTEIN_B>")
        self.special_ids = {
            tokenizer.convert_tokens_to_ids("<PROTEIN_A>"),
            tokenizer.convert_tokens_to_ids("</PROTEIN_A>"),
            tokenizer.convert_tokens_to_ids("<PROTEIN_B>"),
            tokenizer.convert_tokens_to_ids("</PROTEIN_B>"),
            tokenizer.pad_token_id
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        text = (
            "<PROTEIN_A>" + example["sequence1"] + "</PROTEIN_A>"
            "<PROTEIN_B>" + example["sequence2"] + "</PROTEIN_B>" 
        )


        enc = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.block_size,
            add_special_tokens=True, 
        )
        
        input_ids = enc["input_ids"]
        

        if self.conditional_masking:

            targets = [-100] * len(input_ids)
            

            try:
                b_pos = input_ids.index(self.protein_b_token_id)
            except ValueError:
                b_pos = len(input_ids)
                

            for i in range(b_pos + 1, len(input_ids)):
                token_id = input_ids[i]
                

                is_closing_tag = (token_id == self.tokenizer.convert_tokens_to_ids("</PROTEIN_B>"))
                is_eos = (token_id == self.tokenizer.eos_token_id)
                
                if (token_id not in self.special_ids) or is_closing_tag or is_eos:
                    targets[i] = token_id

        else:
            # Full Training (Recommended)
            targets = list(input_ids)
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(targets, dtype=torch.long)
        }




class PreTokenizedPPIDataset(Dataset):
    """
    Loads a pre-tokenized PPI dataset saved to disk by scripts/preprocess_finetune.py.

    """
    def __init__(self, dataset_path: str):
        self.dataset = load_from_disk(dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels":    torch.tensor(item["labels"],    dtype=torch.long),
        }


class FinetuneCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):

        max_len = max(len(x["input_ids"]) for x in batch)
        
        input_ids_batch = []
        labels_batch = []
        
        for item in batch:
            ids = item["input_ids"]
            lbls = item["labels"]
            
            pad_len = max_len - len(ids)
            

            padded_ids = torch.cat([ids, torch.tensor([self.pad_token_id] * pad_len, dtype=torch.long)])
            

            padded_lbls = torch.cat([lbls, torch.tensor([-100] * pad_len, dtype=torch.long)])
            
            input_ids_batch.append(padded_ids)
            labels_batch.append(padded_lbls)
            
        return torch.stack(input_ids_batch), torch.stack(labels_batch)