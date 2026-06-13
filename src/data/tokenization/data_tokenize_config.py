from dataclasses import dataclass

@dataclass
class TokenizerConfig:
    max_seq_len: int
    dtype: str
    train_ratio: float
    total_tokens: int
    batch_size: int
    write_block: int
    random_seed: int
    
    def __post_init__(self):

        assert 0 < self.train_ratio < 1
        assert self.max_seq_len > 0
        assert self.total_tokens > 0



