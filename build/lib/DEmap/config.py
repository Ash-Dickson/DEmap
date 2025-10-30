from dataclasses import dataclass


# CFG params for running DEmap, contains deafults.
@dataclass
class Config:
    run_line: str
    init_n: int = 16
    probe_n: int = 10000
    max_iters: int = 10
    random_seed: int = 123
