import torch
from dataclasses import dataclass

@dataclass
class PairCollator:
    pad_id: int
    def pad(self, seqs):
        m = max(len(s) for s in seqs)
        return torch.tensor([s + [self.pad_id]*(m-len(s)) for s in seqs])

    def __call__(self, batch):
        ch = self.pad([ex["input_ids_chosen"] for ex in batch])
        rj = self.pad([ex["input_ids_rejected"] for ex in batch])
        def labels(x): y = x.clone(); y[y==self.pad_id] = -100; return y
        return {"chosen_ids":ch,"rejected_ids":rj,
                "chosen_labels":labels(ch),"rejected_labels":labels(rj)}
