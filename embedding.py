from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def meanpool(out, mask):
    tokemb = out.last_hidden_state
    maskexp = mask.unsqueeze(-1).expand(tokemb.size()).float()
    return torch.sum(tokemb * maskexp, 1) / torch.clamp(maskexp.sum(1), min=1e-9)

def getemb(txt):
    enc = tok(txt, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc)
    emb = meanpool(out, enc["attention_mask"])
    return emb[0].numpy()