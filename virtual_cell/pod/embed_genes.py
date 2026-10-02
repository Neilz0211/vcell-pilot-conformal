"""Pod-side: pretrained gene embeddings for the perturbed genes (option B).

ESM2 (facebook/esm2_t33_650M_UR50D, MIT): mean over residues of the last hidden layer for the gene's canonical
reviewed UniProt protein (human Swiss-Prot, first entry per gene symbol; sequences longer than 1,022 residues are
truncated). Geneformer (ctheodoris/Geneformer, V2-104M, Apache-2.0): the input token embedding of the gene's Ensembl id.
usage: python embed_genes.py targets.tsv OUT.npz   (targets.tsv: symbol <tab> ensembl id)"""
import pickle
import re
import sys
import urllib.request

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, BertForMaskedLM

rows = [l.split("\t") for l in open(sys.argv[1]).read().splitlines() if l.strip()]
syms = [r[0] for r in rows]; ens = [r[1] if len(r) > 1 else "" for r in rows]
out = {"symbols": np.array(syms)}

# --- ESM2
url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29"
fa = urllib.request.urlopen(url, timeout=600).read().decode()
seq = {}
for rec in fa.split(">")[1:]:
    head, *body = rec.split("\n"); m = re.search(r" GN=(\S+)", head)
    if m and m.group(1) not in seq:
        seq[m.group(1)] = "".join(body)
print("uniprot genes", len(seq), "targets with protein", sum(s in seq for s in syms), flush=True)
tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D", torch_dtype=torch.float16).cuda().eval()
E = np.full((len(syms), model.config.hidden_size), np.nan, np.float32)
order = sorted([i for i, s in enumerate(syms) if s in seq], key=lambda i: len(seq[syms[i]]))
B = 16
with torch.no_grad():
    for k in range(0, len(order), B):
        idx = order[k:k + B]; batch = [seq[syms[i]][:1022] for i in idx]
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        h = model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].clone(); mask[:, 0] = 0
        lens = enc["attention_mask"].sum(1) - 1; mask[torch.arange(len(idx)), lens] = 0   # drop <cls> and <eos>
        e = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        E[idx] = e.cpu().numpy()
        if k % 320 == 0: print("esm2", k, "/", len(order), flush=True)
out["esm2"] = E

# --- Geneformer token embeddings
td = pickle.load(open(hf_hub_download("ctheodoris/Geneformer", "geneformer/token_dictionary_gc104M.pkl"), "rb"))
gf = BertForMaskedLM.from_pretrained("ctheodoris/Geneformer", subfolder="Geneformer-V2-104M")
W = gf.bert.embeddings.word_embeddings.weight.detach().numpy()
G = np.full((len(syms), W.shape[1]), np.nan, np.float32)
hit = 0
for i, e in enumerate(ens):
    if e in td:
        G[i] = W[td[e]]; hit += 1
print("geneformer tokens found", hit, "of", len(syms), flush=True)
out["geneformer"] = G
np.savez_compressed(sys.argv[2], **out)
print("WROTE", sys.argv[2], {k: v.shape for k, v in out.items()}, flush=True)
