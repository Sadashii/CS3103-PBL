"""FinBERT sentiment (ProsusAI/finbert) for every stored news item. Output data/news_fin_<SYM>.parquet: created_at, s (pos - neg)."""
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import config

tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").cuda().half().eval()
print("labels", mdl.config.id2label, flush=True)
pos = [i for i, l in mdl.config.id2label.items() if l.lower().startswith("pos")][0]
neg = [i for i, l in mdl.config.id2label.items() if l.lower().startswith("neg")][0]
for sym in sys.argv[1:] or ["JPM", "BAC", "C", "WFC", "GS", "MS"]:
    src = config.DATA_DIR / "raw" / f"news_{sym}.parquet"
    if not src.exists():
        continue
    n = pd.read_parquet(src)
    txt = (n.headline.fillna("") + ". " + n.summary.fillna("").str.slice(0, 300)).tolist()
    out = []
    with torch.no_grad():
        for i in range(0, len(txt), 64):
            b = tok(txt[i:i + 64], padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
            p = torch.softmax(mdl(**b).logits.float(), -1).cpu().numpy()
            out += list(p[:, pos] - p[:, neg])
    n["s"] = out
    n[["created_at", "s"]].to_parquet(config.DATA_DIR / f"news_fin_{sym}.parquet")
    print(sym, len(n), "scored", flush=True)
