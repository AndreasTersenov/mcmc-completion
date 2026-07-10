"""Gate (iv) dataset generation. Writes to $SCRATCH/ics-zoo/<name>/ (big
outputs -> scratch per CLAUDE.md): data.npz (ZooData arrays), targets.pkl.

Arms (identical compute at train time; P12 varies zoo DIVERSITY only):
  train4: 4 train families x d{2,4,8,16} x 64 targets, 2 contexts each
  train2: {gmm, funnel}    x d{2,4,8,16} x 128 targets, 2 contexts each
  eval:   per train family x d x 12 unseen-theta targets + per held-out
          family x d x 12 targets; 1 fresh context each (K=128)
"""

import argparse
import os
import pickle
import sys
import time
import zlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ics.context import generate_context_for_target
from ics.train import build_zoo_dataset
from ics.zoo import FAMILIES_HELDOUT, FAMILIES_TRAIN, sample_target

DS = (2, 4, 8, 16)
K = 128
N_POOL = 20_000


def np_tree(t):
    return jax.tree_util.tree_map(np.asarray, t)


def gen_train(name, families, per_cell, key):
    specs = [(f, d, i) for f in families for d in DS for i in range(per_cell)]
    t0 = time.time()
    targets, ctxs, data = build_zoo_dataset(key, specs, n_ctx=2, K=K, n_pool=N_POOL)
    out = os.path.join(os.environ["SCRATCH"], "ics-zoo", name)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "data.npz"),
             **{k: np.asarray(v) for k, v in data._asdict().items()})
    with open(os.path.join(out, "targets.pkl"), "wb") as f:
        pickle.dump([(s[0], s[1], np_tree(t)) for s, t in zip(specs, targets)], f)
    print(f"{name}: {len(specs)} targets in {time.time()-t0:.0f}s -> {out}", flush=True)


def gen_eval(key):
    rows = []
    t0 = time.time()
    for fam in FAMILIES_TRAIN + FAMILIES_HELDOUT:
        heldout = fam in FAMILIES_HELDOUT
        for d in DS:
            for i in range(12):
                cell = zlib.crc32(f"{fam}|{d}|{i}".encode())  # NOT hash(): salted
                kt, kc = jr.split(jr.fold_in(key, cell & 0x7FFFFFFF))
                # unseen theta: eval seed stream disjoint from training folds
                t = sample_target(jr.fold_in(kt, 555_000 + i), fam, d)
                ctx = generate_context_for_target(kc, t, K=K)
                rows.append(dict(family=fam, d=d, idx=i, heldout=heldout,
                                 target=np_tree(t), context=np_tree(ctx)))
        print(f"eval {fam} done [{time.time()-t0:.0f}s]", flush=True)
    out = os.path.join(os.environ["SCRATCH"], "ics-zoo", "eval")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eval_set.pkl"), "wb") as f:
        pickle.dump(rows, f)
    print(f"eval: {len(rows)} targets -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["train4", "train2", "eval"])
    args = ap.parse_args()
    if args.which == "train4":
        gen_train("train4", list(FAMILIES_TRAIN), 64, jr.key(777_001))
    elif args.which == "train2":
        gen_train("train2", ["gmm", "funnel"], 128, jr.key(777_002))
    else:
        gen_eval(jr.key(777_003))
    print("GEN-DONE")
