#!/usr/bin/env python
"""NA-TSF: STL decomposition -> residual Conv1d encoder -> LSTM forecaster.

Deterministic training (bit-for-bit repeatable) on Electricity / Traffic with
synthetic anomalies (const / missing / gaussian) injected into the training split.

  python na_tsf/train.py --dataset electricity --ano_type missing --ano_ratio 0.3
  python na_tsf/train.py --dataset traffic --tune 1      # grid search -> CSV
"""
import os, csv, itertools, datetime, argparse, random, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

# --------------------------------------------------------------------- #
#  Global determinism switches (must be set *before* any tensor exists) #
# --------------------------------------------------------------------- #
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"          # CUDA ≥ 11.2
torch.use_deterministic_algorithms(True, warn_only=False)  # raise on nondet ops  :contentReference[oaicite:0]{index=0}
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# --------------------------------------------------------------------- #
#  Reproducibility helpers                                              #
# --------------------------------------------------------------------- #
def set_seed_all(seed: int = 0):
    """Reset every RNG stream so results are bit-for-bit repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------------------------------------------------- #
#  Model                                                                #
# --------------------------------------------------------------------- #
class RobustDenoiser(nn.Module):
    """Residual Conv1d encoder: (trend, seasonal, resid) channels -> per-step embedding."""
    def __init__(self, emb_dim=16, drop=0.2):
        super().__init__()
        self.d = nn.Dropout(drop)
        self.c1 = nn.Conv1d(3, 8, 3, padding=1);  self.b1  = nn.BatchNorm1d(8)
        self.c2 = nn.Conv1d(8, emb_dim, 3, padding=1); self.b2  = nn.BatchNorm1d(emb_dim)
        self.r1 = nn.Conv1d(8, emb_dim, 1)

    def forward(self, x):                         # x: [B,T,3]
        h = self.d(F.relu(self.b1(self.c1(x.transpose(1,2)))))
        h = self.d(F.relu(self.b2(self.c2(h)) + self.r1(h)))
        return h.transpose(1,2)                  # -> [B,T,E]

class DecompLSTM(nn.Module):
    def __init__(self, hidden=10, layers=2, emb_dim=16):
        super().__init__()
        self.denoiser = RobustDenoiser(emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(self.denoiser(x))
        return self.head(out[:, -1]).squeeze(-1)

# --------------------------------------------------------------------- #
#  Data helpers                                                         #
# --------------------------------------------------------------------- #
# Data location: <repo>/data by default, override with NA_TSF_DATA=/path/to/dir
DATA_DIR = os.environ.get("NA_TSF_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")

def add_noise(x, rate, scale, kind):
    """Same RNG calls as RobustTSF's dataset.adding_noise, so NA-TSF and the
    baselines are trained on identical anomalies at identical positions."""
    random.seed(0); np.random.seed(0)
    out = x.copy()
    for i in range(len(x)):
        if kind == "const":      sig = scale
        elif kind == "missing":  sig = -x[i] + 0.0001
        elif kind == "gaussian": sig = np.random.normal(0, scale)
        else:                    sig = 0.0
        if random.random() <= rate:
            out[i] = x[i] + sig
    return out

def causal_stl(series, targets, seq_len, context, period):
    """Features for predicting series[t]: robust STL of series[t-context:t]
    (past only), keeping the last seq_len steps -> [N, seq_len, 3]."""
    X = np.empty((len(targets), seq_len, 3), dtype=np.float32)
    for j, t in enumerate(targets):
        r = STL(series[t-context:t], period=period, robust=True).fit()
        X[j] = np.stack([r.trend, r.seasonal, r.resid], 1)[-seq_len:]
    return X

def cached_features(series, targets, args, split):
    """STL per window is the slow part; cache it per (data, noise, STL) setting."""
    key = (f"{args.dataset}_{split}_{args.ano_type}_{args.ano_ratio}_{args.ano_scale}"
           f"_p{args.stl_period}_c{args.stl_context}_L{args.seq_len}.npy")
    path = os.path.join(CACHE_DIR, key)
    if os.path.exists(path):
        return np.load(path)
    X = causal_stl(series, targets, args.seq_len, args.stl_context, args.stl_period)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, X)
    return X

class SeqDS(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def load_splits(args):
    """RobustTSF protocol: 70/30 train/val, scaler fit on train only,
    anomalies injected into the (scaled) training split only."""
    raw = pd.read_csv(os.path.join(DATA_DIR, f"{args.dataset}.csv"), index_col=0)["OT"].values.astype(float)
    tr_end = int(0.7 * len(raw))
    scaler = StandardScaler().fit(raw[:tr_end].reshape(-1, 1))
    train, val = (scaler.transform(a.reshape(-1, 1)).ravel() for a in (raw[:tr_end], raw[tr_end:]))
    return train, add_noise(train, args.ano_ratio, args.ano_scale, args.ano_type), val

def make_loaders(args, seed, verbose=True):
    assert args.stl_context >= max(args.seq_len, 2 * args.stl_period + 1), "stl_context too short"
    train, train_noisy, val = load_splits(args)
    full = np.concatenate([train_noisy, val])   # val windows may look back into (noisy) train history
    L = args.seq_len
    # Same target positions as RobustTSF's windows (minus a short STL warm-up at the start of train).
    t_tr = np.arange(max(L, args.stl_context), len(train) - 1)
    t_va = np.arange(len(train) + L, len(full) - 1)
    Xtr, ytr = cached_features(train_noisy, t_tr, args, "train"), train_noisy[t_tr]   # noisy targets, like the baselines
    Xv,  yv  = cached_features(full, t_va, args, "val"), full[t_va]                   # clean val targets

    g = torch.Generator().manual_seed(seed)
    if verbose:
        print(f"Data loaded » train {Xtr.shape}  val {Xv.shape}  anomalies {(train_noisy != train).sum()}")
    return (DataLoader(SeqDS(Xtr, ytr), batch_size=args.batch, shuffle=True, generator=g, num_workers=0),
            DataLoader(SeqDS(Xv, yv), batch_size=args.batch, shuffle=False))

# --------------------------------------------------------------------- #
#  Training / Validation                                                #
# --------------------------------------------------------------------- #
def evaluate(model, loader, device, cfg):
    model.eval(); mse=mae=tot=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.float().to(device),y.float().to(device)
            p = model(x)
            mse += len(y)*F.mse_loss(p,y).item()
            mae += len(y)*F.l1_loss(p,y).item(); tot += len(y)
    return mse/tot, mae/tot

def train_once(cfg):
    # reset RNGs *inside* every run so tune/solo behave identically
    set_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(cfg, cfg.seed, verbose=cfg.verbose)
    model = DecompLSTM(cfg.hidden, cfg.layers, cfg.embed).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if cfg.verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {total_params:,}")

    best_mae = best_mse = float('inf')
    for ep in range(cfg.epochs):
        model.train(); mae_sum=n=0
        if cfg.verbose:
            print(f"\nEpoch {ep:02d} (lr {opt.param_groups[0]['lr']:.4g})")

        for x,y in train_loader:
            x,y = x.float().to(device), y.float().to(device)
            loss = F.l1_loss(model(x),y)
            opt.zero_grad(); loss.backward(); opt.step()
            mae_sum += len(y)*loss.item(); n += len(y)

        val_mse,val_mae = evaluate(model,val_loader,device,cfg)

        if val_mae < best_mae:
            best_mae, best_mse = val_mae, val_mse

        if cfg.verbose:
            print(f"Train MAE {mae_sum/n:.4f}  Val MAE {val_mae:.4f}  Best MAE {best_mae:.4f}")

    return best_mse,best_mae,val_mse,val_mae

# --------------------------------------------------------------------- #
#  Results logging & hyper-parameter grid search                        #
# --------------------------------------------------------------------- #
FIELDS = ["timestamp", "dataset", "noise", "ratio", "seed", "stl", "context", "hidden", "embed",
          "mse_best", "mae_best", "mse_last", "mae_last"]

def log_result(path, cfg, metrics):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerow([datetime.datetime.now().isoformat(), cfg.dataset, cfg.ano_type, cfg.ano_ratio, cfg.seed,
                    cfg.stl_period, cfg.stl_context, cfg.hidden, cfg.embed, *metrics])

def grid_search(base):
    GRID = {"stl": range(2, 10), "hidden": range(4, 10), "embed": range(2, 17, 2)}
    NOISE = [("none", 0.0), ("const", 0.1), ("const", 0.3), ("missing", 0.1),
             ("missing", 0.3), ("gaussian", 0.1), ("gaussian", 0.3)]
    out = base.out or f"tune_results_{base.dataset}.csv"
    combos = list(itertools.product(GRID["stl"], GRID["hidden"], GRID["embed"], NOISE))
    start = time.time()
    for idx, (stl, hid, emb, (ntype, ratio)) in enumerate(combos, 1):
        cfg = argparse.Namespace(**{**vars(base), "stl_period": stl, "hidden": hid, "embed": emb,
                                    "ano_type": ntype, "ano_ratio": ratio})
        print(f"\n=== Run {idx}/{len(combos)} : stl={stl} hid={hid} emb={emb} {ntype}:{ratio} ===")
        log_result(out, cfg, train_once(cfg))
        print(f"• finished. ETA {(time.time()-start)/idx*(len(combos)-idx)/3600:.1f} Hrs")

# --------------------------------------------------------------------- #
#  Defaults & CLI                                                       #
# --------------------------------------------------------------------- #
DEFAULTS=dict(dataset="electricity", epochs=30, seq_len=16, batch=128, lr=0.01, layers=2,
              hidden=9, embed=12, stl_period=3, stl_context=64, seed=0,
              ano_type="const", ano_ratio=0.1, ano_scale=0.5,
              verbose=1, out="")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", type=int, default=0, help="grid search if 1")
    for k,v in DEFAULTS.items():
        kw = dict(choices=["electricity", "traffic"]) if k == "dataset" else {}
        parser.add_argument(f"--{k}", type=type(v), default=v, **kw)
    args = parser.parse_args()

    if args.verbose:
        print("*** NA-TSF (deterministic) ***")
        print("Config:", vars(args))
        print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
        print("Deterministic:", torch.are_deterministic_algorithms_enabled())

    if args.tune:
        grid_search(args)
    else:
        b_mse,b_mae,l_mse,l_mae = metrics = train_once(args)
        if args.out:
            log_result(args.out, args, metrics)
        print("\n=== FINAL ===")
        print(f"Best MSE {b_mse:.4f}  MAE {b_mae:.4f}")
        print(f"Last MSE {l_mse:.4f}  MAE {l_mae:.4f}")

