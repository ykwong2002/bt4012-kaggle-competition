# final screen (21 oct) --> network weight, full-graph network, two network views; one cached run gates all three
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlp_screen as ms
import torch
torch.set_num_threads(1)
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GroupKFold
from scipy import stats
import proxy
from mlpnb_screen import CFG, N_EPOCHS, mlp_fit_predict

_C = {}
def reduced_feats(df, feats, thr=0.95):
    # same greedy rule as drop_near_duplicates in the notebook
    c = df[feats].corr().abs()
    up = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
    drop = {col for col in up.columns if (up[col] > thr).any()}
    return [f for f in feats if f not in drop]

def compute(tr, pr, all_df, seed, n1=295, n2=355):
    key = (len(tr), int(tr.time_step.max()), int(pr.time_step.min()), int(pr.time_step.max()), seed)
    if key in _C:
        return _C[key]
    t0 = time.time()
    feats = proxy.load()['tree']
    tr_steps, pr_steps = set(tr.time_step.unique()), set(pr.time_step.unique())
    X, y, steps = tr[feats].values, tr[proxy.LABEL].values.astype(int), tr.time_step.values
    unl = all_df[all_df.time_step.isin(tr_steps) & all_df[proxy.LABEL].isna()]
    folds = list(GroupKFold(n_splits=5).split(X, y, groups=steps))
    parts = []
    for a, b in folds:
        m = proxy.lgbm(proxy.P_C, n1, seed).fit(X[a], y[a])
        parts.append(pd.Series(m.predict_proba(X[b])[:, 1], index=tr.txId.values[b]))
        held = unl[unl.time_step.isin(set(steps[b]))]
        if len(held):
            parts.append(pd.Series(m.predict_proba(held[feats].values)[:, 1], index=held.txId.values))
    full = proxy.lgbm(proxy.P_C, n1, seed).fit(X, y)
    rest = all_df[all_df.time_step.isin(pr_steps)]
    parts.append(pd.Series(full.predict_proba(rest[feats].values)[:, 1], index=rest.txId.values))
    nodes = pd.concat([tr[['txId']], pr[['txId']]])
    nbs = proxy.neighbour_scores(pd.concat(parts), nodes)
    f2 = feats + proxy.NB_SCORE
    tr2 = tr[['txId', proxy.LABEL] + feats].merge(nbs, on='txId', how='left')
    pr2 = pr[['txId'] + feats].merge(nbs, on='txId', how='left')
    stack = proxy.lgbm(proxy.P_C, n2, seed).fit(tr2[f2].values, y).predict_proba(pr2[f2].values)[:, 1]
    out = {'stack': stack}
    red = reduced_feats(tr, feats) + proxy.NB_SCORE
    for name, cols in [('mlp_full', f2), ('mlp_red', red)]:
        A, B = tr2[cols].fillna(0).values, pr2[cols].fillna(0).values
        qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0).fit(np.vstack([A, B]))
        out[name] = mlp_fit_predict(qt.transform(A).astype(np.float32), y, qt.transform(B).astype(np.float32), seed)
    _C[key] = out
    print('  fold {}-{} -> {}-{} seed {} | {:.0f}s | inputs full {} reduced {} | spearman full-vs-red {:.3f}'.format(
        tr.time_step.min(), tr.time_step.max(), pr.time_step.min(), pr.time_step.max(), seed, time.time() - t0,
        len(f2), len(red), stats.spearmanr(out['mlp_full'], out['mlp_red']).correlation), flush=True)
    return out

def recipe(parts, weights):
    def fn(tr, pr, all_df, seed):
        out = compute(tr, pr, all_df, seed)
        if len(parts) == 1:
            return out[parts[0]]
        return np.average([stats.rankdata(out[p]) / len(out[p]) for p in parts], axis=0, weights=weights)
    return fn

CANDIDATES = [
    ('stack : reduced 1:1 (current style)', ['stack', 'mlp_red'], [1, 1]),
    ('stack alone', ['stack'], [1]),
    ('reduced network alone', ['mlp_red'], [1]),
    ('full network alone', ['mlp_full'], [1]),
    ('stack : reduced 1:2', ['stack', 'mlp_red'], [1, 2]),
    ('stack : reduced 1:3', ['stack', 'mlp_red'], [1, 3]),
    ('stack : full 1:1', ['stack', 'mlp_full'], [1, 1]),
    ('stack : full 1:2', ['stack', 'mlp_full'], [1, 2]),
    ('stack : reduced : full 1:1:1', ['stack', 'mlp_red', 'mlp_full'], [1, 1, 1]),
]

if __name__ == '__main__':
    seeds = (0, 1, 2)
    per = {}
    for name, parts, w in CANDIDATES:
        rows = []
        for s in seeds:
            res, _ = proxy.evaluate(recipe(parts, w), seeds=(s,), proxies=['fwd'], verbose=False)
            rows.append((res.auc.iloc[0], res.ap.iloc[0], res.worst.iloc[0]))
        per[name] = np.array(rows)
        m = per[name].mean(0)
        print('{:<38} auc {:.4f} | ap {:.4f} | worst-step {:.4f} | per-seed auc {}'.format(name, m[0], m[1], m[2], ' '.join('{:.4f}'.format(r[0]) for r in rows)), flush=True)
    base = per[CANDIDATES[0][0]]
    bm = base.mean(0)
    print('\npass rule vs current style: ap >= +0.0015, auc not lower, worst-step not more than 0.005 lower, every seed auc higher')
    for name, _, _ in CANDIDATES[1:]:
        m = per[name].mean(0)
        seeds_up = bool((per[name][:, 0] > base[:, 0]).all())
        ok = m[1] - bm[1] >= 0.0015 and m[0] >= bm[0] and m[2] >= bm[2] - 0.005 and seeds_up
        print('  {:<38} ap {:+.4f} auc {:+.4f} worst {:+.4f} seeds-up {} -> {}'.format(name, m[1] - bm[1], m[0] - bm[0], m[2] - bm[2], seeds_up, 'PASS' if ok else 'fail'))
