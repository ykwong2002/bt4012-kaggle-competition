# network neighbour scores in the stack, on the forward-chaining proxy --> stack and blend, with and without them
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlp_screen as ms  # the notebook's own network code (TunedMLP, run_epochs, predict_mlp)
import torch
torch.set_num_threads(1)  # lightgbm and torch in one process on macos --> single-thread torch avoids the openmp deadlock
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GroupKFold
from scipy import stats
import proxy

CFG = dict(kind='fgsm', eps=0.05, d=24, sigma=0.2, dropout=0.2)
N_EPOCHS = 12
MLPNB = ['nbm_in_mean', 'nbm_in_max', 'nbm_in_n', 'nbm_out_mean', 'nbm_out_max', 'nbm_out_n']
_C = {}

def mlp_fit_predict(Xa, ya, Xb, seed):
    ms.set_seed(seed)
    swa, _, _ = ms.run_epochs(ms.TunedMLP(Xa.shape[1], CFG), Xa, ya, N_EPOCHS, CFG, None, swa_last=5)
    return ms.predict_mlp(swa, Xb)

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
    # stage 1 lightgbm, as in recipe_G
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
    # network scores --> labelled training rows out-of-fold by time step, prediction rows from the full fit
    f_mlp = feats + proxy.NB_SCORE
    tr_m = tr[['txId'] + feats].merge(nbs, on='txId', how='left').fillna(0)
    pr_m = pr[['txId'] + feats].merge(nbs, on='txId', how='left').fillna(0)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0).fit(np.vstack([tr_m[f_mlp].values, pr_m[f_mlp].values]))
    Qt, Qp = qt.transform(tr_m[f_mlp].values).astype(np.float32), qt.transform(pr_m[f_mlp].values).astype(np.float32)
    mparts = []
    for a, b in folds:
        mparts.append(pd.Series(mlp_fit_predict(Qt[a], y[a], Qt[b], seed), index=tr.txId.values[b]))
    p_mlp_pr = mlp_fit_predict(Qt, y, Qp, seed)
    mparts.append(pd.Series(p_mlp_pr, index=pr.txId.values))
    nbm = proxy.neighbour_scores(pd.concat(mparts), nodes)
    nbm.columns = ['txId'] + MLPNB
    f2 = feats + proxy.NB_SCORE
    tr2 = tr[['txId', proxy.LABEL] + feats].merge(nbs, on='txId', how='left').merge(nbm, on='txId', how='left')
    pr2 = pr[['txId'] + feats].merge(nbs, on='txId', how='left').merge(nbm, on='txId', how='left')
    yt = tr2[proxy.LABEL].values.astype(int)
    g = proxy.lgbm(proxy.P_C, n2, seed).fit(tr2[f2].values, yt).predict_proba(pr2[f2].values)[:, 1]
    g_nb = proxy.lgbm(proxy.P_C, n2, seed).fit(tr2[f2 + MLPNB].values, yt).predict_proba(pr2[f2 + MLPNB].values)[:, 1]
    _C[key] = dict(G=g, G_mlpnb=g_nb, mlp=p_mlp_pr)
    print('  fold {}-{} -> {}-{} seed {} done in {:.0f}s | spearman stack-vs-mlp {:.3f} -> {:.3f}'.format(
        tr.time_step.min(), tr.time_step.max(), pr.time_step.min(), pr.time_step.max(), seed, time.time() - t0,
        stats.spearmanr(g, p_mlp_pr).correlation, stats.spearmanr(g_nb, p_mlp_pr).correlation), flush=True)
    return _C[key]

def recipe(parts):
    def fn(tr, pr, all_df, seed):
        out = compute(tr, pr, all_df, seed)
        return np.c_[tuple(out[p] for p in parts)] if len(parts) > 1 else out[parts[0]]
    return fn

if __name__ == '__main__':
    rows = []
    for name, parts in [('stack (G)', ['G']), ('stack + network nb scores', ['G_mlpnb']), ('network alone', ['mlp']),
                        ('blend G : network 1:1 (current)', ['G', 'mlp']), ('blend G+nb : network 1:1', ['G_mlpnb', 'mlp'])]:
        res, secs = proxy.evaluate(recipe(parts), seeds=(0, 1, 2), proxies=['fwd'], verbose=False)
        rows.append((name, res.auc.iloc[0], res.ap.iloc[0], res.worst.iloc[0]))
        print('{:<34} auc {:.4f} | ap {:.4f} | worst-step {:.4f} | {:.0f}s'.format(name, res.auc.iloc[0], res.ap.iloc[0], res.worst.iloc[0], secs), flush=True)
    cur, new = rows[3], rows[4]
    ok = new[2] - cur[2] >= 0.0015 and new[1] >= cur[1] and new[3] >= cur[3] - 0.005
    print('\nblend with nb scores vs current blend --> ap {:+.4f} | auc {:+.4f} | worst {:+.4f} -> {}'.format(new[2] - cur[2], new[1] - cur[1], new[3] - cur[3], 'PASS' if ok else 'fail'))
