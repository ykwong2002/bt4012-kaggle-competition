# final round (22 oct) --> segment-specific blend weights and forward-only neighbour scores, gated on forward-chaining folds
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlp_screen as ms
import torch
torch.set_num_threads(1)
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GroupKFold
from sklearn import metrics
from scipy import stats
import proxy
from mlpnb_screen import mlp_fit_predict
from final_screen import reduced_feats

def n_out_of(df):
    # feat_5 sits on a regular grid --> decoded number of outputs (see 8.19b)
    u = np.unique(np.round(proxy.load()['df'].feat_5.values.astype(float), 6))
    step = np.min(np.diff(u))
    return np.round((df.feat_5.values - u.min()) / step) + 1

def stage1(tr, pr, all_df, seed, mode, n1=295, min_step=8):
    feats = proxy.load()['tree']
    tr_steps, pr_steps = set(tr.time_step.unique()), set(pr.time_step.unique())
    X, y, steps = tr[feats].values, tr[proxy.LABEL].values.astype(int), tr.time_step.values
    unl = all_df[all_df.time_step.isin(tr_steps) & all_df[proxy.LABEL].isna()]
    parts = []
    if mode == 'kfold':
        for a, b in GroupKFold(n_splits=5).split(X, y, groups=steps):
            m = proxy.lgbm(proxy.P_C, n1, seed).fit(X[a], y[a])
            parts.append(pd.Series(m.predict_proba(X[b])[:, 1], index=tr.txId.values[b]))
            held = unl[unl.time_step.isin(set(steps[b]))]
            if len(held):
                parts.append(pd.Series(m.predict_proba(held[feats].values)[:, 1], index=held.txId.values))
    else:
        # forward only --> steps before min_step use grouped folds among themselves, every later step is scored by a model trained on earlier steps only
        early = steps < min_step
        Xe, ye, se, ide = X[early], y[early], steps[early], tr.txId.values[early]
        for a, b in GroupKFold(n_splits=min(5, len(set(se)))).split(Xe, ye, groups=se):
            m = proxy.lgbm(proxy.P_C, n1, seed).fit(Xe[a], ye[a])
            parts.append(pd.Series(m.predict_proba(Xe[b])[:, 1], index=ide[b]))
            held = unl[unl.time_step.isin(set(se[b]))]
            if len(held):
                parts.append(pd.Series(m.predict_proba(held[feats].values)[:, 1], index=held.txId.values))
        for t in sorted(s for s in tr_steps if s >= min_step):
            past = steps < t
            m = proxy.lgbm(proxy.P_C, n1, seed).fit(X[past], y[past])
            now = steps == t
            parts.append(pd.Series(m.predict_proba(X[now])[:, 1], index=tr.txId.values[now]))
            held = unl[unl.time_step == t]
            if len(held):
                parts.append(pd.Series(m.predict_proba(held[feats].values)[:, 1], index=held.txId.values))
    full = proxy.lgbm(proxy.P_C, n1, seed).fit(X, y)
    rest = all_df[all_df.time_step.isin(pr_steps)]
    parts.append(pd.Series(full.predict_proba(rest[feats].values)[:, 1], index=rest.txId.values))
    return pd.concat(parts)

_C = {}
def compute(tr, pr, all_df, seed, n2=355):
    key = (len(tr), int(tr.time_step.max()), int(pr.time_step.min()), int(pr.time_step.max()), seed)
    if key in _C:
        return _C[key]
    t0 = time.time()
    feats = proxy.load()['tree']
    y = tr[proxy.LABEL].values.astype(int)
    nodes = pd.concat([tr[['txId']], pr[['txId']]])
    red = reduced_feats(tr, feats) + proxy.NB_SCORE
    out = {'y': pr[proxy.LABEL].values.astype(int), 'n_out': n_out_of(pr), 'step': pr.time_step.values}
    for mode in ['kfold', 'forward']:
        nbs = proxy.neighbour_scores(stage1(tr, pr, all_df, seed, mode), nodes)
        f2 = feats + proxy.NB_SCORE
        tr2 = tr[['txId', proxy.LABEL] + feats].merge(nbs, on='txId', how='left')
        pr2 = pr[['txId'] + feats].merge(nbs, on='txId', how='left')
        out['stack_' + mode] = proxy.lgbm(proxy.P_C, n2, seed).fit(tr2[f2].values, y).predict_proba(pr2[f2].values)[:, 1]
        A, B = tr2[red].fillna(0).values, pr2[red].fillna(0).values
        qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0).fit(np.vstack([A, B]))
        out['mlp_' + mode] = mlp_fit_predict(qt.transform(A).astype(np.float32), y, qt.transform(B).astype(np.float32), seed)
    _C[key] = out
    print('  fold {}-{} -> {}-{} seed {} | {:.0f}s'.format(tr.time_step.min(), tr.time_step.max(), pr.time_step.min(), pr.time_step.max(), seed, time.time() - t0), flush=True)
    return out

def rk(p):
    return stats.rankdata(p) / len(p)

def blend(o, mode, w_one, w_other):
    # w --> network share; one-output rows and the rest can get different shares, ranks are global so scores stay comparable
    rs, rm = rk(o['stack_' + mode]), rk(o['mlp_' + mode])
    w = np.where(o['n_out'] == 1, w_one, w_other)
    return (1 - w) * rs + w * rm

def fold_auc(folds, fn):
    return np.mean([metrics.roc_auc_score(o['y'], fn(o)) for o in folds])

if __name__ == '__main__':
    seeds = (0, 1, 2)
    # populate the cache for every forward-chaining fold and seed through the proxy's own split logic
    proxy.evaluate(lambda tr, pr, a, s: compute(tr, pr, a, s)['stack_kfold'], seeds=seeds, proxies=['fwd'], verbose=False)
    by_fold = {}
    for k, o in _C.items():
        by_fold.setdefault((k[1], k[2]), []).append(o)
    names = {(15, 16): '1-15 -> 16-21', (21, 22): '1-21 -> 22-27', (27, 28): '1-27 -> 28-35'}
    print('\nA. forward-only neighbour scores (per fold auc, 3 seeds)')
    for f, folds in sorted(by_fold.items()):
        print('  {:<14} stack kfold {:.4f} forward {:.4f} | blend 1:1 kfold {:.4f} forward {:.4f}'.format(
            names.get(f, f), fold_auc(folds, lambda o: o['stack_kfold']), fold_auc(folds, lambda o: o['stack_forward']),
            fold_auc(folds, lambda o: blend(o, 'kfold', .5, .5)), fold_auc(folds, lambda o: blend(o, 'forward', .5, .5))))
    print('\nB. segment-specific network share, chosen on 1-21 -> 22-27 (kfold scores)')
    grid = [0.25, 0.4, 0.5, 0.6, 0.75]
    tune = by_fold[(21, 22)]
    best = max(((a, b) for a in grid for b in grid), key=lambda ab: fold_auc(tune, lambda o: blend(o, 'kfold', *ab)))
    best_single = max(grid, key=lambda w: fold_auc(tune, lambda o: blend(o, 'kfold', w, w)))
    print('  chosen shares --> one-output {} other {} | best single share {}'.format(best[0], best[1], best_single))
    for f in [(15, 16), (27, 28)]:
        folds = by_fold[f]
        print('  {:<14} 1:1 {:.4f} | best single {:.4f} | segment {:.4f} | per seed segment-minus-1:1 {}'.format(
            names[f], fold_auc(folds, lambda o: blend(o, 'kfold', .5, .5)), fold_auc(folds, lambda o: blend(o, 'kfold', best_single, best_single)),
            fold_auc(folds, lambda o: blend(o, 'kfold', *best)),
            ' '.join('{:+.4f}'.format(metrics.roc_auc_score(o['y'], blend(o, 'kfold', *best)) - metrics.roc_auc_score(o['y'], blend(o, 'kfold', .5, .5))) for o in folds)))
    print('\nper-seed forward-minus-kfold blend auc on all folds:', ' '.join('{:+.4f}'.format(
        metrics.roc_auc_score(o['y'], blend(o, 'forward', .5, .5)) - metrics.roc_auc_score(o['y'], blend(o, 'kfold', .5, .5))) for folds in by_fold.values() for o in folds))
