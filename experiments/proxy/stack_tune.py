# stage-2 lightgbm settings of the stack on the forward-chaining proxy --> stage-1 scores computed once per fold and seed
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import proxy

_C = {}
def stage2_inputs(tr, pr, all_df, seed, n1=295):
    key = (len(tr), int(tr.time_step.max()), int(pr.time_step.min()), int(pr.time_step.max()), seed)
    if key in _C:
        return _C[key]
    feats = proxy.load()['tree']
    tr_steps, pr_steps = set(tr.time_step.unique()), set(pr.time_step.unique())
    X, y, steps = tr[feats].values, tr[proxy.LABEL].values.astype(int), tr.time_step.values
    unl = all_df[all_df.time_step.isin(tr_steps) & all_df[proxy.LABEL].isna()]
    parts = []
    for a, b in GroupKFold(n_splits=5).split(X, y, groups=steps):
        m = proxy.lgbm(proxy.P_C, n1, seed).fit(X[a], y[a])
        parts.append(pd.Series(m.predict_proba(X[b])[:, 1], index=tr.txId.values[b]))
        held = unl[unl.time_step.isin(set(steps[b]))]
        if len(held):
            parts.append(pd.Series(m.predict_proba(held[feats].values)[:, 1], index=held.txId.values))
    full = proxy.lgbm(proxy.P_C, n1, seed).fit(X, y)
    rest = all_df[all_df.time_step.isin(pr_steps)]
    parts.append(pd.Series(full.predict_proba(rest[feats].values)[:, 1], index=rest.txId.values))
    nbs = proxy.neighbour_scores(pd.concat(parts), pd.concat([tr[['txId']], pr[['txId']]]))
    f2 = feats + proxy.NB_SCORE
    tr2 = tr[['txId', proxy.LABEL] + feats].merge(nbs, on='txId', how='left')
    pr2 = pr[['txId'] + feats].merge(nbs, on='txId', how='left')
    _C[key] = (tr2[f2].values, tr2[proxy.LABEL].values.astype(int), pr2[f2].values)
    return _C[key]

BASE = dict(num_leaves=31, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, min_child_samples=50)
def recipe(n_trees, **change):
    params = dict(BASE, **change)
    def fn(tr, pr, all_df, seed):
        Xt, yt, Xp = stage2_inputs(tr, pr, all_df, seed)
        return lgb.LGBMClassifier(n_estimators=n_trees, random_state=seed, verbose=-1, n_jobs=-1, **params).fit(Xt, yt).predict_proba(Xp)[:, 1]
    return fn

if __name__ == '__main__':
    configs = [('base (31 leaves, lr 0.03, 355 trees)', recipe(355)),
               ('15 leaves', recipe(355, num_leaves=15)), ('63 leaves', recipe(355, num_leaves=63)),
               ('colsample 0.3', recipe(355, colsample_bytree=0.3)), ('colsample 0.7', recipe(355, colsample_bytree=0.7)),
               ('min_child 20', recipe(355, min_child_samples=20)), ('min_child 100', recipe(355, min_child_samples=100)),
               ('lr 0.015, 710 trees', recipe(710, learning_rate=0.015)), ('250 trees', recipe(250)), ('500 trees', recipe(500)),
               ('reg_lambda 5', recipe(355, reg_lambda=5.0)), ('extra_trees', recipe(355, extra_trees=True))]
    for name, fn in configs:
        res, secs = proxy.evaluate(fn, seeds=(0, 1, 2), proxies=['fwd'], verbose=False)
        print('{:<36} auc {:.4f} | ap {:.4f} | worst-step {:.4f} | {:.0f}s'.format(name, res.auc.iloc[0], res.ap.iloc[0], res.worst.iloc[0], secs), flush=True)
