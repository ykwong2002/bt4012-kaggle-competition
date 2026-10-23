# xgboost and catboost as the stage-2 model of the stack, alone and rank-blended with lightgbm, on the forward-chaining proxy
import numpy as np, pandas as pd
import xgboost as xgb, catboost as cb
from sklearn.model_selection import GroupKFold
from scipy import stats
import proxy

_CACHE = {}
def stage2_all(tr, pr, all_df, seed, n1=295, n2=355):
    key = (len(tr), int(tr.time_step.max()), int(pr.time_step.min()), int(pr.time_step.max()), seed)
    if key in _CACHE:
        return _CACHE[key]
    # same as proxy.recipe_G up to the stage-2 inputs
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
    Xt, yt, Xp = tr2[f2].values, tr2[proxy.LABEL].values.astype(int), pr2[f2].values
    out = {'lgb': proxy.lgbm(proxy.P_C, n2, seed).fit(Xt, yt).predict_proba(Xp)[:, 1]}
    out['xgb'] = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.5,
                                   min_child_weight=5, random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, yt).predict_proba(Xp)[:, 1]
    out['cat'] = cb.CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, random_seed=seed, verbose=0,
                                       thread_count=-1).fit(Xt, yt).predict_proba(Xp)[:, 1]
    _CACHE[key] = out
    return out

def recipe(names):
    def fn(tr, pr, all_df, seed):
        out = stage2_all(tr, pr, all_df, seed)
        return np.c_[tuple(out[n] for n in names)] if len(names) > 1 else out[names[0]]
    return fn

if __name__ == '__main__':
    for name, names in [('lgb (= G)', ['lgb']), ('xgb', ['xgb']), ('cat', ['cat']), ('lgb + xgb', ['lgb', 'xgb']),
                        ('lgb + cat', ['lgb', 'cat']), ('lgb + xgb + cat', ['lgb', 'xgb', 'cat'])]:
        res, secs = proxy.evaluate(recipe(names), seeds=(0, 1, 2), proxies=['fwd'], verbose=False)
        print('{:<16} auc {:.4f} | ap {:.4f} | worst-step {:.4f} | {:.0f}s'.format(name, res.auc.iloc[0], res.ap.iloc[0], res.worst.iloc[0], secs), flush=True)
