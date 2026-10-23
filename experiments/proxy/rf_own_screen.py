# shallow random forests and an own-features tree model, alone and blended with the stack, on the forward-chaining proxy
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import proxy
from stack_tune import stage2_inputs, BASE

_P = {}
def preds(tr, pr, all_df, seed):
    key = (len(tr), int(tr.time_step.max()), int(pr.time_step.min()), int(pr.time_step.max()), seed)
    if key in _P:
        return _P[key]
    Xt, yt, Xp = stage2_inputs(tr, pr, all_df, seed)
    out = {'G': lgb.LGBMClassifier(n_estimators=355, random_state=seed, verbose=-1, n_jobs=-1, **BASE).fit(Xt, yt).predict_proba(Xp)[:, 1]}
    rf = dict(n_estimators=500, max_depth=12, min_samples_leaf=5, max_features=0.33, n_jobs=-1, random_state=seed)
    out['rf_stack'] = RandomForestClassifier(**rf).fit(Xt, yt).predict_proba(Xp)[:, 1]
    # own features only --> the provided transaction features, no graph summaries, no neighbour scores
    kept = proxy.load()['kept']
    yo = tr[proxy.LABEL].values.astype(int)
    out['lgb_own'] = lgb.LGBMClassifier(n_estimators=295, random_state=seed, verbose=-1, n_jobs=-1, **BASE).fit(tr[kept].values, yo).predict_proba(pr[kept].values)[:, 1]
    out['rf_own'] = RandomForestClassifier(**rf).fit(tr[kept].values, yo).predict_proba(pr[kept].values)[:, 1]
    _P[key] = out
    return out

def recipe(parts, weights=None):
    def fn(tr, pr, all_df, seed):
        out = preds(tr, pr, all_df, seed)
        if len(parts) == 1:
            return out[parts[0]]
        w = weights or [1] * len(parts)
        return np.average([stats.rankdata(out[p]) / len(out[p]) for p in parts], axis=0, weights=w)
    return fn

if __name__ == '__main__':
    rows = []
    for name, parts, w in [('G (stack, used)', ['G'], None), ('rf as stack final model', ['rf_stack'], None),
                           ('G 3 : rf stack 1', ['G', 'rf_stack'], [3, 1]), ('G 1 : rf stack 1', ['G', 'rf_stack'], [1, 1]),
                           ('lgb own features alone', ['lgb_own'], None), ('rf own features alone', ['rf_own'], None),
                           ('G 3 : lgb own 1', ['G', 'lgb_own'], [3, 1]), ('G 1 : lgb own 1', ['G', 'lgb_own'], [1, 1]),
                           ('G 3 : rf own 1', ['G', 'rf_own'], [3, 1]), ('G 1 : rf own 1', ['G', 'rf_own'], [1, 1])]:
        res, secs = proxy.evaluate(recipe(parts, w), seeds=(0, 1, 2), proxies=['fwd'], verbose=False)
        rows.append((name, res.auc.iloc[0], res.ap.iloc[0], res.worst.iloc[0]))
        print('{:<26} auc {:.4f} | ap {:.4f} | worst-step {:.4f} | {:.0f}s'.format(name, res.auc.iloc[0], res.ap.iloc[0], res.worst.iloc[0], secs), flush=True)
    base = rows[0]
    print('\npass rule: ap >= +0.0015, auc not lower, worst-step not more than 0.005 lower')
    for name, auc, ap, worst in rows[1:]:
        ok = ap - base[2] >= 0.0015 and auc >= base[1] and worst >= base[3] - 0.005
        print('  {:<26} ap {:+.4f} auc {:+.4f} worst {:+.4f} -> {}'.format(name, ap - base[2], auc - base[1], worst - base[3], 'PASS' if ok else 'fail'))
