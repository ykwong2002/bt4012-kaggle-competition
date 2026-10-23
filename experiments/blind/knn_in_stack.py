# challenger2 density features inside our stack + network blend, same four folds and seeds as the benchmark
#     PROXY_JOBS=4 caffeinate -i python3 .dev/blind/knn_in_stack.py
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'proxy'))
sys.path.insert(0, os.path.join(HERE, '..'))
import proxy
import final_screen
from scipy import stats

kf = pd.read_pickle(os.path.join(HERE, 'challenger2', 'knn_feats.pkl'))
KNN = [c for c in kf.columns if c not in ('txId', 'fold')]
FOLDS = [('f1', range(1, 16), range(16, 22)), ('f2', range(1, 22), range(22, 28)), ('f3', range(1, 28), range(28, 36)), ('f4', range(1, 22), range(28, 36))]
df = proxy.load()['df']
tr_all = df[~df.is_test]
base = list(proxy.load()['tree'])
rows = []
for seed in (0, 1, 2):
    for name, trs, prs in FOLDS:
        feat = kf[kf.fold == name].drop(columns='fold') if 'fold' in kf.columns and (kf.fold == name).any() else kf.drop(columns='fold', errors='ignore')
        t = tr_all.merge(feat, on='txId', how='left')
        tr = proxy.lab_rows(t[t.time_step.isin(list(trs))])
        pr = proxy.lab_rows(t[t.time_step.isin(list(prs))])
        proxy.load()['tree'] = base + KNN
        try:
            final_screen._C.clear()
            o = final_screen.compute(tr, pr, t, seed)
        finally:
            proxy.load()['tree'] = base
        rk = lambda p: stats.rankdata(p) / len(p)
        rows.append(pd.DataFrame({'txId': pr.txId.values, 'fold': name, 'seed': seed, 'stack': o['stack'], 'net': o['mlp_red'],
                                  'score': (rk(o['stack']) + 3 * rk(o['mlp_red'])) / 4}))
        pd.concat(rows).to_pickle(os.path.join(HERE, 'knn_in_stack.pkl'))
print('KNN_STACK_DONE')
