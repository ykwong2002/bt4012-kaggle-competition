# benchmark for the blind challenger --> our best blend (stack 1 : network 3) on the three forward folds, seeds 0-2
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'proxy'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import proxy
from final_screen import compute
from scipy import stats

FOLDS = [('f1', range(1, 16), range(16, 22)), ('f2', range(1, 22), range(22, 28)), ('f3', range(1, 28), range(28, 36))]
# long-horizon fold (7-14 steps ahead, like the test period) --> run with --long
if '--long' in sys.argv:
    FOLDS = [('f4', range(1, 22), range(28, 36))]
df = proxy.load()['df']
tr_all = df[~df.is_test]
rows = []
for seed in (0, 1, 2):
    for name, trs, prs in FOLDS:
        tr = proxy.lab_rows(tr_all[tr_all.time_step.isin(list(trs))])
        pr = proxy.lab_rows(tr_all[tr_all.time_step.isin(list(prs))])
        o = compute(tr, pr, tr_all, seed)
        rk = lambda p: stats.rankdata(p) / len(p)
        rows.append(pd.DataFrame({'txId': pr.txId.values, 'fold': name, 'seed': seed, 'stack': o['stack'],
                                  'net': o['mlp_red'], 'score': (rk(o['stack']) + 3 * rk(o['mlp_red'])) / 4}))
out = pd.concat(rows)
out.to_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_f4.pkl' if '--long' in sys.argv else 'baseline_folds.pkl'))
print('BASELINE_DONE', len(out))
