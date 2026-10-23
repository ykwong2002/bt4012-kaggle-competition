# scores a challenger on the forward folds
# input --> a pickle/parquet/csv with columns txId, fold, seed, score for labelled rows of the prediction steps
#   f1 --> train on steps 1-15, predict 16-21 | f2 --> train 1-21, predict 22-27 | f3 --> train 1-27, predict 28-35
#   f4 (long horizon, 7-14 steps ahead like the test period) --> train 1-21, predict 28-35
# metrics per seed --> pooled auc and average precision over f1-f3 after rank-normalising each fold, worst single-step auc over f1-f3,
#   per-fold auc, and f4 auc / ap reported separately
#     python3 .dev/blind/evaluate.py my_preds.pkl
import sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

lab = pd.read_csv('data/train.csv', usecols=['txId', 'time_step', 'label']).dropna()

def score(path):
    p = path if isinstance(path, pd.DataFrame) else (pd.read_pickle(path) if path.endswith('.pkl') else (pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)))
    rows = []
    for seed, g in p.groupby('seed'):
        g = g.merge(lab, on='txId')
        row = dict(seed=seed)
        m = g[g.fold.isin(['f1', 'f2', 'f3'])].copy()
        if len(m):
            m['r'] = m.groupby('fold').score.transform(lambda s: stats.rankdata(s) / len(s))
            per = [roc_auc_score(h.label, h.score) for _, h in m.groupby('time_step') if 0 < h.label.sum() < len(h)]
            row.update(auc=roc_auc_score(m.label, m.r), ap=average_precision_score(m.label, m.r), worst=min(per))
        for f, h in g.groupby('fold'):
            row['auc_' + f] = roc_auc_score(h.label, h.score)
            if f == 'f4':
                row['ap_f4'] = average_precision_score(h.label, h.score)
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == '__main__':
    r = score(sys.argv[1])
    print(r.round(4).to_string(index=False))
    print('mean -->', ' | '.join('{} {:.4f}'.format(c, r[c].mean()) for c in r.columns if c != 'seed'))
