"""Summarise testlike_scores_long.csv: correlations with public and matched-vs-unmatched paired deltas."""
import os, itertools
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
pub = pd.Series(dict(A=0.9496, B=0.9388, C=0.9512, D=0.9450, E=0.9443, F=0.9500, G=0.9588))
t = pd.read_csv(os.path.join(HERE, 'testlike_scores_long.csv'))
plain = pd.concat([pd.read_csv(os.path.join(HERE, 'seed0', 'proxy_scores_long.csv')), pd.read_csv(os.path.join(HERE, 'proxy_scores_long_seed12.csv'))])
plain = plain[plain.proxy == 'fwd']
tab = t.groupby('config')[['auc', 'ap', 'worst']].agg(['mean', 'std']).round(4)
tab[('plain_fwd', 'auc')] = plain.groupby('recipe').auc.mean().round(4)
tab[('plain_fwd', 'auc_sd')] = plain.groupby('recipe').auc.std().round(4)
print(tab.to_string())
rows = []
for name, src, col in [('test-like fwd', t[~t.config.str.contains('matched')].rename(columns={'config': 'recipe'}), 'auc'),
                       ('test-like fwd', t[~t.config.str.contains('matched')].rename(columns={'config': 'recipe'}), 'ap'),
                       ('plain fwd', plain, 'auc'), ('plain fwd', plain, 'ap')]:
    s = src.groupby('recipe')[col].mean().reindex(pub.index)
    for excl in ['', 'D']:
        ss, pp = s.drop(list(excl)), pub.drop(list(excl))
        pairs = sum(np.sign(ss[a] - ss[b]) == np.sign(pp[a] - pp[b]) for a, b in itertools.combinations(pp.index, 2))
        rows.append(dict(proxy=name, metric=col, without=excl or '-', spearman=stats.spearmanr(ss, pp)[0], pearson=stats.pearsonr(ss, pp)[0],
                         pairs='{}/{}'.format(pairs, len(pp) * (len(pp) - 1) // 2), ranking=''.join(ss.sort_values(ascending=False).index)))
corr = pd.DataFrame(rows)
corr.to_csv(os.path.join(HERE, 'testlike_correlations.csv'), index=False)
print(corr.round(3).to_string())
for base in ['C', 'G']:
    a = t[t.config == base].set_index('seed'); b = t[t.config == base + '-matched'].set_index('seed')
    for m in ['auc', 'ap']:
        d = b[m] - a[m]
        print('{}-matched minus {} {}: per seed {} | mean {:+.4f} sd {:.4f}'.format(base, base, m, d.round(4).tolist(), d.mean(), d.std()))
