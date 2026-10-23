"""Average proxy scores over all seed runs and correlate with the public leaderboard."""
import os, glob, itertools
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC = dict(A=0.9496, B=0.9388, C=0.9512, D=0.9450, E=0.9443, F=0.9500, G=0.9588)
files = {os.path.join(HERE, 'seed0', 'proxy_scores_long.csv'): 1, os.path.join(HERE, 'proxy_scores_long_seed12.csv'): 2}
parts = [pd.read_csv(f).assign(w=w) for f, w in files.items() if os.path.exists(f)]
long = pd.concat(parts)
metrics = ['auc', 'ap', 'worst', 'mstep', 'wauc']
agg = long.groupby(['recipe', 'proxy']).apply(lambda d: pd.Series({m: np.average(d[m], weights=d.w) for m in metrics}))
pub = pd.Series(PUBLIC)
rows = []
for proxy in agg.index.get_level_values('proxy').unique():
    for m in metrics:
        s = agg.xs(proxy, level='proxy')[m].reindex(pub.index)
        pairs = [np.sign(s[a] - s[b]) == np.sign(pub[a] - pub[b]) for a, b in itertools.combinations(pub.index, 2)]
        rows.append(dict(proxy=proxy, metric=m, spearman=stats.spearmanr(s, pub)[0], pearson=stats.pearsonr(s, pub)[0],
                         pairs_ok='{}/21'.format(sum(pairs)), first=s.idxmax(), last=s.idxmin(), ranking=''.join(s.sort_values(ascending=False).index)))
corr = pd.DataFrame(rows).sort_values(['spearman', 'pearson'], ascending=False).reset_index(drop=True)
corr.to_csv(os.path.join(HERE, 'correlations_all_seeds.csv'), index=False)
table = agg.unstack('proxy')
table.columns = ['{}_{}'.format(p, m) for m, p in table.columns]
table = table[sorted(table.columns)]
table['public'] = pub
table.to_csv(os.path.join(HERE, 'proxy_scores_all_seeds.csv'))
pd.set_option('display.width', 250)
print('public ranking:', ''.join(pub.sort_values(ascending=False).index), '| seeds weight', long.w.sum() / len(long) if False else '')
print(table[[c for c in table.columns if c.endswith('_auc') or c.endswith('_ap') or c == 'public']].round(4).to_string())
print(corr.round(3).to_string())
