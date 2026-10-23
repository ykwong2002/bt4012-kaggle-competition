# round 10 screen --> direction-aware graph features and multi-step score smoothing on the forward-chaining proxy
import os, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import proxy

HERE = os.path.dirname(os.path.abspath(__file__))
DIR_CACHE = os.path.join(HERE, 'direction.pkl')
DIR = ['in_nb_outdeg_mean', 'in_nb_outdeg_max', 'out_nb_indeg_mean', 'out_nb_indeg_max', 'reciprocal',
       'up2_reach', 'down2_reach', 'chain_count', 'in_nb_source_share', 'out_nb_sink_share']

def build_direction():
    # every feature uses only the edge list, which is complete in both periods
    t0 = time.time()
    e = proxy.load()['edges'][['txId1', 'txId2']].drop_duplicates()
    indeg, outdeg = e.txId2.value_counts(), e.txId1.value_counts()
    nodes = pd.Index(pd.concat([e.txId1, e.txId2]).unique())
    out = pd.DataFrame(index=nodes)
    i_d, o_d = indeg.reindex(nodes).fillna(0), outdeg.reindex(nodes).fillna(0)
    # senders of a node --> how many recipients do they pay
    s = e.assign(v=e.txId1.map(outdeg).fillna(0))
    g = s.groupby('txId2').v.agg(['mean', 'max'])
    out['in_nb_outdeg_mean'], out['in_nb_outdeg_max'] = g['mean'], g['max']
    # receivers of a node --> how many payments do they collect
    r = e.assign(v=e.txId2.map(indeg).fillna(0))
    g = r.groupby('txId1').v.agg(['mean', 'max'])
    out['out_nb_indeg_mean'], out['out_nb_indeg_max'] = g['mean'], g['max']
    # payments in both directions between the same pair
    rev = e.rename(columns={'txId1': 'txId2', 'txId2': 'txId1'})
    both = e.merge(rev, on=['txId1', 'txId2'])
    out['reciprocal'] = both.groupby('txId1').size()
    # distinct nodes two steps upstream and downstream (a -> b -> c)
    two = e.merge(e, left_on='txId2', right_on='txId1', suffixes=('_a', '_b'))[['txId1_a', 'txId2_b']]
    two = two[two.txId1_a != two.txId2_b].drop_duplicates()
    out['up2_reach'] = two.groupby('txId2_b').size()
    out['down2_reach'] = two.groupby('txId1_a').size()
    out['chain_count'] = i_d * o_d
    # share of senders that have no senders themselves, share of receivers that pay no one
    out['in_nb_source_share'] = s.assign(z=(s.txId1.map(indeg).fillna(0) == 0)).groupby('txId2').z.mean()
    out['out_nb_sink_share'] = r.assign(z=(r.txId2.map(outdeg).fillna(0) == 0)).groupby('txId1').z.mean()
    out = out.fillna(0).astype(np.float32)
    out.index.name = 'txId'
    out = out.reset_index()
    out.to_pickle(DIR_CACHE)
    print('direction features built in {:.0f}s for {} nodes'.format(time.time() - t0, len(out)), flush=True)
    return out

def attach():
    fe = proxy.load()
    if 'reciprocal' not in fe['df'].columns:
        d = pd.read_pickle(DIR_CACHE) if os.path.exists(DIR_CACHE) else build_direction()
        df = fe['df'].merge(d, on='txId', how='left')
        df[DIR] = df[DIR].fillna(0).astype(np.float32)
        fe['df'] = df
    return fe

def recipe_G_dir(tr, pr, all_df, seed):
    fe = proxy.load()
    saved = fe['tree']
    fe['tree'] = saved + DIR
    try:
        return proxy.recipe_G(tr, pr, all_df, seed)
    finally:
        fe['tree'] = saved

# cache recipe G predictions per fold and seed so the smoothing grid costs nothing extra
_G = {}
def cached_G(tr, pr, all_df, seed):
    key = (len(tr), int(tr.time_step.max()), int(pr.time_step.min()), int(pr.time_step.max()), seed)
    if key not in _G:
        _G[key] = proxy.recipe_G(tr, pr, all_df, seed)
    return _G[key]

def smooth(pr, p, alpha, rounds):
    # nudge each score towards the mean score of its scored neighbours, only rows being predicted are visible
    s0 = pd.Series(proxy.rank01(p), index=pr.txId.values)
    e = proxy.load()['edges']
    und = pd.concat([e[['txId1', 'txId2']].set_axis(['a', 'b'], axis=1), e[['txId2', 'txId1']].set_axis(['a', 'b'], axis=1)])
    und = und[und.a.isin(s0.index) & und.b.isin(s0.index)]
    s = s0.copy()
    for _ in range(rounds):
        nb = und.assign(v=und.b.map(s)).groupby('a').v.mean()
        has = s.index.isin(nb.index)
        s = s0.copy()
        s[has] = (1 - alpha) * s0[has] + alpha * nb.reindex(s0.index[has]).values
    return s.values

def recipe_smooth(alpha, rounds):
    return lambda tr, pr, all_df, seed: smooth(pr, cached_G(tr, pr, all_df, seed), alpha, rounds)

if __name__ == '__main__':
    fe = attach()
    df = fe['df']
    lab = df[df[proxy.LABEL].notna() & ~df.is_test]
    y_adv = df.is_test.astype(int).values
    print('direction features --> univariate auc on labelled train | train-vs-test auc of the feature alone')
    for c in DIR:
        print('  {:<20} {:.3f} | {:.3f}'.format(c, roc_auc_score(lab[proxy.LABEL].astype(int), lab[c]), roc_auc_score(y_adv, df[c])), flush=True)
    seeds = (0, 1, 2)
    rows = []
    def run(name, fn):
        res, secs = proxy.evaluate(fn, seeds=seeds, proxies=['fwd'], verbose=False)
        res['recipe'] = name
        rows.append(res)
        print('{:<16} auc {:.4f} | ap {:.4f} | worst-step {:.4f} | {:.0f}s'.format(name, res.auc.iloc[0], res.ap.iloc[0], res.worst.iloc[0], secs), flush=True)
    run('G', cached_G)
    run('G_dir', recipe_G_dir)
    for a in [0.1, 0.2, 0.3, 0.5]:
        for k in [1, 2, 3]:
            run('G_smooth_a{}_k{}'.format(a, k), recipe_smooth(a, k))
    out = pd.concat(rows)
    out.to_csv(os.path.join(HERE, 'round10_results.csv'), index=False)
    base = out[out.recipe == 'G'].iloc[0]
    print('\nverdicts (build if ap gain >= 0.001 and auc does not fall)')
    d = out[out.recipe == 'G_dir'].iloc[0]
    print('  direction features --> ap {:+.4f} | auc {:+.4f} -> {}'.format(d.ap - base.ap, d.auc - base.auc, 'BUILD' if d.ap - base.ap >= 0.001 and d.auc >= base.auc else 'drop'))
    sm = out[out.recipe.str.startswith('G_smooth')].sort_values('ap', ascending=False).iloc[0]
    print('  best smoothing {} --> ap {:+.4f} | auc {:+.4f} -> {}'.format(sm.recipe, sm.ap - base.ap, sm.auc - base.auc, 'BUILD' if sm.ap - base.ap >= 0.001 else 'drop'))
