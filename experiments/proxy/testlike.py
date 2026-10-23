"""Test-like forward-chaining proxy.

In the test period, test.csv holds only the (hidden-label) rows; the other neighbours appear only in the edge list
and have no features. To mimic that, a 'labelled-only' graph table (view L) is built in which unlabelled training rows are
featureless: they still count in in/out/total degree, but are excluded from neighbour mean/max and from *_nb_known.
Edges never cross time steps, so a row's graph features depend only on its own step and one table serves every fold.

  plain fwd       : train rows full view, validation rows full view
  test-like fwd   : train rows full view, validation rows view L      (unmatched training)
  test-like fwd   : train rows view L,    validation rows view L      (matched training, 'C-matched')
For G, featureless rows (unlabelled rows in view L) receive no stage-1 score.

    PROXY_JOBS=4 caffeinate -i python3 .dev/proxy/testlike.py
"""
import os, sys, time, pickle
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proxy
from proxy import LABEL

HERE = proxy.HERE
CACHE_L = os.path.join(HERE, 'features_labelled_only.pkl')
FWD = [(range(1, 16), range(16, 22)), (range(1, 22), range(22, 28)), (range(1, 28), range(28, 36))]


def graph_table(df, edges, nb, visible):
    g = pd.DataFrame({'txId': df.txId.values})
    g['in_degree'] = g.txId.map(edges.txId2.value_counts()).fillna(0)
    g['out_degree'] = g.txId.map(edges.txId1.value_counts()).fillna(0)
    g['total_degree'] = g.in_degree + g.out_degree
    for c in ['in_degree', 'out_degree', 'total_degree']:
        g['log_' + c] = np.log1p(g[c])
    feats = df.loc[visible, ['txId'] + nb]
    ind = edges.merge(feats, left_on='txId1', right_on='txId').drop(columns=['txId', 'txId1'])
    ia = ind.groupby('txId2')[nb].agg(['mean', 'max'])
    ia.columns = ['in_{}_{}'.format(s, f) for f, s in ia.columns]
    ia['in_nb_known'] = ind.groupby('txId2').size()
    outd = edges.merge(feats, left_on='txId2', right_on='txId').drop(columns=['txId', 'txId2'])
    oa = outd.groupby('txId1')[nb].agg(['mean', 'max'])
    oa.columns = ['out_{}_{}'.format(s, f) for f, s in oa.columns]
    oa['out_nb_known'] = outd.groupby('txId1').size()
    g = g.merge(ia, left_on='txId', right_index=True, how='left').merge(oa, left_on='txId', right_index=True, how='left').fillna(0)
    return g


def load_views():
    fe = proxy.load()
    df = fe['df']
    if os.path.exists(CACHE_L):
        gL = pd.read_pickle(CACHE_L)
    else:
        t0 = time.time()
        vis = (df[LABEL].notna() | df.is_test).values
        gL = graph_table(df, fe['edges'], fe['nb'], vis)
        gL[fe['graph']] = gL[fe['graph']].astype(np.float32)
        # sanity: all-visible rebuild must equal the cached full view
        gF = graph_table(df, fe['edges'], fe['nb'], np.ones(len(df), bool))
        assert np.allclose(gF[fe['graph']].values, df[fe['graph']].values, atol=1e-4)
        gL.to_pickle(CACHE_L)
        print('labelled-only graph view built in {:.0f}s'.format(time.time() - t0))
    full = df[~df.is_test].copy()
    full['visible'] = True
    L = full.drop(columns=fe['graph']).merge(gL[['txId'] + fe['graph']], on='txId', how='left')
    L['visible'] = L[LABEL].notna().values
    return full, L


def evaluate_fwd(recipe, seeds, train_view, val_view):
    t0 = time.time()
    fn = proxy.RECIPES[recipe]
    rows = []
    for seed in seeds:
        parts = []
        for trs, prs in FWD:
            tr = proxy.lab_rows(train_view[train_view.time_step.isin(list(trs))])
            pr = proxy.lab_rows(val_view[val_view.time_step.isin(list(prs))])
            pool = pd.concat([train_view[train_view.time_step.isin(list(trs))], val_view[val_view.time_step.isin(list(prs))]])
            P = proxy.combine(np.asarray(fn(tr, pr, pool, seed)))
            parts.append((pr[LABEL].values.astype(int), P, pr.time_step.values, pr.adv_w.values))
        rows.append(dict(seed=seed, **proxy.metrics_for(parts)))
    return pd.DataFrame(rows), time.time() - t0


if __name__ == '__main__':
    full, L = load_views()
    for name, v in [('full view', full), ('labelled-only view', L)]:
        lab = v[v[LABEL].notna() & (v.time_step >= 16)]
        print('{}: share of fwd validation rows (steps 16-35) with no visible neighbour = {:.3f}'.format(
            name, ((lab.in_nb_known + lab.out_nb_known) == 0).mean()))
    seeds = (0, 1, 2)
    configs = [('C', 'testlike', full, L), ('C-matched', 'testlike', L, L)]
    configs += [(r, 'testlike', full, L) for r in 'ABDEF']
    configs += [('G', 'testlike', full, L), ('G-matched', 'testlike', L, L)]
    out = os.path.join(HERE, 'testlike_scores_long.csv')
    allrows = []
    for label, proxy_name, tv, vv in configs:
        res, sec = evaluate_fwd(label.split('-')[0], seeds, tv, vv)
        res.insert(0, 'config', label)
        res['seconds'] = sec
        allrows.append(res)
        pd.concat(allrows).to_csv(out, index=False)
        print('{:10s} auc {:.4f} +- {:.4f} | ap {:.4f} | {:.0f}s'.format(label, res.auc.mean(), res.auc.std(), res.ap.mean(), sec), flush=True)
