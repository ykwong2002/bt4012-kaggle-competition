# lecture 08 screen --> score-aware network features built from the stage-1 out-of-fold scores
import os, time, pickle
import numpy as np, pandas as pd, networkx as nx
import scipy.sparse as sp
from sklearn.model_selection import GroupKFold
import proxy

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCT = os.path.join(HERE, 'sna_struct.pkl')
HOP = ['nb1_q25', 'nb1_q75', 'nb1_high_cnt', 'nb1_high_share', 'nb2_max', 'nb2_mean', 'nb2_q75', 'nb2_high_cnt', 'nb2_scored']
PPR = ['ppr_exposure', 'ppr_own_minus_restart', 'ppr_nb_mean', 'ppr_nb_max']
COM = ['com_size', 'com_mean_other', 'com_max_other', 'com_high_share_other']
GROUPS = {'hop': HOP, 'ppr': PPR, 'com': COM}

def build_struct():
    # structure only, no scores --> undirected adjacency, a time step for every node, louvain communities per step
    t0 = time.time()
    fe = proxy.load()
    e = fe['edges'][['txId1', 'txId2']]
    ids = pd.Index(pd.concat([e.txId1, e.txId2, fe['df'].txId]).unique())
    pos = pd.Series(np.arange(len(ids)), index=ids)
    a, b = pos[e.txId1].values, pos[e.txId2].values
    keep = a != b
    A = sp.coo_matrix((np.ones(keep.sum() * 2), (np.r_[a[keep], b[keep]], np.r_[b[keep], a[keep]])), shape=(len(ids), len(ids))).tocsr()
    A.data[:] = 1.0
    # every edge stays inside one time step, so edge-only nodes take the step of their neighbours
    step = pd.Series(np.nan, index=ids)
    step[fe['df'].txId.values] = fe['df'].time_step.values
    est = pd.DataFrame({'a': e.txId1.values, 'b': e.txId2.values})
    est['t'] = est.a.map(step).fillna(est.b.map(step))
    for col in ['a', 'b']:
        miss = step[est[col].values].isna().values
        step[est[col].values[miss]] = est.t.values[miss]
    coms = pd.Series(-1, index=ids)
    graphs = {}
    cid = 0
    U = nx.Graph()
    U.add_edges_from(zip(e.txId1.values[keep], e.txId2.values[keep]))
    for t, nodes in step.groupby(step).groups.items():
        Gt = U.subgraph(nodes).copy()
        graphs[int(t)] = Gt
        for c in nx.community.louvain_communities(Gt, seed=0):
            coms[list(c)] = cid
            cid += 1
    out = dict(ids=ids, pos=pos, A=A, step=step, coms=coms, graphs=graphs)
    with open(STRUCT, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print('structure built in {:.0f}s | nodes {} | communities {}'.format(time.time() - t0, len(ids), cid), flush=True)
    return out

_S = None
def struct():
    global _S
    if _S is None:
        if os.path.exists(STRUCT):
            with open(STRUCT, 'rb') as f:
                _S = pickle.load(f)
        else:
            _S = build_struct()
    return _S

def hop_features(smap, thr, targets):
    S = struct()
    pos, A = S['pos'], S['A']
    s = np.full(len(pos), np.nan)
    s[pos[smap.index].values] = smap.values
    scored = ~np.isnan(s)
    high = np.where(scored, s > thr, False).astype(float)
    rows = pos[targets].values
    A1 = A[rows]
    out = pd.DataFrame(index=targets)
    # one hop --> spread of neighbour scores and how many neighbours look illicit
    coo = A1.tocoo()
    d = pd.DataFrame({'r': coo.row, 'v': s[coo.col]}).dropna()
    g = d.groupby('r').v
    out['nb1_q25'] = g.quantile(0.25).reindex(range(len(rows))).values
    out['nb1_q75'] = g.quantile(0.75).reindex(range(len(rows))).values
    n1 = A1 @ scored.astype(float)
    out['nb1_high_cnt'] = A1 @ high
    out['nb1_high_share'] = np.where(n1 > 0, out['nb1_high_cnt'].values / np.maximum(n1, 1), np.nan)
    # two hops --> nodes exactly two steps away, self and direct neighbours removed
    A2 = (A1 @ A).tocsr()
    A2.data[:] = 1.0
    A2 = A2 - A2.multiply(A1)
    A2 = A2.tolil()
    for i, r in enumerate(rows):
        A2[i, r] = 0
    A2 = A2.tocsr()
    A2.eliminate_zeros()
    coo = A2.tocoo()
    d = pd.DataFrame({'r': coo.row, 'v': s[coo.col]}).dropna()
    g = d.groupby('r').v
    out['nb2_max'] = g.max().reindex(range(len(rows))).values
    out['nb2_mean'] = g.mean().reindex(range(len(rows))).values
    out['nb2_q75'] = g.quantile(0.75).reindex(range(len(rows))).values
    out['nb2_high_cnt'] = A2 @ high
    out['nb2_scored'] = A2 @ scored.astype(float)
    return out

def ppr_features(smap, targets, alpha=0.85):
    S = struct()
    step, graphs = S['step'], S['graphs']
    expo, own = {}, {}
    tsteps = set(step[targets].dropna().astype(int))
    for t in tsteps:
        Gt = graphs[t]
        p = {v: float(smap[v]) for v in Gt.nodes if v in smap.index}
        tot = sum(p.values())
        if tot <= 0:
            continue
        r = nx.pagerank(Gt, alpha=alpha, personalization=p)
        n = Gt.number_of_nodes()
        for v, val in r.items():
            # scaled by the number of nodes in the step --> comparable across steps
            expo[v] = val * n
            own[v] = (val - (1 - alpha) * p.get(v, 0.0) / tot) * n
    expo, own = pd.Series(expo), pd.Series(own)
    out = pd.DataFrame(index=targets)
    out['ppr_exposure'] = expo.reindex(targets).values
    out['ppr_own_minus_restart'] = own.reindex(targets).values
    A, pos = S['A'], S['pos']
    ev = np.full(len(pos), np.nan)
    ev[pos[expo.index].values] = expo.values
    coo = A[pos[targets].values].tocoo()
    d = pd.DataFrame({'r': coo.row, 'v': ev[coo.col]}).dropna()
    g = d.groupby('r').v
    out['ppr_nb_mean'] = g.mean().reindex(range(len(targets))).values
    out['ppr_nb_max'] = g.max().reindex(range(len(targets))).values
    return out

def com_features(smap, thr, targets):
    S = struct()
    coms = S['coms']
    size = coms.value_counts()
    sc = pd.DataFrame({'c': coms[smap.index].values, 's': smap.values})
    sc['h'] = (sc.s > thr).astype(float)
    agg = sc.groupby('c').agg(n=('s', 'size'), tot=('s', 'sum'), hi=('h', 'sum'))
    # top two scores per community --> max of the other members when a node is itself the top scorer
    top2 = sc.sort_values('s', ascending=False).groupby('c').s.apply(lambda x: list(x.values[:2]))
    c_t = coms[targets].values
    s_t = smap.reindex(targets).values
    has = ~np.isnan(s_t)
    a = agg.reindex(c_t)
    n_other = a.n.values - has
    out = pd.DataFrame(index=targets)
    out['com_size'] = size.reindex(c_t).values
    out['com_mean_other'] = np.where(n_other > 0, (a.tot.values - np.nan_to_num(s_t)) / np.maximum(n_other, 1), np.nan)
    hi_self = np.where(has, s_t > thr, False)
    out['com_high_share_other'] = np.where(n_other > 0, (a.hi.values - hi_self) / np.maximum(n_other, 1), np.nan)
    t2 = top2.reindex(c_t)
    mx = []
    for lst, v in zip(t2.values, s_t):
        if not isinstance(lst, list) or len(lst) == 0:
            mx.append(np.nan)
        elif not np.isnan(v) and v == lst[0]:
            mx.append(lst[1] if len(lst) > 1 else np.nan)
        else:
            mx.append(lst[0])
    out['com_max_other'] = mx
    return out

def recipe_G_sna(groups):
    def fn(tr, pr, all_df, seed, n1=295, n2=355):
        # same as proxy.recipe_G up to the score map
        feats = proxy.load()['tree']
        tr_steps, pr_steps = set(tr.time_step.unique()), set(pr.time_step.unique())
        X, y, steps = tr[feats].values, tr[proxy.LABEL].values.astype(int), tr.time_step.values
        vis = all_df['visible'] if 'visible' in all_df else pd.Series(True, index=all_df.index)
        unl = all_df[all_df.time_step.isin(tr_steps) & all_df[proxy.LABEL].isna() & vis]
        parts = []
        for a, b in GroupKFold(n_splits=5).split(X, y, groups=steps):
            m = proxy.lgbm(proxy.P_C, n1, seed).fit(X[a], y[a])
            parts.append(pd.Series(m.predict_proba(X[b])[:, 1], index=tr.txId.values[b]))
            held = unl[unl.time_step.isin(set(steps[b]))]
            if len(held):
                parts.append(pd.Series(m.predict_proba(held[feats].values)[:, 1], index=held.txId.values))
        full = proxy.lgbm(proxy.P_C, n1, seed).fit(X, y)
        rest = all_df[all_df.time_step.isin(pr_steps) & vis]
        parts.append(pd.Series(full.predict_proba(rest[feats].values)[:, 1], index=rest.txId.values))
        smap = pd.concat(parts)
        smap = smap[~smap.index.duplicated()]
        nodes = pd.concat([tr[['txId']], pr[['txId']]])
        nbs = proxy.neighbour_scores(smap, nodes)
        thr = float(smap.reindex(tr.txId.values).quantile(0.9))
        targets = nodes.txId.values
        extra, cols = [], []
        if 'hop' in groups:
            extra.append(hop_features(smap, thr, targets)); cols += HOP
        if 'ppr' in groups:
            extra.append(ppr_features(smap, targets)); cols += PPR
        if 'com' in groups:
            extra.append(com_features(smap, thr, targets)); cols += COM
        ex = pd.concat(extra, axis=1)
        ex.index.name = 'txId'
        ex = ex[~ex.index.duplicated()].reset_index()
        tr2 = tr[['txId', proxy.LABEL] + feats].merge(nbs, on='txId', how='left').merge(ex, on='txId', how='left')
        pr2 = pr[['txId'] + feats].merge(nbs, on='txId', how='left').merge(ex, on='txId', how='left')
        return proxy.fit_lgb(proxy.P_C, n2, feats + proxy.NB_SCORE + cols, tr2, pr2, seed)
    return fn

if __name__ == '__main__':
    import sys
    struct()
    which = sys.argv[1:] or ['G', 'hop', 'ppr', 'com']
    rows = []
    for name in which:
        fn = 'G' if name == 'G' else recipe_G_sna(name.split('+'))
        res, secs = proxy.evaluate(fn, seeds=(0, 1, 2), proxies=['fwd'], verbose=False)
        res['recipe'] = name
        rows.append(res)
        print('{:<12} auc {:.4f} | ap {:.4f} | worst-step {:.4f} | {:.0f}s'.format(name, res.auc.iloc[0], res.ap.iloc[0], res.worst.iloc[0], secs), flush=True)
    out = pd.concat(rows)
    out.to_csv(os.path.join(HERE, 'sna_results_{}.csv'.format('_'.join(w.replace('+', 'p') for w in which))), index=False)
