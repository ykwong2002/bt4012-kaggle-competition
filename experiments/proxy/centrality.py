# screen centrality features on the forward-chaining proxy before building anything in the notebook
import os, sys, time, pickle
import numpy as np, pandas as pd, networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import proxy

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'centrality.pkl')
CENT = ['pagerank', 'rev_pagerank', 'clustering', 'core_number', 'log_component_size', 'in_out_balance']

def build_centrality():
    # whole edge list, edge-only nodes included --> structure is complete in both periods
    t0 = time.time()
    e = proxy.load()['edges']
    G = nx.from_pandas_edgelist(e, 'txId1', 'txId2', create_using=nx.DiGraph)
    n = G.number_of_nodes()
    pr = nx.pagerank(G, alpha=0.85)
    rpr = nx.pagerank(G.reverse(copy=False), alpha=0.85)
    U = G.to_undirected()
    U.remove_edges_from(nx.selfloop_edges(U))
    clus = nx.clustering(U)
    core = nx.core_number(U)
    comp = {}
    for c in nx.connected_components(U):
        for v in c:
            comp[v] = len(c)
    indeg, outdeg = dict(G.in_degree()), dict(G.out_degree())
    nodes = list(G.nodes())
    out = pd.DataFrame({'txId': nodes})
    # pagerank times the node count --> equals pagerank within the node's own cluster times the cluster size, comparable across time steps
    out['pagerank'] = [pr[v] * n for v in nodes]
    out['rev_pagerank'] = [rpr[v] * n for v in nodes]
    out['clustering'] = [clus[v] for v in nodes]
    out['core_number'] = [core[v] for v in nodes]
    out['log_component_size'] = [np.log1p(comp[v]) for v in nodes]
    out['in_out_balance'] = [np.log1p(indeg[v]) - np.log1p(outdeg[v]) for v in nodes]
    out.to_pickle(CACHE)
    print('centrality built in {:.0f}s for {} nodes'.format(time.time() - t0, n))
    return out

def attach():
    fe = proxy.load()
    if 'pagerank' in fe['df'].columns:
        return fe
    cent = pd.read_pickle(CACHE) if os.path.exists(CACHE) else build_centrality()
    df = fe['df'].merge(cent, on='txId', how='left')
    df[CENT] = df[CENT].fillna(0).astype(np.float32)
    fe['df'] = df
    return fe

def recipe_C_cent(tr, pr, all_df, seed):
    return proxy.fit_lgb(proxy.P_C, 295, proxy.load()['tree'] + CENT, tr, pr, seed)

def recipe_G_cent(tr, pr, all_df, seed):
    # recipe G with the centrality features added in both stages
    fe = proxy.load()
    saved = fe['tree']
    fe['tree'] = saved + CENT
    try:
        return proxy.recipe_G(tr, pr, all_df, seed)
    finally:
        fe['tree'] = saved

if __name__ == '__main__':
    fe = attach()
    lab = fe['df'][fe['df'][proxy.LABEL].notna() & ~fe['df'].is_test]
    print('univariate auc of each centrality feature on labelled training rows:')
    for c in CENT:
        print('  {:<20} {:.3f}'.format(c, roc_auc_score(lab[proxy.LABEL].astype(int), lab[c])))
    rows = []
    for name, fn in [('C', 'C'), ('C_cent', recipe_C_cent), ('G', 'G'), ('G_cent', recipe_G_cent)]:
        res, secs = proxy.evaluate(fn, seeds=(0, 1, 2), proxies=['fwd'], verbose=False)
        res['recipe'] = name
        rows.append(res)
        print(res.to_string(index=False), '| {:.0f}s'.format(secs), flush=True)
    pd.concat(rows).to_csv(os.path.join(HERE, 'centrality_results.csv'), index=False)
