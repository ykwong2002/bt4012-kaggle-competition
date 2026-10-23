# research round (22 oct) --> label-free deep graph infomax embeddings (gin encoder) as extra stacking inputs
# matched graph --> unlabelled training rows and edge-only nodes are featureless, same as the test period
# per fold the encoder sees features of steps up to the end of the prediction block, never any label
#     PROXY_JOBS=4 caffeinate -i python3 .dev/proxy/dgi_screen.py
import os, sys, time
import numpy as np, pandas as pd
import torch
torch.set_num_threads(2)
from torch import nn
from torch_geometric.nn import GINConv, DeepGraphInfomax
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proxy
from proxy import LABEL

HERE = proxy.HERE
N_COMP = 16
EMB = ['dgi_{}'.format(i) for i in range(N_COMP)]


class Encoder(nn.Module):
    def __init__(self, n_in, hid=64):
        super().__init__()
        mlp = lambda a, b: nn.Sequential(nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Linear(b, b))
        self.c1, self.c2 = GINConv(mlp(n_in, hid)), GINConv(mlp(hid, hid))
        self.act = nn.PReLU(hid)

    def forward(self, x, ei):
        return self.act(self.c2(torch.relu(self.c1(x, ei)), ei))


def node_table():
    fe = proxy.load()
    df, edges = fe['df'], fe['edges']
    vis = (df[LABEL].notna() | df.is_test).values
    step_of = dict(zip(df.txId.values, df.time_step.values))
    # edge-only nodes take the step of a neighbour (edges never cross steps)
    ids = pd.unique(np.r_[df.txId.values, edges.txId1.values, edges.txId2.values])
    extra = [i for i in ids if i not in step_of]
    e2 = pd.concat([edges[['txId1', 'txId2']], edges[['txId2', 'txId1']].set_axis(['txId1', 'txId2'], axis=1)])
    fill = e2[e2.txId1.isin(set(extra))].assign(s=lambda d: d.txId2.map(step_of)).dropna().groupby('txId1').s.first()
    step_of.update(fill.astype(int).to_dict())
    nodes = pd.DataFrame({'txId': ids})
    nodes['step'] = nodes.txId.map(step_of)
    nodes = nodes.dropna().astype({'step': int}).reset_index(drop=True)
    X = np.zeros((len(nodes), len(fe['kept']) + 1), np.float32)
    pos = pd.Series(np.arange(len(nodes)), index=nodes.txId.values)
    rows = pos[df.txId.values[vis]].values
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0)
    X[rows, :-1] = qt.fit_transform(df.loc[vis, fe['kept']].values)
    X[rows, -1] = 1.0  # has-features flag
    return nodes, X, pos, edges


def train_dgi(nodes, X, pos, edges, last_step, seed=0, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    graphs = []
    for s in range(1, last_step + 1):
        idx = np.where(nodes.step.values == s)[0]
        loc = pd.Series(np.arange(len(idx)), index=nodes.txId.values[idx])
        e = edges[edges.txId1.isin(loc.index) & edges.txId2.isin(loc.index)]
        a, b = loc[e.txId1.values].values, loc[e.txId2.values].values
        ei = torch.tensor(np.r_[np.c_[a, b], np.c_[b, a]].T, dtype=torch.long)
        graphs.append((idx, torch.tensor(X[idx]), ei))
    enc = Encoder(X.shape[1])
    model = DeepGraphInfomax(64, encoder=enc, summary=lambda z, *a, **k: torch.sigmoid(z.mean(0)),
                             corruption=lambda x, ei: (x[torch.randperm(x.size(0))], ei))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train(); tot = 0
        for g in np.random.permutation(len(graphs)):
            _, x, ei = graphs[g]
            opt.zero_grad()
            pz, nz, summ = model(x, ei)
            loss = model.loss(pz, nz, summ)
            loss.backward(); opt.step(); tot += loss.item()
    model.eval()
    Z = np.zeros((len(nodes), 64), np.float32)
    with torch.no_grad():
        for idx, x, ei in graphs:
            Z[idx] = enc(x, ei).numpy()
    return Z, tot / len(graphs)


EMB_BY_FOLD = {}


def recipe_G_dgi(tr, pr, all_df, seed):
    emb = EMB_BY_FOLD[int(pr.time_step.max())]
    add = lambda d: d.merge(emb, on='txId', how='left')
    old = list(proxy.load()['tree'])
    proxy.load()['tree'] = old + EMB
    try:
        return proxy.recipe_G(add(tr), add(pr), add(all_df), seed)
    finally:
        proxy.load()['tree'] = old


if __name__ == '__main__':
    t0 = time.time()
    nodes, X, pos, edges = node_table()
    print('nodes {} | featureless {} | {:.0f}s'.format(len(nodes), int((X[:, -1] == 0).sum()), time.time() - t0), flush=True)
    for (name, trs, prs) in [f for f in proxy.FITS if f[0] in ('t15', 't21', 't27')]:
        last = max(prs)
        t1 = time.time()
        Z, loss = train_dgi(nodes, X, pos, edges, last)
        keep = nodes.step.values <= last
        trmask = nodes.step.values <= max(trs)
        pca = PCA(N_COMP, random_state=0).fit(Z[trmask])
        emb = pd.DataFrame(pca.transform(Z[keep]).astype(np.float32), columns=EMB)
        emb.insert(0, 'txId', nodes.txId.values[keep])
        EMB_BY_FOLD[last] = emb
        print('dgi fold {} (steps 1-{}) loss {:.3f} | {:.0f}s'.format(name, last, loss, time.time() - t1), flush=True)
    rows = []
    for label, fn in [('G', 'G'), ('G+dgi', recipe_G_dgi)]:
        res, sec = proxy.evaluate(fn, seeds=(0, 1, 2), proxies=['fwd'], verbose=False)
        res.insert(0, 'config', label)
        rows.append(res)
        print('{:6s} auc {:.4f} | ap {:.4f} | worst {:.4f} | {:.0f}s'.format(label, res.auc[0], res.ap[0], res.worst[0], sec), flush=True)
    out = pd.concat(rows)
    out.to_csv(os.path.join(HERE, 'dgi_scores.csv'), index=False)
    g, d = out.iloc[0], out.iloc[1]
    ok = (d.ap - g.ap >= 0.0015) and (d.auc >= g.auc) and (d.worst - g.worst >= -0.005)
    print('dgi vs G --> d_auc {:+.4f} d_ap {:+.4f} d_worst {:+.4f} --> {}'.format(d.auc - g.auc, d.ap - g.ap, d.worst - g.worst, 'PASS' if ok else 'fail'))
