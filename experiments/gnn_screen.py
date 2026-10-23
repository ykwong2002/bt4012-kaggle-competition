# graph neural network screen --> gcn, gat and direction-aware graphsage on the validation split, test-like graph
import os, sys, time, random
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import QuantileTransformer
from sklearn import metrics
from scipy import stats

torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIME_PROXIES = ['feat_100', 'feat_101', 'feat_103', 'feat_136', 'feat_137', 'feat_139']
VALID_START = 28

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load():
    tr = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    te = pd.read_csv(os.path.join(ROOT, 'data', 'test.csv'))
    ed = pd.read_csv(os.path.join(ROOT, 'data', 'txs_edgelist.csv'))
    feats = [c for c in tr.columns if c.startswith('feat_') and c not in TIME_PROXIES]
    lab = tr[tr.label.notna()].reset_index(drop=True)
    # quantile scaling fitted on every row's features, no labels
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0, subsample=200000)
    qt.fit(pd.concat([tr[feats], te[feats]]).values)
    return lab, te, ed, feats, qt

def graph(nodes, ed, X):
    # test-like graph --> only labelled training rows and test rows are nodes, edges among them
    pos = pd.Series(np.arange(len(nodes)), index=nodes.txId.values)
    e = ed[ed.txId1.isin(pos.index) & ed.txId2.isin(pos.index)]
    ei = torch.tensor(np.vstack([pos[e.txId1].values, pos[e.txId2].values]), dtype=torch.long)
    return torch.tensor(X, dtype=torch.float32), ei

class Net(nn.Module):
    def __init__(self, kind, d_in, h=128, dropout=0.3):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        if kind == 'gcn':
            self.c1, self.c2 = GCNConv(d_in, h), GCNConv(h, h)
        elif kind == 'gat':
            self.c1, self.c2 = GATConv(d_in, h // 4, heads=4), GATConv(h, h // 4, heads=4)
        else:
            # direction aware --> separate aggregators for senders (in) and receivers (out), plus the node itself
            self.s1, self.i1, self.o1 = nn.Linear(d_in, h), SAGEConv(d_in, h, root_weight=False), SAGEConv(d_in, h, root_weight=False)
            self.s2, self.i2, self.o2 = nn.Linear(h, h), SAGEConv(h, h, root_weight=False), SAGEConv(h, h, root_weight=False)
        # skip connection --> the output layer also sees the raw features
        self.out = nn.Linear(h + d_in, 1)

    def layer(self, x, ei, which):
        if self.kind in ('gcn', 'gat'):
            und = to_undirected(ei)
            return (self.c1 if which == 1 else self.c2)(x, und)
        s, i, o = (self.s1, self.i1, self.o1) if which == 1 else (self.s2, self.i2, self.o2)
        return s(x) + i(x, ei) + o(x, ei.flip(0))

    def forward(self, x, ei):
        h = torch.relu(self.layer(x, ei, 1))
        h = nn.functional.dropout(h, self.dropout, self.training)
        h = torch.relu(self.layer(h, ei, 2))
        h = nn.functional.dropout(h, self.dropout, self.training)
        return self.out(torch.cat([h, x], dim=1)).squeeze(-1)

def fit(kind, x, ei, y, train_mask, eval_mask, seed, max_epochs=400, patience=30, lr=5e-3):
    set_seed(seed)
    model = Net(kind, x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y, dtype=torch.float32)
    best, best_state, wait, best_ep = -1, None, 0, 0
    for ep in range(1, max_epochs + 1):
        model.train(); opt.zero_grad()
        crit(model(x, ei)[train_mask], yt[train_mask]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(x, ei)[eval_mask]).numpy()
        auc = metrics.roc_auc_score(y[eval_mask.numpy()], p)
        if auc > best:
            best, best_ep, wait = auc, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x, ei)[eval_mask]).numpy()
    return p, best_ep

if __name__ == '__main__':
    lab, te, ed, feats, qt = load()
    X = qt.transform(lab[feats].values)
    x, ei = graph(lab, ed, X)
    y = lab.label.values.astype(int)
    tmask = torch.tensor(lab.time_step.values < VALID_START)
    vmask = ~tmask
    D = np.load(os.path.join(ROOT, 'outputs', 'mlp_screen_inputs.npz'))
    stack_va = D['stack_va']; mlp3_va = np.mean([stats.rankdata(p) / len(p) for p in D['mlp3_va_seeds']], axis=0)
    yva = y[vmask.numpy()]
    assert len(stack_va) == len(yva) and metrics.roc_auc_score(yva, stack_va) > 0.99, 'validation order mismatch'
    print('graph --> nodes {} | edges {} | validation rows {} | stack valid {:.4f} | adversarial mlp valid {:.4f}'.format(
        x.shape[0], ei.shape[1], int(vmask.sum()), metrics.roc_auc_score(yva, stack_va), metrics.roc_auc_score(yva, mlp3_va)), flush=True)
    r = lambda p: stats.rankdata(p) / len(p)
    base = (r(stack_va) + r(mlp3_va)) / 2
    print('current best recipe (stack 1 : adversarial mlp 1) valid {:.4f}'.format(metrics.roc_auc_score(yva, base)), flush=True)
    for kind in sys.argv[1:] or ['gcn', 'gat', 'sage_dir']:
        t0 = time.time()
        preds, eps_ = [], []
        for s in [42, 7, 123]:
            p, ep = fit(kind, x, ei, y, tmask, vmask, s)
            preds.append(p); eps_.append(ep)
        avg = np.mean([r(p) for p in preds], axis=0)
        per = [metrics.roc_auc_score(yva, p) for p in preds]
        blend3 = (r(stack_va) + r(mlp3_va) + avg) / 3
        print('{:<9} valid {:.4f} | per seed {} | epochs {} | spearman stack {:.3f} mlp3 {:.3f} | blend stack:mlp3:gnn 1:1:1 valid {:.4f} | {:.0f}s'.format(
            kind, metrics.roc_auc_score(yva, avg), ' '.join('{:.4f}'.format(a) for a in per), eps_,
            stats.spearmanr(avg, stack_va).correlation, stats.spearmanr(avg, mlp3_va).correlation,
            metrics.roc_auc_score(yva, blend3), time.time() - t0), flush=True)
