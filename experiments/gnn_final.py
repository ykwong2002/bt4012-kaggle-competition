# gcn on all labelled steps --> test predictions for a diagnostic upload (not a final pick)
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy import stats
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_screen import load, graph, Net, set_seed, ROOT

lab, te, ed, feats, qt = load()
nodes = pd.concat([lab[['txId']], te[['txId']]], ignore_index=True)
X = qt.transform(pd.concat([lab[feats], te[feats]]).values)
x, ei = graph(nodes, ed, X)
n_lab = len(lab)
y = np.r_[lab.label.values.astype(float), np.zeros(len(te))]
train_mask = torch.tensor(np.r_[np.ones(n_lab, bool), np.zeros(len(te), bool)])
yt = torch.tensor(y, dtype=torch.float32)
preds = []
# epoch counts found by early stopping on validation for each seed in gnn_screen.py
for seed, n_ep in [(42, 133), (7, 83), (123, 153)]:
    set_seed(seed)
    model = Net('gcn', x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(n_ep):
        model.train(); opt.zero_grad()
        crit(model(x, ei)[train_mask], yt[train_mask]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        preds.append(torch.sigmoid(model(x, ei)[~train_mask]).numpy())
p = np.mean([stats.rankdata(q) / len(q) for q in preds], axis=0)
ss = pd.read_csv(os.path.join(ROOT, 'data', 'sample_submission.csv'))
assert (ss['index'].values == te['index'].values).all()
sub = pd.DataFrame({'index': ss['index'].values, 'target': p})
assert len(sub) == 15329 and np.isfinite(sub.target).all() and sub.target.between(0, 1).all()
sub.to_csv(os.path.join(ROOT, 'submissions', 'gcn_alone.csv'), index=False)
best = pd.read_csv(os.path.join(ROOT, 'submissions', 'blend_lr_mlp3_1to1.csv')).target.values
print('gcn_alone written | spearman with best {:.4f}'.format(stats.spearmanr(p, best).correlation))
