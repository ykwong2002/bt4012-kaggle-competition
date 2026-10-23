# tuning screen for the adversarial mlp --> eps grid, 3-step bim, virtual adversarial training, adversarial old mlp
import os, time, random, sys
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
from scipy import stats

torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = open(os.path.join(ROOT, '.dev', 'build_notebook.py')).read()
def block(a, b):
    i = SRC.index(a); return SRC[i:SRC.index(b, i)]
SEED = 42
DEVICE = torch.device('cpu')
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
# the notebook's own model code --> FraudMLP, make_loader, predict_mlp, PeriodicEmbedding, EmbedMLP
exec(block('class FraudMLP(nn.Module):', 'def train_mlp('))
exec(block('class PeriodicEmbedding(nn.Module):', 'def adversarial_loss('))

D = np.load(os.path.join(ROOT, 'outputs', 'mlp_screen_inputs.npz'))
SEEDS3 = [42, 7, 123]
crit = nn.BCEWithLogitsLoss()

def fgsm(model, xb, yb, eps):
    x = xb.clone().requires_grad_(True)
    clean = crit(model(x), yb)
    g, = torch.autograd.grad(clean, x, retain_graph=True)
    return 0.5 * (clean + crit(model((xb + eps * g.sign()).detach()), yb))

def bim(model, xb, yb, eps, steps=3):
    x = xb.clone()
    for _ in range(steps):
        x.requires_grad_(True)
        g, = torch.autograd.grad(crit(model(x), yb), x)
        x = torch.max(torch.min(x.detach() + (eps / 2) * g.sign(), xb + eps), xb - eps)
    return 0.5 * (crit(model(xb), yb) + crit(model(x.detach()), yb))

def bern_kl(p, q):
    p, q = p.clamp(1e-6, 1 - 1e-6), q.clamp(1e-6, 1 - 1e-6)
    return (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).mean()

def vat(model, ub, eps_l2=0.5, xi=1e-6):
    # virtual adversarial training --> keep the prediction stable around unlabelled rows, no labels needed
    with torch.no_grad():
        p = torch.sigmoid(model(ub))
    d = torch.randn_like(ub)
    d = (d / d.norm(dim=1, keepdim=True)).requires_grad_(True)
    g, = torch.autograd.grad(bern_kl(p, torch.sigmoid(model(ub + xi * d))), d)
    r = eps_l2 * g / (g.norm(dim=1, keepdim=True) + 1e-12)
    return bern_kl(p, torch.sigmoid(model(ub + r.detach())))

def step_loss(model, xb, yb, cfg, U):
    kind = cfg['kind']
    if kind == 'fgsm':
        loss = fgsm(model, xb, yb, cfg['eps'])
    elif kind == 'bim':
        loss = bim(model, xb, yb, cfg['eps'])
    else:
        loss = crit(model(xb), yb)
    if cfg.get('vat') and U is not None:
        idx = torch.randint(0, U.shape[0], (xb.shape[0],))
        loss = loss + cfg['vat'] * vat(model, U[idx])
    return loss

def run_epochs(model, X, y, n_epochs, cfg, U, X_va=None, y_va=None, patience=5, swa_last=None):
    opt = optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-3))
    loader = make_loader(X, y, batch_size=cfg.get('batch', 256), shuffle=True)
    swa = torch.optim.swa_utils.AveragedModel(model) if swa_last else None
    best, best_ep, best_state, wait = -1, 1, None, 0
    for ep in range(1, n_epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            step_loss(model, xb, yb, cfg, U).backward()
            opt.step()
        if swa is not None and ep > n_epochs - swa_last:
            swa.update_parameters(model)
        if X_va is not None:
            auc = metrics.roc_auc_score(y_va, predict_mlp(model, X_va))
            if auc > best:
                best, best_ep, wait = auc, ep, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            if wait >= patience:
                break
    return swa, best_ep, best_state

class TunedMLP(nn.Module):
    def __init__(self, n_features, cfg):
        super().__init__()
        d = cfg.get('d', 8)
        self.embed = PeriodicEmbedding(n_features, k=cfg.get('k', 8), d=d, sigma=cfg.get('sigma', 0.1))
        self.body = FraudMLP(n_features * d, hidden_sizes=cfg.get('hidden', (256, 128)), dropout=cfg.get('dropout', 0.3))

    def forward(self, x):
        return self.body(self.embed(x))

def screen_embed(cfg):
    # same protocol as section 8.18/8.20 --> early stopping picks the epochs, then swa over the last 5
    Xtr, ytr, Xva, yva = D['Q_tr'].astype(np.float32), D['y_tr'], D['Q_va'].astype(np.float32), D['y_va']
    U = torch.tensor(Xva) if cfg.get('vat') else None
    preds = []
    for s in SEEDS3:
        set_seed(s)
        _, ep, _ = run_epochs(TunedMLP(Xtr.shape[1], cfg), Xtr, ytr, 60, cfg, U, Xva, yva)
        set_seed(s)
        sw = cfg.get('swa_last', 5)
        swa, _, _ = run_epochs(TunedMLP(Xtr.shape[1], cfg), Xtr, ytr, max(ep, sw) + 2, cfg, U, swa_last=sw)
        preds.append(predict_mlp(swa, Xva))
    return preds

def screen_old(cfg):
    # old matched-graph mlp --> early stopping keeps the best epoch, as train_mlp does
    Xtr, ytr, Xva, yva = D['X_tr_m'].astype(np.float32), D['y_tr'], D['X_va_m'].astype(np.float32), D['y_va']
    preds = []
    for s in SEEDS3:
        set_seed(s)
        m = FraudMLP(Xtr.shape[1])
        _, _, state = run_epochs(m, Xtr, ytr, 100, cfg, None, Xva, yva)
        m.load_state_dict(state)
        preds.append(predict_mlp(m, Xva))
    return preds

def summary(name, preds, ref=None):
    yva = D['y_va']
    per = [metrics.roc_auc_score(yva, p) for p in preds]
    avg = np.mean([stats.rankdata(p) / len(p) for p in preds], axis=0)
    line = '{:<28} 3-seed auc {:.4f} | per seed {} | sd {:.4f}'.format(name, metrics.roc_auc_score(yva, avg), ' '.join('{:.4f}'.format(a) for a in per), np.std(per))
    if ref is not None:
        line += ' | spearman with ref {:.4f}'.format(stats.spearmanr(avg, ref).correlation)
    print(line, flush=True)
    return avg

if __name__ == '__main__':
    which = sys.argv[1:] or ['ref', 'eps0.01', 'eps0.05', 'eps0.1', 'bim0.03', 'vat', 'old', 'old_fgsm']
    ref = summary('fgsm 0.03 (reference, saved)', list(D['mlp3_va_seeds'][:3]))
    cfgs = {'eps0.01': ('embed', dict(kind='fgsm', eps=0.01)), 'eps0.05': ('embed', dict(kind='fgsm', eps=0.05)),
            'eps0.1': ('embed', dict(kind='fgsm', eps=0.1)), 'bim0.03': ('embed', dict(kind='bim', eps=0.03)),
            'vat': ('embed', dict(kind='fgsm', eps=0.03, vat=1.0)),
            'old': ('old', dict(kind='plain')), 'old_fgsm': ('old', dict(kind='fgsm', eps=0.03))}
    base = dict(kind='fgsm', eps=0.03)
    for name, extra in [('d16', dict(d=16)), ('k16', dict(k=16)), ('sigma0.05', dict(sigma=0.05)), ('sigma0.2', dict(sigma=0.2)),
                        ('dropout0.2', dict(dropout=0.2)), ('dropout0.4', dict(dropout=0.4)), ('hidden512', dict(hidden=(512, 256))),
                        ('lr5e-4', dict(lr=5e-4)), ('batch512', dict(batch=512)), ('swa10', dict(swa_last=10)),
                        ('combo_s02_d16', dict(sigma=0.2, d=16)), ('combo_s02_d16_k16', dict(sigma=0.2, d=16, k=16)),
                        ('t_s015', dict(sigma=0.15, d=16)), ('t_s03', dict(sigma=0.3, d=16)), ('t_d24', dict(sigma=0.2, d=24)),
                        ('t_d32', dict(sigma=0.2, d=32)), ('t_k12', dict(sigma=0.2, d=16, k=12)), ('t_eps005', dict(sigma=0.2, d=16, eps=0.05)),
                        ('t_drop02', dict(sigma=0.2, d=16, dropout=0.2)), ('t_h384', dict(sigma=0.2, d=16, hidden=(384, 192))),
                        ('t_combo', dict(sigma=0.2, d=24, dropout=0.2, eps=0.05))]:
        cfgs[name] = ('embed', dict(base, **extra))
    for w in which:
        if w == 'ref':
            continue
        t0 = time.time()
        fam, cfg = cfgs[w]
        preds = screen_embed(cfg) if fam == 'embed' else screen_old(cfg)
        summary('{} ({:.0f}s)'.format(w, time.time() - t0), preds, ref if fam == 'embed' else None)
