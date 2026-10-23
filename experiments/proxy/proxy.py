"""Offline validation proxy for the BT4012 Kaggle competition.

Usage (from anywhere):
    caffeinate -i python3 .dev/proxy/proxy.py                 # run recipes A-G, write results
    caffeinate -i python3 .dev/proxy/proxy.py --recipes A C    # subset
    python3 -c "import sys; sys.path.insert(0,'.dev/proxy'); import proxy; print(proxy.evaluate('C'))"

Features are built once and cached in .dev/proxy/features.pkl.
Every recipe is a function fit_predict(train_df, pred_df, all_df, seed) -> score array for pred_df.
Proxies (all trees are fixed counts, nothing is tuned on a proxy block):
  base   train 1-27 -> 28-35        (the current validation, for reference)
  gap    train 1-20 -> 28-35
  wide   train 1-20 -> 21-35 pooled
  fwd    forward chaining 1-15->16-21, 1-21->22-27, 1-27->28-35, pooled (scores rank-normalised per fold)
  block  blocked CV: hold out 1-7, 8-14, 15-21, 22-28, 29-35 (train on the rest), pooled
Metrics per proxy: auc (pooled), ap, worst (lowest per-step auc), mstep (mean per-step auc),
  wauc (auc weighted by train-vs-test adversarial odds of each row).
"""
import os, sys, time, pickle, argparse
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, 'data')
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'features.pkl')
TIME_PROXIES = ['feat_100', 'feat_101', 'feat_103', 'feat_136', 'feat_137', 'feat_139']
E_DROP = ['feat_2', 'feat_3', 'feat_107', 'feat_109', 'feat_113', 'feat_115', 'feat_143', 'feat_151']
LABEL = 'label'
N_JOBS = int(os.environ.get('PROXY_JOBS', -1))  # set PROXY_JOBS=4 to leave CPU for other work


# ----------------------------------------------------------------------------- features
def build_features():
    t0 = time.time()
    tr = pd.read_csv(os.path.join(DATA, 'train.csv'))
    te = pd.read_csv(os.path.join(DATA, 'test.csv'))
    edges = pd.read_csv(os.path.join(DATA, 'txs_edgelist.csv'))
    raw = [c for c in tr.columns if c.startswith('feat_')]
    kept = [f for f in raw if f not in TIME_PROXIES]
    lab = tr[tr[LABEL].notna() & (tr.time_step < 28)]
    # neighbour feature subset: top 30 by univariate auc on steps 1-27 (same rule as the notebook)
    strength = {f: abs(roc_auc_score(lab[LABEL], lab[f]) - 0.5) for f in kept}
    nb = sorted(kept, key=lambda f: -strength[f])[:30]
    te = te.assign(**{LABEL: np.nan})
    nodes = pd.concat([tr[['txId', 'time_step', LABEL] + raw], te[['txId', 'time_step', LABEL] + raw]], ignore_index=True)
    nodes['is_test'] = np.r_[np.zeros(len(tr), bool), np.ones(len(te), bool)]
    g = pd.DataFrame({'txId': nodes.txId.values})
    g['in_degree'] = g.txId.map(edges.txId2.value_counts()).fillna(0)
    g['out_degree'] = g.txId.map(edges.txId1.value_counts()).fillna(0)
    g['total_degree'] = g.in_degree + g.out_degree
    for c in ['in_degree', 'out_degree', 'total_degree']:
        g['log_' + c] = np.log1p(g[c])
    feats = nodes[['txId'] + nb]
    ind = edges.merge(feats, left_on='txId1', right_on='txId').drop(columns=['txId', 'txId1'])
    ia = ind.groupby('txId2')[nb].agg(['mean', 'max'])
    ia.columns = ['in_{}_{}'.format(s, f) for f, s in ia.columns]
    ia['in_nb_known'] = ind.groupby('txId2').size()
    outd = edges.merge(feats, left_on='txId2', right_on='txId').drop(columns=['txId', 'txId2'])
    oa = outd.groupby('txId1')[nb].agg(['mean', 'max'])
    oa.columns = ['out_{}_{}'.format(s, f) for f, s in oa.columns]
    oa['out_nb_known'] = outd.groupby('txId1').size()
    g = g.merge(ia, left_on='txId', right_index=True, how='left').merge(oa, left_on='txId', right_index=True, how='left').fillna(0)
    graph = [c for c in g.columns if c != 'txId']
    df = nodes.merge(g, on='txId', how='left')
    df[kept + graph] = df[kept + graph].astype(np.float32)

    # adversarial odds: how test-like each training row looks (raw kept features, out-of-fold)
    X = df[kept].values
    y = df.is_test.values.astype(int)
    p = np.zeros(len(df))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31, verbose=-1, n_jobs=-1, random_state=0).fit(X[a], y[a])
        p[b] = m.predict_proba(X[b])[:, 1]
    p = np.clip(p, 0.01, 0.99)
    df['adv_w'] = p / (1 - p)
    print('adversarial auc train vs test {:.3f}'.format(roc_auc_score(y, p)))

    out = dict(df=df, edges=edges, kept=kept, graph=graph, nb=nb)
    with open(CACHE, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print('features built in {:.0f}s -> {} ({} kept raw + {} graph)'.format(time.time() - t0, CACHE, len(kept), len(graph)))
    return out


_FE = None
def load():
    global _FE
    if _FE is None:
        if os.path.exists(CACHE):
            with open(CACHE, 'rb') as f:
                _FE = pickle.load(f)
        else:
            _FE = build_features()
        _FE['tree'] = _FE['kept'] + _FE['graph']
        _FE['tree_E'] = [f for f in _FE['tree'] if f not in E_DROP and not any(f.endswith('_' + d) for d in E_DROP)]
    return _FE


# ----------------------------------------------------------------------------- recipes
P_A = dict(num_leaves=127, learning_rate=0.1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
P_C = dict(num_leaves=31, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, min_child_samples=50)
P_F = dict(num_leaves=15, learning_rate=0.02, subsample=0.8, subsample_freq=1, colsample_bytree=0.4, min_child_samples=100)


def lgbm(params, n, seed):
    return lgb.LGBMClassifier(n_estimators=n, random_state=seed, verbose=-1, n_jobs=N_JOBS, **params)


def lab_rows(df):
    return df[df[LABEL].notna()]


def fit_lgb(params, n, feats, tr, pr, seed):
    m = lgbm(params, n, seed).fit(tr[feats].values, tr[LABEL].values.astype(int))
    return m.predict_proba(pr[feats].values)[:, 1]


def rank01(p):
    return stats.rankdata(p) / len(p)


def recipe_A(tr, pr, all_df, seed):
    return fit_lgb(P_A, 101, load()['tree'], tr, pr, seed)


def recipe_B(tr, pr, all_df, seed):
    feats = load()['tree']
    pa = fit_lgb(P_A, 101, feats, tr, pr, seed)
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=1, n_jobs=N_JOBS, random_state=seed)
    rf.fit(tr[feats].values, tr[LABEL].values.astype(int))
    pb = rf.predict_proba(pr[feats].values)[:, 1]
    # components returned as columns, rank-averaged on whatever subset is evaluated
    return np.c_[pa, pb]


def recipe_C(tr, pr, all_df, seed):
    return fit_lgb(P_C, 295, load()['tree'], tr, pr, seed)


def recipe_D(tr, pr, all_df, seed):
    # 'train on steps >= 15 of 35' expressed as: drop the first 40% of the training window
    lo, hi = tr.time_step.min(), tr.time_step.max()
    cut = lo + int(round(0.4 * (hi - lo + 1)))
    return fit_lgb(P_A, 63, load()['tree'], tr[tr.time_step >= cut], pr, seed)


def recipe_E(tr, pr, all_df, seed):
    return fit_lgb(P_A, 84, load()['tree_E'], tr, pr, seed)


def recipe_F(tr, pr, all_df, seed):
    return fit_lgb(P_F, 1104, load()['tree'], tr, pr, seed)


NB_SCORE = ['nb_in_score_mean', 'nb_in_score_max', 'nb_in_scored', 'nb_out_score_mean', 'nb_out_score_max', 'nb_out_scored']


def neighbour_scores(score_map, nodes):
    e = load()['edges']
    e = e.assign(s1=e.txId1.map(score_map), s2=e.txId2.map(score_map))
    ia = e.dropna(subset=['s1']).groupby('txId2')['s1'].agg(['mean', 'max', 'size'])
    ia.columns = NB_SCORE[:3]
    oa = e.dropna(subset=['s2']).groupby('txId1')['s2'].agg(['mean', 'max', 'size'])
    oa.columns = NB_SCORE[3:]
    out = pd.DataFrame({'txId': nodes.txId.values})
    out = out.merge(ia, left_on='txId', right_index=True, how='left').merge(oa, left_on='txId', right_index=True, how='left')
    return out.fillna({'nb_in_scored': 0, 'nb_out_scored': 0})


def recipe_G(tr, pr, all_df, seed, n1=295, n2=355):
    feats = load()['tree']
    tr_steps = set(tr.time_step.unique())
    pr_steps = set(pr.time_step.unique())
    X, y, steps = tr[feats].values, tr[LABEL].values.astype(int), tr.time_step.values
    vis = all_df['visible'] if 'visible' in all_df else pd.Series(True, index=all_df.index)
    unl = all_df[all_df.time_step.isin(tr_steps) & all_df[LABEL].isna() & vis]
    parts = []
    for a, b in GroupKFold(n_splits=5).split(X, y, groups=steps):
        m = lgbm(P_C, n1, seed).fit(X[a], y[a])
        parts.append(pd.Series(m.predict_proba(X[b])[:, 1], index=tr.txId.values[b]))
        held = unl[unl.time_step.isin(set(steps[b]))]
        if len(held):
            parts.append(pd.Series(m.predict_proba(held[feats].values)[:, 1], index=held.txId.values))
    full = lgbm(P_C, n1, seed).fit(X, y)
    rest = all_df[all_df.time_step.isin(pr_steps) & vis]  # every node with features in the prediction steps
    parts.append(pd.Series(full.predict_proba(rest[feats].values)[:, 1], index=rest.txId.values))
    smap = pd.concat(parts)
    nbs = neighbour_scores(smap, pd.concat([tr[['txId']], pr[['txId']]]))
    tr2 = tr[['txId', LABEL] + feats].merge(nbs, on='txId', how='left')
    pr2 = pr[['txId'] + feats].merge(nbs, on='txId', how='left')
    return fit_lgb(P_C, n2, feats + NB_SCORE, tr2, pr2, seed)


RECIPES = dict(A=recipe_A, B=recipe_B, C=recipe_C, D=recipe_D, E=recipe_E, F=recipe_F, G=recipe_G)
PUBLIC = dict(A=0.9496, B=0.9388, C=0.9512, D=0.9450, E=0.9443, F=0.9500, G=0.9588)


def combine(P):
    P = np.asarray(P)
    if P.ndim == 1:
        return P
    return np.mean([rank01(P[:, j]) for j in range(P.shape[1])], axis=0)


# ----------------------------------------------------------------------------- splits and metrics
# fits: (name, train steps, predict steps)
FITS = [
    ('t27', range(1, 28), range(28, 36)),
    ('t20', range(1, 21), range(21, 36)),
    ('t15', range(1, 16), range(16, 22)),
    ('t21', range(1, 22), range(22, 28)),
]
BLOCKS = [range(1, 8), range(8, 15), range(15, 22), range(22, 29), range(29, 36)]
FITS += [('b%d' % i, [s for s in range(1, 36) if s not in blk], blk) for i, blk in enumerate(BLOCKS)]

# proxies: list of (fit name, steps evaluated); multiple parts are pooled after per-part rank normalisation
PROXIES = {
    'base': [('t27', range(28, 36))],
    'gap': [('t20', range(28, 36))],
    'wide': [('t20', range(21, 36))],
    'fwd': [('t15', range(16, 22)), ('t21', range(22, 28)), ('t27', range(28, 36))],
    'block': [('b%d' % i, blk) for i, blk in enumerate(BLOCKS)],
}


def metrics_for(parts):
    """parts: list of (y, score, step, adv_w) per fold. pooled metrics after per-fold rank normalisation."""
    ys, ss, st, ws = [], [], [], []
    for y, s, step, w in parts:
        ys.append(y); ss.append(rank01(s) if len(parts) > 1 else s); st.append(step); ws.append(w)
    y, s, step, w = map(np.concatenate, (ys, ss, st, ws))
    per = []
    for t in np.unique(step):
        k = step == t
        if 0 < y[k].sum() < k.sum():
            per.append(roc_auc_score(y[k], s[k]))
    return dict(auc=roc_auc_score(y, s), ap=average_precision_score(y, s), worst=min(per), mstep=float(np.mean(per)),
                wauc=roc_auc_score(y, s, sample_weight=w))


def evaluate(recipe, seeds=(0,), proxies=None, verbose=True):
    fe = load()
    df = fe['df']
    tr_all = df[~df.is_test]
    fn = RECIPES[recipe] if isinstance(recipe, str) else recipe
    proxies = proxies or list(PROXIES)
    needed = sorted({f for p in proxies for f, _ in PROXIES[p]})
    rows = []
    t0 = time.time()
    for seed in seeds:
        preds = {}
        for name, trs, prs in FITS:
            if name not in needed:
                continue
            tr = lab_rows(tr_all[tr_all.time_step.isin(list(trs))])
            pr = lab_rows(tr_all[tr_all.time_step.isin(list(prs))])
            preds[name] = (pr, fn(tr, pr, tr_all, seed))
        for p in proxies:
            parts = []
            for name, steps in PROXIES[p]:
                pr, P = preds[name]
                k = pr.time_step.isin(list(steps)).values
                sub = pr[k]
                parts.append((sub[LABEL].values.astype(int), combine(np.asarray(P)[k]), sub.time_step.values, sub.adv_w.values))
            rows.append(dict(recipe=recipe if isinstance(recipe, str) else 'custom', seed=seed, proxy=p, **metrics_for(parts)))
    res = pd.DataFrame(rows).groupby(['recipe', 'proxy'], sort=False).mean(numeric_only=True).drop(columns='seed').reset_index()
    if verbose:
        print('recipe {} evaluated in {:.0f}s'.format(recipe, time.time() - t0))
    return res, time.time() - t0


def correlations(table):
    out = []
    pub = pd.Series(PUBLIC)
    for (proxy, metric), s in table.items():
        s = s.reindex(pub.index)
        out.append(dict(proxy=proxy, metric=metric, spearman=stats.spearmanr(s, pub)[0], pearson=stats.pearsonr(s, pub)[0],
                        top=s.idxmax(), bottom=s.idxmin()))
    return pd.DataFrame(out).sort_values('spearman', ascending=False).reset_index(drop=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--recipes', nargs='*', default=list(RECIPES))
    ap.add_argument('--seeds', nargs='*', type=int, default=[0])
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    if args.rebuild or not os.path.exists(CACHE):
        build_features()
    load()
    allres, timing = [], {}
    for r in args.recipes:
        res, sec = evaluate(r, seeds=args.seeds)
        timing[r] = sec
        allres.append(res)
        pd.concat(allres).to_csv(os.path.join(HERE, 'proxy_scores_long' + args.tag + '.csv'), index=False)
    long = pd.concat(allres)
    wide = long.pivot(index='recipe', columns='proxy', values=['auc', 'ap', 'worst', 'mstep', 'wauc'])
    wide.columns = [(p, m) for m, p in wide.columns]
    wide = wide[sorted(wide.columns)]
    wide['public', 'auc'] = pd.Series(PUBLIC)
    wide['seconds', ''] = pd.Series(timing)
    wide.to_csv(os.path.join(HERE, 'proxy_scores' + args.tag + '.csv'))
    print(wide.round(4).to_string())
    if len(args.recipes) >= 4:
        corr = correlations({c: wide[c] for c in wide.columns if c[0] not in ('public', 'seconds')})
        corr.to_csv(os.path.join(HERE, 'proxy_correlations' + args.tag + '.csv'), index=False)
        print(corr.round(3).to_string())
