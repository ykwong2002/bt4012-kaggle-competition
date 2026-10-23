# round 12 screen --> reconstruction-error features (lecture 04) and decoded input/output ratios
import os, time
import numpy as np, pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import proxy

HERE = os.path.dirname(os.path.abspath(__file__))
REC = ['rec_err', 'rec_err_top', 'rec_err_rest', 'rec_err_in_mean', 'rec_err_out_mean']
RAT = ['n_in', 'n_out', 'spent_share', 'obs_in_share', 'unspent_out', 'unobserved_in', 'one_in_one_out']

def decode_count(x):
    # values sit on a regular grid --> (x - min) / step + 1 is a whole number
    u = np.unique(np.round(x, 6))
    step = np.min(np.diff(u))
    return np.round((x - u.min()) / step) + 1

def build(fe):
    df = fe['df']
    kept, nb = fe['kept'], fe['nb']
    t0 = time.time()
    # reconstruction error --> quantile scaling then pca, fitted on every row without labels
    X = df[kept].values
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0, subsample=200000)
    Z = qt.fit_transform(X)
    pca = PCA(n_components=10, random_state=0).fit(Z)
    R = (Z - pca.inverse_transform(pca.transform(Z))) ** 2
    top_idx = [kept.index(f) for f in nb]
    rest_idx = [i for i in range(len(kept)) if i not in top_idx]
    out = pd.DataFrame({'txId': df.txId.values})
    out['rec_err'] = R.mean(1)
    out['rec_err_top'] = R[:, top_idx].mean(1)
    out['rec_err_rest'] = R[:, rest_idx].mean(1)
    e = fe['edges']
    em = out.set_index('txId').rec_err
    out['rec_err_in_mean'] = out.txId.map(e.assign(v=e.txId1.map(em)).dropna().groupby('txId2').v.mean())
    out['rec_err_out_mean'] = out.txId.map(e.assign(v=e.txId2.map(em)).dropna().groupby('txId1').v.mean())
    # decoded counts --> feat_4 is the number of inputs and feat_5 the number of outputs
    n_in, n_out = decode_count(df.feat_4.values.astype(float)), decode_count(df.feat_5.values.astype(float))
    ind, outd = df.in_degree.values, df.out_degree.values
    assert (ind <= n_in + 1e-6).all() and (outd <= n_out + 1e-6).all()
    out['n_in'], out['n_out'] = n_in, n_out
    out['spent_share'] = outd / n_out
    out['obs_in_share'] = ind / n_in
    out['unspent_out'] = n_out - outd
    out['unobserved_in'] = n_in - ind
    out['one_in_one_out'] = ((n_in == 1) & (n_out == 1)).astype(float)
    print('round 12 features built in {:.0f}s'.format(time.time() - t0), flush=True)
    return out

def attach():
    fe = proxy.load()
    if 'rec_err' not in fe['df'].columns:
        cache = os.path.join(HERE, 'round12_feats.pkl')
        f = pd.read_pickle(cache) if os.path.exists(cache) else build(fe)
        f.to_pickle(cache)
        df = fe['df'].merge(f, on='txId', how='left')
        df[REC + RAT] = df[REC + RAT].astype(np.float32)
        fe['df'] = df
    return fe

def recipe_with(extra):
    def fn(tr, pr, all_df, seed):
        fe = proxy.load()
        saved = fe['tree']
        fe['tree'] = saved + extra
        try:
            return proxy.recipe_G(tr, pr, all_df, seed)
        finally:
            fe['tree'] = saved
    return fn

if __name__ == '__main__':
    fe = attach()
    df = fe['df']
    lab = df[df[proxy.LABEL].notna() & ~df.is_test]
    y_adv = df.is_test.astype(int).values
    print('feature --> univariate auc on labelled train | train-vs-test auc of the feature alone')
    for c in REC + RAT:
        print('  {:<18} {:.3f} | {:.3f}'.format(c, roc_auc_score(lab[proxy.LABEL].astype(int), lab[c].fillna(-1)), roc_auc_score(y_adv, df[c].fillna(-1))), flush=True)
    rows = []
    for name, fn in [('G', 'G'), ('G_rec', recipe_with(REC)), ('G_ratio', recipe_with(RAT))]:
        res, secs = proxy.evaluate(fn, seeds=(0, 1, 2), proxies=['fwd'], verbose=False)
        res['recipe'] = name
        rows.append(res)
        print('{:<8} auc {:.4f} | ap {:.4f} | worst-step {:.4f} | {:.0f}s'.format(name, res.auc.iloc[0], res.ap.iloc[0], res.worst.iloc[0], secs), flush=True)
    out = pd.concat(rows)
    out.to_csv(os.path.join(HERE, 'round12_results.csv'), index=False)
    base = out[out.recipe == 'G'].iloc[0]
    for name in ['G_rec', 'G_ratio']:
        r = out[out.recipe == name].iloc[0]
        ok = (r.ap - base.ap >= 0.0015) and (r.auc >= base.auc) and (r.worst >= base.worst - 0.005)
        print('{} --> ap {:+.4f} | auc {:+.4f} | worst {:+.4f} -> {}'.format(name, r.ap - base.ap, r.auc - base.auc, r.worst - base.worst, 'PASS' if ok else 'fail'))
