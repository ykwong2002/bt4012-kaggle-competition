# do the six time-proxy features help or hurt on later steps --> recipes c and g with and without them
import proxy
TIME = proxy.TIME_PROXIES
def with_time(base):
    def fn(tr, pr, all_df, seed):
        fe = proxy.load()
        saved = fe['tree']
        fe['tree'] = saved + TIME
        try:
            return proxy.RECIPES[base](tr, pr, all_df, seed)
        finally:
            fe['tree'] = saved
    return fn
for name, fn in [('C without (used)', 'C'), ('C with time proxies', with_time('C')), ('G without (used)', 'G'), ('G with time proxies', with_time('G'))]:
    res, secs = proxy.evaluate(fn, seeds=(0, 1, 2), proxies=['fwd'], verbose=False)
    print('{:<22} auc {:.4f} | ap {:.4f} | worst-step {:.4f} | {:.0f}s'.format(name, res.auc.iloc[0], res.ap.iloc[0], res.worst.iloc[0], secs), flush=True)
