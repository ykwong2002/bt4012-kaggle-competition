# per-step prior correction (saerens em) on the forward-chaining proxy --> does shifting each step by its estimated fraud rate help pooled auc
import numpy as np, pandas as pd
import proxy
from round10 import cached_G

def em_prior(p, pi_tr, iters=100, tol=1e-6):
    # saerens et al 2002 --> re-estimate the class prior of a new batch from posteriors trained under pi_tr
    pi = pi_tr
    for _ in range(iters):
        a = (pi / pi_tr) * p
        b = ((1 - pi) / (1 - pi_tr)) * (1 - p)
        post = a / (a + b)
        new = post.mean()
        if abs(new - pi) < tol:
            break
        pi = new
    return pi

def shift(p, pi_tr, pi_new, strength=1.0):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    delta = np.log(pi_new / pi_tr) - np.log((1 - pi_new) / (1 - pi_tr))
    return logit + strength * delta

def recipe_prior(strength, mode='em'):
    def fn(tr, pr, all_df, seed):
        p = np.asarray(cached_G(tr, pr, all_df, seed))
        if strength == 0:
            return p
        pi_tr = float(tr[proxy.LABEL].mean())
        out = np.empty(len(p))
        for t in np.unique(pr.time_step.values):
            k = pr.time_step.values == t
            pi_t = em_prior(p[k], pi_tr) if mode == 'em' else float(p[k].mean())
            pi_t = float(np.clip(pi_t, 1e-4, 1 - 1e-4))
            out[k] = shift(p[k], pi_tr, pi_t, strength)
        return out
    return fn

if __name__ == '__main__':
    rows = []
    for name, fn in [('G (no shift)', recipe_prior(0)), ('em shift x1', recipe_prior(1.0)), ('em shift x0.5', recipe_prior(0.5)),
                     ('mean-prob shift x1', recipe_prior(1.0, 'mean')), ('em shift x2', recipe_prior(2.0))]:
        res, secs = proxy.evaluate(fn, seeds=(0, 1, 2), proxies=['fwd'], verbose=False)
        print('{:<20} auc {:.4f} | ap {:.4f} | worst-step {:.4f} | {:.0f}s'.format(name, res.auc.iloc[0], res.ap.iloc[0], res.worst.iloc[0], secs), flush=True)
