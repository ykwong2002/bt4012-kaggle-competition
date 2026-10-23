# test-like recheck of the two-step score features --> validation rows only see labelled neighbours, as in test
import proxy, testlike, sna_screen
proxy.RECIPES['Ghop'] = sna_screen.recipe_G_sna(['hop'])
full, L = testlike.load_views()
for name in ['G', 'Ghop']:
    res, sec = testlike.evaluate_fwd(name, (0, 1, 2), full, L)
    print('{:<6} test-like auc {:.4f} | ap {:.4f} | worst-step {:.4f} | {:.0f}s'.format(name, res.auc.mean(), res.ap.mean(), res.worst.mean(), sec), flush=True)
