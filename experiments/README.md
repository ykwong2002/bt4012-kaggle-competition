# offline experiment scripts

these scripts produced the offline screens recorded in `experiment_log.csv` (rows marked built / screened out, and the forward-chaining evidence for uploaded rows)

- `proxy/proxy.py` --> forward-chaining check (train 1-15 / 1-21 / 1-27, validate on the next block, 3 seeds) and the shared feature cache
- `proxy/testlike.py` --> test-like variant where validation rows see only labelled neighbours
- `proxy/*_screen.py`, `proxy/round*.py` --> individual screens (centrality, social network features, priors, gbdt stage 2, random forest, mlp neighbour scores, final rounds, graph embeddings)
- `mlp_screen.py` --> adversarial mlp tuning screens
- `gnn_screen.py`, `gnn_final.py` --> gcn / gat / graphsage screen and the uploaded gcn (E56, E58)
- `blind/` --> scorer with the long-horizon fold and the benchmark used against the independent challenger agents (E76-E81); the challengers' own code is not included, their methods are described in the log

they were run from a private working folder (`.dev/`), expect the data in `data/`, and some read helper code from the notebook generator, so paths may need adjusting; the submissions themselves are all produced by `e1155727.ipynb`
code written with ai assistance (claude code), see the report's ai disclosure
