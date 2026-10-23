# Competition overview (BT4012 fraud analytics, private Kaggle competition)

Task: predict the probability that each Bitcoin transaction in the test set is illicit (target = 1).
Metric: ROC AUC on test.csv rows. Public leaderboard = 30% of test rows, private leaderboard = the other 70% (decides ranking).
Submission: CSV with columns index,target (see data/sample_submission.csv), one row per test.csv row, matched on `index`.
Limits: 3 submissions per day; two final submissions are chosen; closes 24 Oct 2026 23:55 SGT.
Rules: no external data; work must be the student's own code, reproducible in one notebook; methods should be explainable by an undergraduate.

Files (all in data/):
- train.csv: txId, time_step (1-35), feat_1..feat_165 (anonymised numeric), label (1 illicit, 0 licit, blank = unknown / unlabelled)
- test.csv: index, txId, time_step (36-49), feat_1..feat_165; labels hidden
- txs_edgelist.csv: directed edges txId1 -> txId2 (payment flow) between transactions
- sample_submission.csv
