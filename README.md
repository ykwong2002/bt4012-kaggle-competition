# BT4012 In-class Kaggle Competition 2026 - e1155727

Predicting the probability that a Bitcoin transaction is illicit, scored by ROC AUC.
Competition page: https://www.kaggle.com/competitions/bt-4012-competition-2026

## Files

| path | purpose |
|---|---|
| `e1155727.ipynb` | full pipeline: EDA, graph features, four models, blending, submission files |
| `requirements.txt` | pinned package versions used to produce the results |
| `data/` | the four competition csv files (not committed, see below) |
| `outputs/` | figures and metric tables written by the notebook (used in the report) |
| `submissions/` | submission csv files written by the notebook |
| `report/` | the competition report |

## Reproducing the results

1. Download the competition data from Kaggle and place `train.csv`, `test.csv`,
   `txs_edgelist.csv` and `sample_submission.csv` in a folder called `data/` next to the notebook.
2. Install the dependencies (Python 3.12 was used):

   ```bash
   pip install -r requirements.txt
   ```

   On macOS, LightGBM also needs the OpenMP runtime: `brew install libomp`. The notebook pins torch to a
   single thread because torch, scikit-learn and LightGBM each ship their own OpenMP runtime on macOS and
   mixing them in one process can deadlock; this does not change any result.
3. Run the notebook top to bottom, either in Jupyter or headlessly:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 e1155727.ipynb
   ```

   The run takes roughly 20 to 30 minutes on a laptop (the drift experiments in section 8 are the slow part). Seeds are fixed (42) so the
   validation numbers and submission files are reproduced exactly on the same machine.
4. Submission files appear in `submissions/`. To upload one with the Kaggle CLI:

   ```bash
   kaggle competitions submit -c bt-4012-competition-2026 -f submissions/<file>.csv -m "<message>"
   ```

## Approach in brief

* time-based validation (train on steps 1-27, validate on 28-35) to mimic the forward-in-time test set
* label-free graph features from the edge list: degrees plus mean and max of neighbour features
* logistic regression, random forest, LightGBM and a PyTorch MLP, tuned with small grids on validation AUC
* rank-average blend of the strongest models, retrained on all labelled rows for the final submission
* adversarial validation to measure train-to-test drift, then feature removal, regularisation and recency variants scored on the public leaderboard

## AI assistance

Claude Code was used as a coding assistant for boilerplate code, debugging and documentation.
All design decisions (validation scheme, features, models, tuning ranges, final submission choice)
were made by the student and are documented in the notebook's decision log and in the report.
