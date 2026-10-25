# BT4012 In-class Kaggle Competition 2026 - e1155727

Predicting the probability that a Bitcoin transaction is illicit, scored by ROC AUC.
Competition page: https://www.kaggle.com/competitions/bt-4012-competition-2026
Repository: https://github.com/ykwong2002/bt4012-kaggle-competition

## Files

| path | purpose |
|---|---|
| `e1155727.ipynb` | full pipeline: EDA, graph features, four models, blending, submission files |
| `requirements.txt` | pinned package versions used to produce the results |
| `data/` | the four competition csv files (not committed, see below) |
| `outputs/` | figures and metric tables written by the notebook (generated on run, not committed) |
| `submissions/` | submission csv files written by the notebook (generated on run, not committed) |
| `report/` | the competition report (submitted separately as e1155727.pdf) |
| `experiments/` | offline screening scripts behind the experiment log (forward-chaining checks, model screens) |
| `experiment_log.csv` | every experiment: hypothesis, change, offline and public results, verdict, lesson |

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
3. Run the notebook top to bottom with the standard Python 3 kernel, either in Jupyter or headlessly:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 e1155727.ipynb
   ```

   The full run takes about 2.5 hours on a laptop; the seed-bagged stacking sections and the 15-seed
   networks (8.25, 8.26) are the slow parts. Keep the machine awake and on power during the run. Seeds are fixed (42) so the
   validation numbers and submission files are reproduced exactly on the same machine.
4. Submission files appear in `submissions/`. To upload one with the Kaggle CLI:

   ```bash
   kaggle competitions submit -c bt-4012-competition-2026 -f submissions/<file>.csv -m "<message>"
   ```

## Final submissions

These are the two files selected for the final (private) leaderboard. Running the notebook top to bottom (step 3 above) regenerates both in `submissions/`: every model is trained on the labelled steps 1-35 and then predicts the test set in the same run, so no saved model weights are needed.

| file | written in notebook section | model | public / private AUC |
|---|---|---|---|
| `submissions/blend_lr_mlp5s15_1to3.csv` | 8.25 | rank blend: stacked LightGBM 1 : 15-seed adversarially trained MLP 3 | 0.9647 / 0.9526 |
| `submissions/blend_lr_mlp_mlp2_mlp3_1to1to1to1.csv` | 8.20 | rank blend, equal weight: stacked LightGBM, original MLP, quantile-embedding MLP, adversarially trained MLP | 0.9635 / 0.9533 |

Seeds are fixed. On the original machine a full rerun reproduced both files exactly (rank correlation 1.0 with the uploaded files, checked on 23 Oct 2026). On a different machine or library build, tiny floating-point differences in LightGBM and PyTorch can shift individual scores very slightly. The cell in section 8.25 also prints the validation metrics reported in Table 2 of the report.

## Approach in brief

* time-based validation (train on steps 1-27, validate on 28-35), later backed by forward-chaining folds and a long-horizon fold, because the test period (steps 36-49) drifts away from training
* label-free graph features from the edge list (degrees, mean and max of neighbour features)
* two-stage stacking: neighbours' out-of-fold LightGBM scores (grouped by time step, so no label leaks) fed to a second LightGBM, plus logistic-regression neighbour scores, 5 seeds
* a neural network with quantile scaling, periodic numeric embeddings, weight averaging and FGSM adversarial training, 15 seeds
* final submission: rank blend stack 1 : network 3 (`submissions/blend_lr_mlp5s15_1to3.csv`, public AUC 0.9647); second pick: stack and three earlier networks at equal weight (`submissions/blend_lr_mlp_mlp2_mlp3_1to1to1to1.csv`, 0.9635)
* 83 experiments logged in `experiment_log.csv`, judged against noise thresholds set in advance

## AI assistance

Claude (Anthropic), through Claude Code, was used as a coding, research and writing assistant: it wrote and debugged code, ran experiments, summarised published methods and drafted the report and documentation, which the student reviewed. AI agents were also used as an experiment advisor, as reviewers, as research assistants and as independent challengers. Near the end, at the student's instruction, the assistant chose and ran some experiments itself (marked AI-selected in `experiment_log.csv`), and the 1:3 weighting of the first final pick was proposed by the assistant and kept by the student. The full disclosure is in the report.
