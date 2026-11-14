# Model zoo results (30-min direction, 5m features, walk-forward)

Columns: validation (2022-24) then test (2025-26). The stacked meta-learner is fitted on the validation predictions only, so its validation numbers are in-sample; judge it on test. placebo_acc = accuracy with shuffled training labels on the 2025 fold.

```
base rate (majority class): val 0.5155  test 0.5093   | rows: val 53173 test 32530
                 model  val_acc  val_auc  val_logloss  val_top10_acc  test_acc  test_auc  test_logloss  test_top10_acc  placebo_acc
                logreg   0.5057   0.5047       0.6946         0.5214    0.5056    0.4909        0.6974          0.4971       0.5070
                   lgb   0.5094   0.5080       0.6979         0.5211    0.5088    0.5086        0.6946          0.5063       0.5058
                   xgb   0.5095   0.5079       0.6986         0.5212    0.5076    0.5077        0.6949          0.4998       0.5128
                   cat   0.5028   0.5041       0.7033         0.5075    0.4997    0.4992        0.6967          0.5014       0.5113
       extra_trees_clf   0.5115   0.5075       0.6935         0.5214    0.5011    0.5014        0.6933          0.5100       0.5172
       extra_trees_reg   0.5069   0.5019       0.6929         0.5053    0.5093    0.5101        0.6930          0.5404       0.5187
                   mlp   0.4870   0.4926       0.6960         0.4915    0.5061    0.4914        0.6956          0.5008       0.4986
                   gru   0.4949   0.4998       0.6957         0.4855    0.5074    0.4975        0.6965          0.5054       0.4898
                 cnn1d   0.4919   0.4937       0.6961         0.4923    0.5004    0.4895        0.6975          0.4992       0.5018
           transformer   0.5133   0.5079       0.6951         0.5147    0.5056    0.4953        0.6957          0.5164       0.5151
        AVERAGE of all   0.5104   0.5075       0.6935         0.5137    0.5069    0.4960        0.6941          0.5017          NaN
STACKED (meta-learner)   0.5188   0.5136       0.6923         0.5237    0.5112    0.5118        0.6928          0.5106          NaN

95% interval on a 50% accuracy with 32530 test rows: +/-0.0054  (overlapping 30-min labels, so the effective interval is wider)
model picked on VALIDATION log-loss: extra_trees_reg  -> its test accuracy 0.5093
```

Reading: test base rate 0.5093; best single model 0.5093 (ExtraTrees regressor), stacked 0.5112 (AUC 0.512). The difference to the base rate is about 0.2 points, inside the +/-0.5 interval. No model shows an edge; all placebo runs stay at 49-52%.
