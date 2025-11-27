# Answers

The original prediction was:

```txt
Deux garçons sont assis sur un canapé qui est vêtus de rouge et noir.</s>
```

Sampling does help fix the error partially.

Only when the **K**, **P**, and **Temperature** hyperparameters are right.

*Greedy* sampling gives the following output:

```txt
Deux garçons sont assis sur un canapé qui est vêtus de rouge et noir.</s>
```

This is similar to the output that we get from the Bi-LSTM.

When doing sampling without **P** and **K** we get:

```txt
Deux garçons sont assis sur un canapé avec des ceriselle et noir.</s>
```

This output is expected, since the we are sampling from all possible tokens.

After doing a grid search of the parameters these produced the highest bleu score:

|  K  |  P  | Temperature |
| --- | --- | ----------- |
| 200 | 0.8 |    0.25     |

The parameters that affected the output the most is the combination of the **P** and **Temperature**.

Predicting with these parameters gives the following output:

```txt
Deux garçons sont assis sur un canapé qui est vêtus de rouge et noir.</s>
```

Interestingly the output is the same as the original prediction.

After reviewing the predictions of the grid (64 combinations) no output was equal to the gold. This leads us to believe that the issue is in the training step of the model.
