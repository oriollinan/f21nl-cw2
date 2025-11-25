# Answers

Sampling does help fix the error partially.

Only when the **K**, **P**, and **Temperature** hyperparameters are right.

*Greedy* sampling gives the following output:

```txt
Deux garçons sont assis sur un canapé qui est en rouge et noir
```

When doing sampling without **P** and **K** we get:

```txt
Deux garçons sont assis sur un d foncés qui est assis sur un canapé rouge et noir
```

With the following parameters we can achive the following output:

| K |  P  | Temperature |
| - | --- | ----------- |
| 4 | 0.8 |    0.25     |

```txt
Deux garçons sont assis sur un canapé qui est en rouge et noir</s>
```

After doing a grid search of the parameters we found
that **Temperature** has the most impact on the generated output.
