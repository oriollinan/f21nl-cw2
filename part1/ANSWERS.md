| name           | embed_size | hidden_size | score |
| -------------- | ---------- | ----------- | ----- |
| bi_lstm_small  | 16         | 16          | 4.85  |
| bi_lstm_medium | 64         | 64          | 53.71 |
| bi_lstm_large  | 128        | 128         | 55.97 |

![image_train_loss](screenshot)
![image_dev_loss](screenshot)

After computing the BLEU score on all 3 experiments, we can see that the best model is the one with the biggest embedding size and hidden size. The results make sense, as a bigger embedding size and hidden size will help the model keep more information regarding the translation.

It is also interesting, that once we reach a embedding and hidden size of 64, the models BLEU score is very similar to the score achieved with the embedding and hidden size of 128, meaning that once we reach a substantial size the model has the capabilities to produce good translations and increasing the size will result in a very small increase in the score.