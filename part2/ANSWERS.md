| name               | embed_size | head_size | layers |
| ------------------ | ---------- | --------- | ------ |
| transformer_small  | 16         | 2         | 2      |
| transformer_medium | 64         | 4         | 4      |
| transformer_large  | 192        | 6         | 6      |

| name               | score |
| ------------------ | ----- |
| transformer_small  | 15.82 |
| transformer_medium | 46.26 |
| transformer_large  | 50.69 |
| bi_lstm_large      | 55.97 |

![image_train_loss](screenshot)
![image_dev_loss](screenshot)

As we could see in the Bi-LSTM models, bigger parameters lead to a better blue score and once we reach a substantial parameter size, the BLEU score starts increasing more slowly. The transformer model which best ensembles to the best performing Bi-LSTM (`bi_lstm_large`) is the `transformer_large`. One explanation for the difference in score achieved could be that the transformers sampling hyperparameters have not been finetuned.
