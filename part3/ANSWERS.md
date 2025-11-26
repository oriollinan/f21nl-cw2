# Answers

## Example 1

The errors seen in both models prediction can be explained by the fact that in the training dataset, there are very few examples with the same linguistic phenomena as the source text (**subject relative clauses** with **coordination**). The fact that the **coordination** is happening in a **relative clause** implies a challenge to both models. This is due to the **coordination** refering to a previous stated noun.

Furthermore, because the phenomena appear near the end of the sentence, the model has greater difficulty translating it. If the phenomena occurred at the beginning instead, both models would generally handle it better, since earlier elements face less competition from surrounding context and receive clearer attention patterns. As a result, the prediction would be more reliable because the model can focus more effectively on the relevant structure.


### Bi-LSTM

The model's prediction is obviously wrong. First of all, `en` should be omitted. Secondly, the coordination `noir et noir` does not make sense as it should be `rouge et noir`.

In the attention heatmap it can also be seen that the target token `en` fully attends the source token `red`. This is a bad prediction, since `en`  is not the correct translation in this context. Such error can make the model's internal state to be disaranged, possibly leading the model to make further mistakes later, like predicting `noir` twice instead of `rouge` and `noir`.

### Transformer

The main error in the prediction is the use of `vêtus de`, which should be omitted.

This error, can be explained by the fact that in our training dataset there are no phrases with the following structure `qui est rouge` or `qui est noir`, as it should predict. Therefore, the model hasn't learnt to use colors after `qui est`. On the other hand, the phrases `vêtus de rouge` and `vêtus de noir` appear multiple times. Meaning that the model has learnt to use the phrase `vêtus de` followed by a color.


## Example 2

The phenomena that can be seen in this example is a **coordination**. But, unlike *Example 1* it is appearing in the begining of the sentence.

In this example both the Bi-LSTM and the Transformer predicted the gold sentence correctly. After reviewing the dataset we believe this is due to 2 factors:

1. `A boy and a girl` appears frequently at the begining of the sentence. Meaning that model has no trouble translating the coordination in this case.
2. `sont assis sur un canapé` also appears multiple times, even refering to colors.

Furthermore, in the case of the Bi-LSTM, we could expect that the predicted token after `qui est` would be `en` as we saw in *Example 1*. But, unlike *Example 1* the attention computed after predicting `est`, attends to `is`, `rouge` and more importantly `.`. The fact that the attention is drawn to `.`, tells the model that the token which will be predicted should be the last one. Thus, predicting `en` would not make sense as the phrase would end up incomplete.

## Example 3

The phenomenas appearing in this examples consist of a **coordination** and a **ambiguous subject noun phrase**. In this case, the **transformer** clearly outperforms the **Bi-LSTM**. As it can be seen, the **Bi-LSTM** has trouble translating the **ambiguous subject noun phrase**. Meanwhile the **transformer** translates the source sentence correctly.

### Bi-LSTM

The model prediction presents some errors. First, `et des filles` even though is syntacticcaly and grammatically correct is not faithful to the source sentence, the expected prediction should be `et filles`. It is worth mentioning that the addition of the word `des` which means `some` in English, doesn't really suppose a change to the meaning of the sentence. As the phrase `Two boys and girls` can be understood as `two girls` or some undefined number of girls which would be the same as `some girls`.

The main problem in the target sentence is the conjugation of the verb `asseoir` (`sit` in English). Instead of `assis` the prediction is `assises`. The main difference between the conjugations, is the gender. Both conjugations are the plural past participle. But, `assis` is masculine and `assises` feminine. The conjugation `assises` would make sense if the sentece would only talk about the girls, omitting the `Two boys`.

In *Example 2* we can see a very similar example, being the main the difference the subject (`A boy and a girl`). The conjugation of the verb `asseoir` in *Example 2* is `assis`, which is correct. This means that the error could rely on the fact that in the subject there are different quantifiers for both boys (`Two`) and girls (`some`), which could destabilize the model into a state that has not been trained before and prevent it to conjugate properly.

### Transformer

The prediction made by the transformer is correct, as can be seen in the attention heatmap. Multiple syntactic heads are visible, providing the model with information about the **ambiguous subject noun phrase** and ensuring a correct translation.

## Example 4

In this example we can see two different linguistic **phenomena**, a **present continuous participle** (`loading`) and a **collective noun phrase in singular form** (`A group of men`). The phrase `a group of men` **appears** multiple times in the dataset, ensuring that both models translate the phrase correctly. It is important to say that in **English** the **present continuous** is used to express an action currently happening.

It can be seen that in both models the verb charger is conjugated in **present participle** (`chargeant`), which is incorrect. Because, in French, a participle **cannot** be used as the main verb of a sentence. The expected conjugation is the **present simple** (`charger`). The errors can be explained by the fact that the models did not learn how the **present participle** works given the challenge it suposes.

Furthermore, both models fail to translate the word cotton properly. The reason for this is that the word cotton appears only seven times in the training dataset. Six of those seven occurrences refer to cotton candy (`barbe à papa`), which is translated differently from cotton (`coton`).

### Bi-LSTM

The predicted sentence is grammatically incorrect. First, as stated previously, the **present participle** (`chargeant`) cannot be used as the main verb in a sentence. One reason that can justify the error is that, after predicting `_charge`, the attention is drawn to `ing_` and `_cott`. Although the expected prediction is `_du`, the model's prediction of `ant` is understandable, as the French suffix -ant functions similarly to the English -ing.

In the case of the phrase `à la main d'un camion`, which should be `du coton sur un camion`, the error can be explained by the fact that the model does not know how to translate the word cotton in this context. Although we can see that the model predicts a token belonging to the French expression for cotton candy (`barbe à papa`), this suggests that the model has learned only how to translate cotton candy and not cotton.


### Transformer

As already explained, the **present participle** (`chargeant`) is being used incorrectly. In the case of the phrase `de la barbe`, which should be `du coton`, the error reflects the issue mentioned earlier: the model assumes that cotton refers to cotton candy and therefore tries to predict something related to it. This is why we see `barbe`, as in `barbe à papa`.

## Example 5

The last example, which at first glance seems the simplest out of all the source sentences, results in an error in both predictions. The error may be due to the absence of sentences in the past tense in the training dataset, resulting in both models conjugating the verb `walked` incorrectly. In both models attention heatmaps, we can see that the predicted tokens `_passent` and `_marchent` are attending to `_walk`, but they are not attending to `ed`. This means that the models do not know that the predicted token should be in the past tense.