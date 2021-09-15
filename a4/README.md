# Neural Machine Translation (NMT) Assignment

## Task
Machine Translation:  
Convert from **Spanish** to **English**.

## Result
### Corpus BLEU score
27.06437722023294 (Based upon the model trained for 13 epochs.)

### Interpretation of BLEU score
*The gist is clear, but has significant grammatical errors.*

For details, have a look at the [table](https://cloud.google.com/translate/automl/docs/evaluate#interpretation) which describes the interpretation of what the score range means.

#### Output
- [csv](./outputs/test_es_en_translation.csv)

- Columns:
    - **source**: Spanish (source) sentences
    - **translation_reference**: English (target) reference sentences
    - **translation_hypothesis**: English translation by NMT model
    - **sentence_bleu_score**: Sentence BLEU score


## Errata
Assignment code had the following error in the function
`utils.py # read_corpus()`  
 
Many sentences in `test.en` have consecutive multiple space characters. 
`line.strip().split(' ')` leads to empty strings in the split output.
Whereas the default `sep` parameter (i.e. `None`) of `split` discards the empty strings from the output.  

This led to increase of BLEU score by approximately 4.6.

#### Reference
https://stackoverflow.com/questions/2492415/how-can-i-split-by-1-or-more-occurrences-of-a-delimiter-in-python

## Note
Assignment heavily inspired by the https://github.com/pcyin/pytorch_nmt repository
