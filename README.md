# NLP Final Competition - Team Wiki
👉 *Published by HackMD: https://hackmd.io/@KoiSharp/rygtpQAL5*
## Proposal
- We will use [pytorch](https://github.com/pytorch/pytorch)(or [Tensorflow]()) and [huggingface transformers API](https://github.com/huggingface/transformers)
### Baseline Proposal
- Vanilla BERT (Standard) as our baseline
- Only consider the prompts, exluding conversations 
- Use AdamW as our optimizer

### Master Proposal
- Use two BERT or DistilBert with different weights to infer `prompt` & `utterance` representations，concatenate the two hypotheses.
- Add a `LayerNorm` layer to receive the concatenated result.
- Use `Linear` layer to do classification.
- Maybe we can use `SAM` to smooth the loss landscape

## Pipline
- [x] [Exploratory Data Analysis](https://hackmd.io/@KoiSharp/rJ8lRrRIc)
    - [x] Preprocessing
    - [x] Visualization
- [ ] Baseline
    - [ ] Tokenization
    - [ ] Custom Dataset
    - [ ] Building Model
    - [ ] Training
    - [ ] Evaluation
- [ ] Master Model
- [ ] Optimization

## TODO:
- Browse [huggingface transformers](https://github.com/huggingface/transformers) API Documents
    - [x] Tokenizers
        - [Doc](https://huggingface.co/docs/transformers/main_classes/tokenizer)
        - [Quick Guide](https://huggingface.co/docs/transformers/preprocessing#nlp)
    - [x] Models
        - [BERT](https://huggingface.co/docs/transformers/model_doc/bert)
        - [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)
    - [x] How to customize our dataset with API
        - *~~[Quick Guide](https://huggingface.co/docs/datasets/datasetscript)~~ This is for data publication.*
        - We can customize our dataset by using pytorch Dataset class directly. See [here](https://huggingface.co/transformers/v3.2.0/custom_datasets.html).
    - [ ] How to customize our model
- Review EDA
    - [x] Preserve punctuation
    - [ ] Remove numeric tokens
    - [ ] Check the word co-occurences in train/valid/test
    - [x] Do **NOT** use multiple '[SEP]' tokens in one sample. See [this discussion](https://discuss.huggingface.co/t/combine-multiple-sentences-together-during-tokenization/3430/4) and [issue#65](https://github.com/huggingface/transformers/issues/65).
        - Is there any alternative to separate sentences rather than using '[SEP]'?
        - Is it possible to create a custom token with its own embedding? (Yes, refer to [here](https://github.com/huggingface/transformers/issues/1413))
        - Currently we used '[SPEAKER_A]' and '[SPEAKER_B]' to separate different utterance.
- There's no way to keep google colab awake now hahaha. Refer to [here](https://stackoverflow.com/questions/57113226/how-to-prevent-google-colab-from-disconnecting).

## History
| Version | Train data | Valid data | Model | Batch size | Result |
| --- | --- | --- | --- | --- | --- |
| [0516](https://hackmd.io/mV9R-4W_TP2Q1n2JSHHcZw) | Random 377/label | All | distilbert-base-uncased | 16/64 | Overfitting
