# NLP Final Competition - Team Wiki
👉 *Published by HackMD: https://hackmd.io/@KoiSharp/rygtpQAL5*
## Proposal
- We will use [pytorch](https://github.com/pytorch/pytorch) and [huggingface transformers API](https://github.com/huggingface/transformers)
### Baseline Proposal
- Vanilla BERT (Standard) as our baseline
- Only consider the conversations, exlude prompts
- Use Adam as our optimizer

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
    - Tokenizers
    - Models
    - How to customize our dataset with API
    - 