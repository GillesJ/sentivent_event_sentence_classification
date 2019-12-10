# SENTiVENT Event Sentence Classification
Sentence-level event classification experiments for the SENTiVENT Event dataset.
Baseline experiments meant for Journal of LRE dataset submission.

## Experiment requirements:
### Initial test-run:
**Test-run goal**:
- Time one training run to get an idea of train Time.
- Get scores with all types included to assess feasibility of the task

**Multilabel sentence-level transformer-based text classification of main economic event types**:
- Clean pronominal event mentions that can only be handled by coreference.
- Parsed data and event annotations to intermediate format.
- use parser code previously written: make importable
- No hyperparam tuning yet
---

**Evaluation:**
- Train-test split: not yet k-fold.
- Test set: Use IAA gold-standard document as test set. 30 / 288 docs
- Multilabel metric: Determine good metric: http://proceedings.mlr.press/v70/wu17a/wu17a.pdf
  - Default is LRAP
  - (AUC, F1-macro, F1-micro, F1-weighted) P, R, ACC, LRAP, Hamming loss, Exact Match Ratio (Subset accuracy)
- Baseline:
  - Random baseline
  - Majority baseline

**Reporting:**
- Event Type Confusion matrix
- Manual error analysis: collect predictions together with plain text (and document metadata)
- Metadata on companies and sector: make master .cvs datafile containing records of each sentence: ("sentence", "y_true", "y_pred", "doc_id", "sen_idx", "company", "sector")
---

### Full run:
- Add cross-validation (if feasible in time)
- - Collect predictions in each fold
- - Collect scores in each fold
- Fine-tune Transformer model embeddings to task (isn't this the case right now?)
- Experiment with including and excluding Macroeconomic event type.

### Nice-to-Haves NOT YET: Wacht resultaat af
- Exclusion of "weakly-realized" event mentions, i.e. events with the majority of participants missing, no attributes, mainly coreferential
- Exclude event type with less than a x (x=200) attestations.
- Sentence-level IAA study. (easy to implement)
---

## TODO (Planning)
- [X] Plan experimental setup
- [X] Read up on Pytorch-Transformers API https://github.com/huggingface/pytorch-transformers
- - [ X ] Collect info on Multiclass classif.
- - [ ] Source: https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d
- - [ ] Variant source: https://github.com/kaushaltrivedi/fast-bert
- - [ ] Test the simple classification without fine-tuning approach
- - [ ] Possible settings
- - [ ] GPU acceleration
- [X] Test simple-transformers package: https://github.com/ThilinaRajapakse/simpletransformers it works!
- [ ] Collect results
- [ ] Error analysis
