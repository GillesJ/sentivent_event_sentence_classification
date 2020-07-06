# SENTiVENT Event Sentence Classification
Sentence-level event classification experiments for the SENTiVENT Event dataset.
Baseline experiments meant for SENTiVENT Event Data manuscript submission.

## Usage:
Train-test-time output: Each train-test set and model is written to its model dir
Reporting: Load each folder with predictions, parse them > summarize and rank


## Install
This depends on the package [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers).

`pipenv install --python 3.7.5 simpletransformers torch pandas`

## Available multilabel models:
MODEL_CLASSES = {
            'bert':       (BertConfig, BertForMultiLabelSequenceClassification, BertTokenizer),
            'roberta':    (RobertaConfig, RobertaForMultiLabelSequenceClassification, RobertaTokenizer),
            'xlnet':      (XLNetConfig, XLNetForMultiLabelSequenceClassification, XLNetTokenizer),
            'xlm':        (XLMConfig, XLMForMultiLabelSequenceClassification, XLMTokenizer),
            'distilbert': (DistilBertConfig, DistilBertForMultiLabelSequenceClassification, DistilBertTokenizer),
            'albert':     (AlbertConfig, AlbertForMultiLabelSequenceClassification, AlbertTokenizer)
}

##Tensorboard for checking loss and accuracy
You need to install Tensorflow to use Tensorboard on your client (simpletransformers actually uses the PyTorch-fork tensorboardx for its tensorboard output and does not depend on TF.):
First install a python version compatible with TF (latest=3.7.5 as of writing):
`pyenv install 3.7.5`
Now install TensorFlow
`pipx install --python /home/gilles/.pyenv/versions/3.7.5/bin/python tensorflow`
Now run the Tensorboard command on the run dir which was created during training:
`tensorboard`

## Utility commands
- Remove large output files: checkpoints and epoch binaries.
1. Change to experiment dir: `cd RUNDIR`
2. Check what you are removing `find . \( -name "epoch*" -or -name "checkpoint*" \) -exec echo "{}" \;`
3. Remove it `find . \( -name "epoch*" -or -name "checkpoint*" \) -exec rm -r "{}" \; -prune`

# Experiment results
###Roberta-large:
- 6 epochs:
Crossvalidation score: {'eval_loss': 0.00614539818296748, 'LRAP': 0.9972541923792937}
Holdout score: {'LRAP': 0.8745366615430941, 'eval_loss': 0.12979916081978723}

- 8 epochs: 2020-01-06_14-41-59-roberta-large: BEST
- 2020-01-07_12-14-25-roberta-large: 4 epochs WORST
 {"model_type": "roberta", "model_name": "roberta-large", "train_args": {"reprocess_input_data": true, "overwrite_output_dir": true, "num_train_epochs": 4, "n_gpu": 1}}
    holdout {'LRAP': 0.4904404584329125, 'eval_loss': 0.22251000754780823}	../models/2020-01-07_12-14-25-roberta-large/holdout
    all_fold_mean {'eval_loss': 0.1659443878521259, 'LRAP': 0.6338597185519774}
    -> way worse than 8 epochs (current best) DELETED
- Roberta large 16 epochs: {"LRAP": 0.8505341862940574, "eval_loss": 0.215741140900978}: 8 is better on holdout
- Roberta large 24 epochs: 

###Albert-xxlarge-v2:
- 4 epochs: TOO LITTLE {'LRAP': 0.4919080370097837, 'eval_loss': 0.21860242708698735}	../models/2019-12-29_21-07-10-albert-xxlarge-v2/holdouttest_predictions.tsv	holdouttest
- 8 epochs: 11	{'LRAP': 0.6902371653891292, 'eval_loss': 0.3356399894538489}	../models/2019-12-26_22-35-33-albert-xxlarge-v2/holdouttest_predictions.tsv	holdouttest DELETED DIR
- 16 epochs: 11	{'LRAP': 0.7661070806622629, 'eval_loss': 0.40053606477494424}	../models/2019-12-29_21-06-28-albert-xxlarge-v2/holdouttest_predictions.tsv	holdouttest
- 32 epochs 2020-01-02_12-24-34-albert-xxlarge-v2: {"model_type": "albert", "model_name": "albert-xxlarge-v2", "train_args": {"reprocess_input_data": true, "overwrite_output_dir": true, "num_train_epochs": 32, "n_gpu": 1, "evaluate_during_training": false}} {'LRAP': 0.5816406867781367, 'eval_loss': 0.5013814651212849} HOLDOUT = BAD (folddirs removed)

DistilRoberta-base:
-  4 epochs: holdout	{'LRAP': 0.8399741222178314, 'eval_loss': 0.123979330349427}	../models/2020-01-07_12-18-42-distilroberta-base/holdout
all_fold_mean	{'eval_loss': 0.007836045015857104, 'LRAP': 0.9948021266208709}	PRETTY GOOD
