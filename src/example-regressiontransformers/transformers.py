import logging
from math import sqrt

from pytorch_transformers import *
import torch
from torch import nn

logger = logging.getLogger(__name__)

MODELS = {
    "bert": (BertModel, BertTokenizer, "bert-base-uncased"),
    "distilbert": (DistilBertModel, DistilBertTokenizer, "distilbert-base-uncased"),
    "gpt": (OpenAIGPTModel, OpenAIGPTTokenizer, "openai-gpt"),
    "gpt2": (GPT2Model, GPT2Tokenizer, "gpt2"),
    "transfoxl": (TransfoXLModel, TransfoXLTokenizer, "transfo-xl-wt103"),
    "xlnet": (XLNetModel, XLNetTokenizer, "xlnet-base-cased"),
    "xlm": (XLMModel, XLMTokenizer, "xlm-mlm-enfr-1024"),
    "roberta": (RobertaModel, RobertaTokenizer, "roberta-base"),
}


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-transformers/blob/master/modeling.py
    """
    return 0.5 * x * (1.0 + torch.erf(x / sqrt(2.0)))


class TransformerRegressor(nn.Module):
    """ Main transformer class that can initialize any kind of transformer in `MODELS`. """

    def __init__(self, config):
        super(TransformerRegressor, self).__init__()

        weights = (
            MODELS[config["name"]][2]
            if config["weights"] == "default"
            else config["weights"]
        )
        self.base_model = MODELS[config["name"]][0].from_pretrained(weights)

        # Freeze parts of pretrained model
        # config['freeze'] can be "all" to freeze all layers,
        # or any number of prefixes, e.g. ['embeddings', 'encoder']
        if config["freeze"] is not None:
            for name, param in self.base_model.named_parameters():
                if config["freeze"] == "all" or name.startswith(
                    tuple(config["freeze"])
                ):
                    param.requires_grad = False
                    logging.info(f"Froze layer {name}...")

        dim = self.base_model.config.hidden_size

        self.pre_classifier = nn.Linear(dim, dim)
        if config["activation"] == "gelu":
            self.activation = gelu
        elif config["activation"] == "relu":
            self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(dim, 1)

        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights.
        Only works for models that have a `.initializer_range` in their config
        Borrowed from pytorch_transformers
        """
        try:
            if isinstance(module, nn.Embedding):
                if module.weight.requires_grad:
                    module.weight.data.normal_(
                        mean=0.0, std=self.base_model.config.initializer_range
                    )
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(
                    mean=0.0, std=self.base_model.config.initializer_range
                )
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        except AttributeError:
            pass

    def forward(self, input_ids, labels, attention_mask=None):
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = out[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = self.activation(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        return logits.view(-1)


class TransformerTokenizer:
    """ Main tokenizer class that can initialize any kind of transformer in `MODELS`. """

    def __init__(self, config):
        self.weights = (
            MODELS[config["name"]][2]
            if config["weights"] == "default"
            else config["weights"]
        )
        self.tokenizer = MODELS[config["name"]][1].from_pretrained(self.weights)

    def __call__(self, text):
        all_input_ids = []
        all_input_mask = []

        for sentence in text:
            input_ids = self.tokenizer.encode(sentence, add_special_tokens=True)
            input_mask = [1] * len(input_ids)

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)

        max_length = max([len(ids) for ids in all_input_ids])
        # Zero-pad up to max batch length.
        for i in range(len(all_input_ids)):
            input_ids = all_input_ids[i]
            input_mask = all_input_mask[i]
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_mask.append(0)

        all_input_ids = torch.LongTensor(all_input_ids)
        all_input_mask = torch.FloatTensor(all_input_mask)

        return all_input_ids, all_input_mask

    def __repr__(self):
        return (
            f"<TransformerTokenizer tokenizer={self.tokenizer} weights={self.weights}>"
        )
