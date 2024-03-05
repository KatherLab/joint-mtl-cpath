# %%
from typing import Union, Tuple, List
import torch
from torch import nn

__all__ = [
    "EncDecTransformer"
]


class EncDecTransformer(nn.Module):
    """An encoder decoder architecture for multilabel classification tasks

    This architecture is a modified version of the one found in [Attention Is
    All You Need][1]: First, we project the features into a lower-dimensional
    feature space, to prevent the transformer architecture's complexity from
    exploding for high-dimensional features.  We add sinusodial [positional
    encodings][1].  We then encode these projected input tokens using a
    transformer encoder stack.  Next, we decode these tokens using a set of
    class tokens, one per output label.  Finally, we forward each of the decoded
    tokens through a fully connected layer to get a label-wise prediction.

                  PE1
                   |
             +--+  v   +---+
        t1 --|FC|--+-->|   |--+
         .   +--+      | E |  |
         .             | x |  |
         .   +--+      | n |  |
        tn --|FC|--+-->|   |--+
             +--+  ^   +---+  |
                   |          |
                  PEn         v
                            +---+   +---+
        c1 ---------------->|   |-->|FC1|--> s1
         .                  | D |   +---+     .
         .                  | x |             .
         .                  | k |   +---+     .
        ck ---------------->|   |-->|FCk|--> sk
                            +---+   +---+

    We opted for this architecture instead of a more traditional [Vision
    Transformer][2] to improve performance for multi-label predictions with many
    labels.  Our experiments have shown that adding too many class tokens to a
    vision transformer decreases its performance, as the same weights have to
    both process the tiles' information and the class token's processing.  Using
    an encoder-decoder architecture alleviates these issues, as the data-flow of
    the class tokens is completely independent of the encoding of the tiles.
    Furthermore, analysis has shown that there is almost no interaction between
    the different classes in the decoder.  While this points to the decoder
    being more powerful than needed in practice, this also means that each
    label's prediction is mostly independent of the others.  As a consequence,
    noisy labels will not negatively impact the accuracy of non-noisy ones.

    In our experiments so far we did not see any improvement by adding
    positional encodings.  We tried

     1. [Sinusodal encodings][1]
     2. Adding absolute positions to the feature vector, scaled down so the
        maximum value in the training dataset is 1.

    Since neither reduced performance and the author percieves the first one to
    be more elegant (as the magnitude of the positional encodings is bounded),
    we opted to keep the positional encoding regardless in the hopes of it
    improving performance on future tasks.

    The architecture _differs_ from the one descibed in [Attention Is All You
    Need][1] as follows:

     1. There is an initial projection stage to reduce the dimension of the
        feature vectors and allow us to use the transformer with arbitrary
        features.
     2. Instead of the language translation task described in [Attention Is All
        You Need][1], where the tokens of the words translated so far are used
        to predict the next word in the sequence, we use a set of fixed, learned
        class tokens in conjunction with equally as many independent fully
        connected layers to predict multiple labels at once.

    [1]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need"
    [2]: https://arxiv.org/abs/2010.11929
        "An Image is Worth 16x16 Words:
         Transformers for Image Recognition at Scale"
    """

    def __init__(
        self,
        d_features: int,
        target_label_class: str,
        target_label_regr: str,
        *,
        d_model: int = 256,
        num_encoder_heads: int = 6,
        num_decoder_heads: int = 6,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 768,
        positional_encoding: bool = True,
    ) -> None:
        super().__init__()

        self.projector = nn.Sequential(nn.Linear(d_features, d_model), nn.ReLU())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_encoder_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            # norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.target_labels = [target_label_class, target_label_regr]

        # One class token per output label
        self.class_tokens = nn.ParameterDict(
            {
                target_label: torch.rand(d_model)
                for target_label in self.target_labels
            }
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_decoder_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            # norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.head_class = nn.ModuleDict(
            {
                target_label_class: nn.Linear(
                    in_features=d_model, out_features=2 #binary classification output
                )
            }
        )

        self.head_regr = nn.ModuleDict(
            {
                target_label_regr: nn.Linear(
                    in_features=d_model, out_features=1 #regression output
                )
            }
        )

        self.positional_encoding = positional_encoding

    def forward(
        self,
        tile_tokens: torch.Tensor,
        task_type: str = "joint",
        baseline: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, _, _ = tile_tokens.shape
        target_label_classification = self.target_labels[0]
        target_label_regression = self.target_labels[1]
        if baseline:
            logits_classification, logits_regression = get_baseline_logits(self, tile_tokens, task_type)
        else: 
            tile_tokens = self.projector(tile_tokens)  # shape: [bs, seq_len, d_model]
            tile_tokens = self.transformer_encoder(tile_tokens)
            class_tokens = torch.stack(
                [self.class_tokens[t] for t in self.target_labels]  # class idx 0, regr idx 1
            ).expand(batch_size, -1, -1)
            class_tokens = (
                self.transformer_decoder(tgt=class_tokens, memory=tile_tokens)
            ).permute(1, 0, 2)  # Permute to [target, batch, d_model]
            logits_classification = self.head_class[target_label_classification](class_tokens[0])
            logits_regression = self.head_regr[target_label_regression](class_tokens[1]).squeeze(dim=1)

        return [logits_classification, logits_regression]


    def shared_modules(self):
        return [self.projector,
                self.transformer_encoder,
                self.transformer_decoder]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()


def get_baseline_logits(self, tile_tokens, task_type):
    batch_size, _, _ = tile_tokens.shape
    target_label_classification = self.target_labels[0]
    target_label_regression = self.target_labels[1]
    tile_tokens = self.projector(tile_tokens)  # shape: [bs, seq_len, d_model]
    tile_tokens = self.transformer_encoder(tile_tokens)

    if task_type == "classification":
        class_tokens = self.class_tokens[target_label_classification].expand(batch_size, -1).unsqueeze(-2)
        class_tokens = (
            self.transformer_decoder(tgt=class_tokens, memory=tile_tokens)
        ).permute(1, 0, 2).squeeze(0)  # Permute to [target, batch, d_model]
        logits_classification = self.head_class[target_label_classification](class_tokens)
        logits_regression = None

    elif task_type == "regression":
        class_tokens = self.class_tokens[target_label_regression].unsqueeze(0).expand(batch_size, -1, -1)
        class_tokens = (
            self.transformer_decoder(tgt=class_tokens, memory=tile_tokens)
        ).permute(1, 0, 2).squeeze(0)  # Permute to [target, batch, d_model]
        logits_classification = None
        logits_regression = self.head_regr[target_label_regression](class_tokens).squeeze(dim=1)

    elif task_type == "joint":
        class_tokens = torch.stack(
            [self.class_tokens[t] for t in self.target_labels]  # class idx 0, regr idx 1
        ).expand(batch_size, -1, -1)
        class_tokens = (
            self.transformer_decoder(tgt=class_tokens, memory=tile_tokens)
        ).permute(1, 0, 2)  # Permute to [target, batch, d_model]
        
        logits_classification = self.head_class[target_label_classification](class_tokens[0])
        logits_regression = self.head_regr[target_label_regression](class_tokens[1]).squeeze(dim=1)
        
    else:
        raise ValueError("Invalid task_type. Use 'classification', 'regression', or 'joint'.")

    return logits_classification, logits_regression
# %%