from config import TestConfig
from pathlib import Path
import torch
import os
import numpy as np
import glob
from helpers.metrics import plot_classification_metrics
from helpers.evaluate import evaluate
from helpers.model import EncDecTransformer
from helpers.data import make_dataloaders, make_dataset_df

torch.manual_seed(1337)
config = TestConfig()
target_labels=[config.target_label_class, config.target_label_regr]
dataset_df = make_dataset_df(
    clini_table=Path(config.clini_table),
    slide_table=Path(config.slide_table),
    feature_dir=Path(config.feature_dir),
    target_labels=target_labels
)

#only want 100% overlap between targets to enable batch_size=1
dataset_df = dataset_df.dropna()

#classification data adaptations
dataset_df[config.target_label_class] = dataset_df[config.target_label_class].map(config.label_mapping)

# regression data adaptations
if config.dummy_regr:
    dataset_df[config.target_label_regr] = np.ones(len(dataset_df[config.target_label_class]))
else:
    dataset_df[config.target_label_regr] = dataset_df[config.target_label_regr].astype(float)

model = EncDecTransformer(
            d_features=768,
            target_label_class=config.target_label_class,
            target_label_regr=config.target_label_regr,
            d_model=config.d_model,
            num_encoder_heads=config.num_encoder_heads,
            num_decoder_heads=config.num_decoder_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            positional_encoding=config.positional_encoding,
        )

model=model.to(config.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
class_loss = torch.nn.CrossEntropyLoss()
regr_loss = torch.nn.MSELoss()
criterion = {'classification':  class_loss,
            'regression':      regr_loss}

model_path = config.model_path

base_path=f"./{config.model_name}"
os.makedirs(base_path, exist_ok=True)

_, test_dl = make_dataloaders(
        train_bags=dataset_df.path.values,
        train_targets={k: v for k, v in dataset_df.loc[:, target_labels].items()},
        valid_bags=dataset_df.path.values,
        valid_targets={k: v for k, v in dataset_df.loc[:, target_labels].items()},
        instances_per_bag=config.instances_per_bag,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

final_df = []
for fold, pth_file in enumerate(sorted(glob.glob(f"{model_path}/fold-*/*pth"))):
    evaluate(model, test_dl, criterion, config.model_name, pth_file, fold, config)

preds_path = glob.glob(f'{config.model_name}/fold-*/*csv')
plot_classification_metrics(preds_path, config, base_path)