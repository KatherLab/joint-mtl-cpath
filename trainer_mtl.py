

import torch.optim as optim
from utils import *
from model import EncDecTransformer
from config import TrainConfig
from pathlib import Path
from data import make_dataset_df, make_dataloaders
import sklearn
import sklearn.model_selection
from datetime import datetime
import random
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from metrics import plot_classification_metrics, plot_regression_pearson
import baseline


torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)

config = TrainConfig()

# run baseline, skips rest of the code below
if config.baseline:
    print(f"Running baseline {config.task_type} Transformer model, without balancing")
    baseline.baseline()
    exit()

# define model, optimiser and scheduler
device = config.device

if config.weight == 'autol':
    autol_lr = 1e-4
    autol_init=0.1

# create logging folder to store training weights and losses
if not os.path.exists('logging'):
    os.makedirs('logging')

print('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
      .format(config.weight.title(), config.grad_method.upper()))

all_tasks = [config.target_label_class, config.target_label_regr]
pri_tasks = [config.target_label_class]

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
dataset_df[config.target_label_class] = torch.tensor(dataset_df[config.target_label_class].map(config.label_mapping).tolist()).numpy()

# regression data adaptations
dataset_df[config.target_label_regr] = dataset_df[config.target_label_regr].astype(float)
# 5 fold cross-validation
skf = sklearn.model_selection.StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=1337)

# Get current date and time
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%d%m%Y_%H%M%S")
base_path=f"./{config.model_name}_{formatted_datetime}"
os.makedirs(base_path)
config.save_to_json(f"{base_path}/config.json")

# ITERATE OVER FOLDS
for fold, (train_index, test_index) in enumerate(skf.split(dataset_df, dataset_df[config.target_label_class])):
    model_name = f"{base_path}/fold-{fold}/fold-{fold}_{config.model_name}_{formatted_datetime}"
    
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

    model = model.to(device)
    total_epoch = config.num_epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
    # choose task weighting here
    if config.weight == 'uncert':
        logsigma = torch.tensor([-0.7] * len(all_tasks), requires_grad=True, device=device)
        params = list(model.parameters()) + [logsigma]
        logsigma_ls = np.zeros([total_epoch, len(all_tasks)], dtype=np.float32)

    if config.weight in ['dwa', 'equal']:
        T = 2.0  # temperature used in dwa
        lambda_weight = np.ones([total_epoch, len(all_tasks)])
        params = model.parameters()

    if config.weight == 'autol':
        params = model.parameters()
        autol = AutoLambda(model, device, all_tasks, pri_tasks, autol_init)
        meta_weight_ls = np.zeros([total_epoch, len(all_tasks)], dtype=np.float32)
        meta_optimizer = torch.optim.AdamW([autol.meta_weights], lr=autol_lr)
    
    # apply gradient methods
    if config.grad_method != 'none':
        rng = np.random.default_rng()
        grad_dims = []
        for mm in model.shared_modules():
            for param in mm.parameters():
                grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), len(all_tasks)).to(device)

    train_metric = TaskMetric(all_tasks, pri_tasks, config.batch_size, total_epoch)
    val_metric = TaskMetric(all_tasks, pri_tasks, config.batch_size, total_epoch)
    test_metric = TaskMetric(all_tasks, pri_tasks, config.batch_size, total_epoch)

    print(f"Fold {fold + 1}/{config.k_folds}")
    os.makedirs(f'{base_path}/fold-{fold}', exist_ok=True)
    # Split the data into training, validation, and test sets for this fold
    train_df, test_df = dataset_df.iloc[train_index], dataset_df.iloc[test_index]
    train_df, valid_df = sklearn.model_selection.train_test_split(train_df, test_size=0.2, stratify=train_df[config.target_label_class], random_state=1337)

    #scale continuous regression values 0-1
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    train_df.loc[:, config.target_label_regr] = min_max_scaler.fit_transform(train_df[config.target_label_regr].values.reshape(-1, 1))
    valid_df.loc[:, config.target_label_regr] = min_max_scaler.transform(valid_df[config.target_label_regr].values.reshape(-1, 1))
    test_df.loc[:, config.target_label_regr] = min_max_scaler.transform(test_df[config.target_label_regr].values.reshape(-1, 1))

    # Create dataloaders for the current fold
    train_loader, val_loader = make_dataloaders(
        train_bags=train_df.path.values,
        train_targets={k: v for k, v in train_df.loc[:, target_labels].items()},
        valid_bags=valid_df.path.values,
        valid_targets={k: v for k, v in valid_df.loc[:, target_labels].items()},
        instances_per_bag=config.instances_per_bag,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Create dataloaders for the current fold
    train_loader, test_loader = make_dataloaders(
        train_bags=train_df.path.values,
        train_targets={k: v for k, v in train_df.loc[:, target_labels].items()},
        valid_bags=test_df.path.values,
        valid_targets={k: v for k, v in test_df.loc[:, target_labels].items()},
        instances_per_bag=config.instances_per_bag,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    virtual_val_loader, _ = make_dataloaders(
        train_bags=train_df.path.values,
        train_targets={k: v for k, v in train_df.loc[:, target_labels].items()},
        valid_bags=test_df.path.values,
        valid_targets={k: v for k, v in test_df.loc[:, target_labels].items()},
        instances_per_bag=config.instances_per_bag,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Train and evaluate multi-task network
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    val_batch = len(val_loader)
    virutal_val_batch = len(virtual_val_loader)

    best_avg_loss_class = float("inf")
    best_avg_loss_regr = float("inf")

    ## START THE EPOCHS HERE
    for index in range(total_epoch):
        # apply Dynamic Weight Average
        if config.weight == 'dwa':
            if index == 0 or index == 1:
                lambda_weight[index, :] = 1.0
            else:
                w = []
                for i, t in enumerate(all_tasks):
                    w += [train_metric.metric[t][index - 1, 0] / train_metric.metric[t][index - 2, 0]]
                w = torch.softmax(torch.tensor(w) / T, dim=0)
                lambda_weight[index] = len(all_tasks) * w.numpy()

        # iteration for all batches
        model.train()
        train_dataset = iter(train_loader)
        if config.weight == 'autol':
            virtual_val_dataset = iter(virtual_val_loader)

        for k in tqdm(range(train_batch), total=(train_batch), leave=False):
            train_data, train_coords, train_target = next(train_dataset)
            class_labels_key, regr_labels_key = list(train_target.keys())
            class_train_target, regr_train_target = list(train_target.values())
            train_data, class_train_target, regr_train_target  = train_data.to(device), class_train_target.type(torch.LongTensor).to(device), regr_train_target.type(torch.float32).to(device)
            train_target=[class_train_target, regr_train_target]

            # update meta-weights with Auto-Lambda
            if config.weight == 'autol':
                virt_val_data, virt_val_coords, virt_val_target = next(virtual_val_dataset)
                class_virt_val_target, regr_virt_val_target = list(virt_val_target.values())
                virt_val_data, class_virt_val_target, regr_virt_val_target  = virt_val_data.to(device), class_virt_val_target.type(torch.LongTensor).to(device), regr_virt_val_target.type(torch.float32).to(device)
                virt_val_target=[class_virt_val_target, regr_virt_val_target]
                meta_optimizer.zero_grad()
                autol.unrolled_backward(train_data, train_target, virt_val_data, virt_val_target,
                                        scheduler.get_last_lr()[0], optimizer)
                meta_optimizer.step()

            # update multi-task network parameters with task weights
            optimizer.zero_grad()
            train_pred = model(train_data)

            #returns [loss_class, loss_regr]
            train_loss = get_loss(train_pred, train_target)

            train_loss_tmp = [0] * len(all_tasks)

            if config.weight in ['equal', 'dwa']:
                train_loss_tmp = [w * train_loss[i] for i, w in enumerate(lambda_weight[index])]

            if config.weight == 'uncert':
                train_loss_tmp = [1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma)]

            if config.weight == 'autol':
                train_loss_tmp = [w * train_loss[i] for i, w in enumerate(autol.meta_weights)]

            loss = sum(train_loss_tmp)

            if config.grad_method == 'none':
                loss.backward()
                optimizer.step()

            # gradient-based methods applied here:
            elif config.grad_method == "graddrop":
                for i in range(len(all_tasks)):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(model, grads, grad_dims, i)
                    model.zero_grad_shared_modules()
                g = graddrop(grads)
                overwrite_grad(model, g, grad_dims, len(all_tasks))
                optimizer.step()

            elif config.grad_method == "pcgrad":
                for i in range(len(all_tasks)):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(model, grads, grad_dims, i)
                    model.zero_grad_shared_modules()
                g = pcgrad(grads, rng, len(all_tasks))
                overwrite_grad(model, g, grad_dims, len(all_tasks))
                optimizer.step()

            elif config.grad_method == "cagrad":
                for i in range(len(all_tasks)):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(model, grads, grad_dims, i)
                    model.zero_grad_shared_modules()
                g = cagrad(grads, len(all_tasks), 0.4, rescale=1)
                overwrite_grad(model, g, grad_dims, len(all_tasks))
                optimizer.step()
            
            scheduler.step()
            train_metric.update_metric(train_pred, train_target, train_loss, all_tasks)

        train_metric.reset()
        # Epoch evaluation test data
        model.eval()
        with torch.no_grad():
            valid_loss_class = 0.0
            valid_loss_regr = 0.0
            y_true_class = []
            y_pred_class = []
            y_true_regr = []
            y_pred_regr = []
            val_dataset = iter(val_loader)
            for k in tqdm(range(val_batch), total=(val_batch), leave=False):
                val_data, val_coords, val_target = next(val_dataset)
                class_val_target, regr_val_target = list(val_target.values())
                val_data, class_val_target, regr_val_target  = val_data.to(device), class_val_target.type(torch.LongTensor).to(device), regr_val_target.type(torch.float32).to(device)
                val_target=[class_val_target, regr_val_target]
                
                # val_pred = [logits_classification, logits_regression]
                val_pred = model(val_data)
                # val_loss = [loss_class, loss_regr]
                val_loss = get_loss(val_pred, val_target)
                val_metric.update_metric(val_pred, val_target, val_loss, all_tasks)

                y_true_class.extend(class_val_target.cpu().numpy())
                y_pred_class.extend(val_pred[0][:, 1].cpu().numpy()) 
                y_true_regr.extend(regr_val_target.cpu().numpy())
                y_pred_regr.extend(val_pred[1].cpu().numpy())
                valid_loss_class += val_loss[0].item()
                valid_loss_regr += val_loss[1].item()
        
        val_metric.reset()
        avg_valid_loss_class = valid_loss_class / len(val_loader)
        avg_valid_loss_regr = valid_loss_regr / len(val_loader)
        metric_class = roc_auc_score(y_true_class, y_pred_class)
        print(f"AUROC (Classification) on Validation: {metric_class:.4f}")
        # auroc_class_list.append(metric)
        print(f"Valid Cross-entropy Loss: {avg_valid_loss_class:.4f}")
        save_criteria_class = (avg_valid_loss_class < best_avg_loss_class)

        # Calculate Pearson's r for regression
        metric_regr, _ = pearsonr(y_true_regr, y_pred_regr)
        print(f"Pearson's r (Regression) on Validation: {metric_regr:.4f}")
        # pearson_r_regr_list.append(metric)
        print(f"Valid MSE Loss: {avg_valid_loss_regr:.4f}")
        save_criteria_regr = (avg_valid_loss_regr < best_avg_loss_regr)

        #NOTE: optimized for lowest CE loss
        if (save_criteria_class):
            print(f"==== New best model found in {index+1}, loss = {avg_valid_loss_class} < {best_avg_loss_class} ===")
            best_avg_loss_class = avg_valid_loss_class
            best_avg_loss_regr = avg_valid_loss_regr
            best_epoch = index+1
            torch.save(model.state_dict(), f'{model_name}.pth')

        if abs(index - best_epoch) == config.early_stopping:
                print(f"Early stopping triggered, no improvement since epoch {best_epoch}...")
                break                             

    ### TESTING
    if best_epoch != -1:
        print(f"Loading model from epoch {best_epoch} for inference...")
        model.load_state_dict(torch.load(f'{model_name}.pth'))
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_loss_class = 0.0
        test_loss_regr = 0.0
        y_true_class = []
        y_pred_class = []
        y_true_regr = []
        y_pred_regr = []
        test_dataset = iter(test_loader)
        for k in tqdm(range(test_batch), total=(test_batch), leave=False):
            test_data, test_coords, test_target = next(test_dataset)

            class_test_target, regr_test_target = list(test_target.values())
            test_data, class_test_target, regr_test_target  = test_data.to(device), class_test_target.type(torch.LongTensor).to(device), regr_test_target.type(torch.float32).to(device)
            test_target=[class_test_target, regr_test_target]

            test_pred = model(test_data)
            test_loss = get_loss(test_pred, test_target)

            test_metric.update_metric(test_pred, test_target, test_loss, all_tasks)
            y_true_class.extend(class_test_target.cpu().numpy())
            y_pred_class.extend(torch.softmax(test_pred[0], dim=1).cpu().numpy())
            y_true_regr.extend(regr_test_target.cpu().numpy())
            y_pred_regr.extend(test_pred[1].cpu().numpy())
            test_loss_class += test_loss[0].item()
            test_loss_regr += test_loss[1].item()

    avg_test_loss_class = test_loss_class / len(test_loader)
    avg_test_loss_regr = test_loss_regr / len(test_loader)

    print(f"Test Loss: Crossentropy={avg_test_loss_class:.4f}, MSE={avg_test_loss_regr:.4f}")
    softmax_class_logit=np.array(y_pred_class)
    auroc_class = roc_auc_score(y_true_class, softmax_class_logit[:, 1].tolist()) #first index of softmaxed logits
    print(f"AUROC (Classification) on Test: {auroc_class:.4f}")

    pearson_r_regr, _ = pearsonr(y_true_regr, y_pred_regr)
    print(f"Pearson's r (Regression) on Test: {pearson_r_regr:.4f}")
    # plot results 
    data = {'y_true_regr': y_true_regr, 'y_pred_regr': y_pred_regr}
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.regplot(x='y_true_regr', y='y_pred_regr', data=df)
    plt.title(f'Regression Correlation Plot\nPearson\'s r: {pearson_r_regr:.4f}', fontsize=16)
    plt.savefig(f'{model_name}-regr_plot.png')

    test_df_class = test_loader.dataset.targets[class_labels_key]
    test_df_class = pd.DataFrame(test_df_class).reset_index(col_level=0, col_fill='')
    test_df_class["class_logits_neg"] = softmax_class_logit[:, 0].tolist()
    test_df_class["class_logits_pos"] = softmax_class_logit[:, 1].tolist()
    test_df_class["class_pred"] = np.argmax(y_pred_class, axis=1)

    test_df_regr = test_loader.dataset.targets[regr_labels_key]
    test_df_regr = pd.DataFrame(test_df_regr).reset_index(col_level=0, col_fill='')
    test_df_regr["regr_logits"] = y_pred_regr

    preds_df = test_df_class.merge(test_df_regr, on='PATIENT', how='left')
    preds_df.to_csv(f'{model_name}-patient_preds.csv')
    
    test_metric.reset()
    
    if config.weight == 'autol':
        meta_weight_ls[index] = autol.meta_weights.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': meta_weight_ls}

        print(get_weight_str(meta_weight_ls[index], all_tasks))

    if config.weight in ['dwa', 'equal']:
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': lambda_weight}

        print(get_weight_str(lambda_weight[index], all_tasks))

    if config.weight == 'uncert':
        logsigma_ls[index] = logsigma.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': logsigma_ls}

        print(get_weight_str(1 / (2 * np.exp(logsigma_ls[index])), all_tasks))


print("Training complete, plotting final metrics...")
preds_files = glob.glob(f"{base_path}/fold-*/*patient_preds.csv")
preds_df = pd.DataFrame(pd.concat(map(pd.read_csv, preds_files)))

if config.task_type == 'classification':
    plot_classification_metrics(preds_files, config, base_path)

elif config.task_type == 'regression':
    y_true_regr = preds_df[config.target_label_regr].values
    y_pred_regr = preds_df["regr_logits"].values
    plot_regression_pearson(y_true_regr, y_pred_regr, base_path)

elif config.task_type == 'joint':
    y_true_class = preds_df[config.target_label_class].values
    y_pred_class = preds_df["class_logits_pos"].values
    y_true_regr = preds_df[config.target_label_regr].values
    y_pred_regr = preds_df["regr_logits"].values
    plot_classification_metrics(preds_files, config, base_path)
    plot_regression_pearson(y_true_regr, y_pred_regr, base_path)
    plot_classification_metrics(preds_files, config, base_path, is_joint=True)

preds_df.to_csv(f"{base_path}/full_preds_df.csv")