import torch
import numpy as np
import torch
import sklearn.preprocessing
import sklearn.model_selection
from data import make_dataloaders, make_dataset_df
from model import EncDecTransformer
from config import TrainConfig
from metrics import plot_regression_pearson, plot_classification_metrics
from datetime import datetime
from pathlib import Path
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import seaborn as sns


def baseline():
    # Set up configurations
    torch.manual_seed(1337)
    config = TrainConfig()
    target_labels=[config.target_label_class, config.target_label_regr]
    # Set up data loaders
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
    dataset_df[config.target_label_regr] = dataset_df[config.target_label_regr].astype(float)

    # 5 fold cross-validation
    skf = sklearn.model_selection.StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=1337)

    # Get current date and time
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%d%m%Y_%H%M%S")
    model_name="baseline_"+config.task_type+"_"+config.target_label_class+"_"+config.target_label_regr+"_mtl-cpath"
    base_path=f"./{model_name}_{formatted_datetime}"
    os.makedirs(base_path)
    config.save_to_json(f"{base_path}/config.json")

    # Iterate over folds
    for fold, (train_index, test_index) in enumerate(skf.split(dataset_df, dataset_df[config.target_label_class])):
        print(f"Fold {fold + 1}/{config.k_folds}")
        # Split the data into training, validation, and test sets for this fold
        train_df, test_df = dataset_df.iloc[train_index], dataset_df.iloc[test_index]
        train_df, valid_df = sklearn.model_selection.train_test_split(train_df, test_size=0.2, stratify=train_df[config.target_label_class], random_state=1337)

        #scale continuous regression values 0-1
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        train_df.loc[:, config.target_label_regr] = min_max_scaler.fit_transform(train_df[config.target_label_regr].values.reshape(-1, 1))
        valid_df.loc[:, config.target_label_regr] = min_max_scaler.transform(valid_df[config.target_label_regr].values.reshape(-1, 1))
        test_df.loc[:, config.target_label_regr] = min_max_scaler.transform(test_df[config.target_label_regr].values.reshape(-1, 1))

        # Create dataloaders for the current fold
        train_dl, valid_dl = make_dataloaders(
            train_bags=train_df.path.values,
            train_targets={k: v for k, v in train_df.loc[:, target_labels].items()},
            valid_bags=valid_df.path.values,
            valid_targets={k: v for k, v in valid_df.loc[:, target_labels].items()},
            instances_per_bag=config.instances_per_bag,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        train_dl, test_dl = make_dataloaders(
            train_bags=train_df.path.values,
            train_targets={k: v for k, v in train_df.loc[:, target_labels].items()},
            valid_bags=test_df.path.values,
            valid_targets={k: v for k, v in test_df.loc[:, target_labels].items()},
            instances_per_bag=config.instances_per_bag,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        example_bags, _, _ = next(iter(train_dl))
        d_features = example_bags.size(-1)

        # Create model, optimizer, criterion
        model = EncDecTransformer(
                    d_features=d_features,
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

        #ensure same device is used
        model=model.to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        class_loss = torch.nn.CrossEntropyLoss()
        regr_loss = torch.nn.MSELoss()
        criterion = {'classification':  class_loss,
                    'regression':      regr_loss}

        os.makedirs(f'{base_path}/fold-{fold}', exist_ok=True)

        # Training loop
        train_baseline(model, train_dl, valid_dl, test_dl, optimizer, criterion, f"{base_path}/fold-{fold}/fold-{fold}_{model_name}_{formatted_datetime}", config)

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


def train_baseline(model, train_loader, valid_loader, test_loader, 
          optimizer, criterion, model_name, config):
    criteron_class = criterion['classification']
    criterion_regr = criterion['regression']
    
    best_metric = float('-inf')  # Initialize with a high value
    best_avg_loss_class = float("inf")
    best_avg_loss_regr = float("inf")
    best_epoch = -1
    n_samples=len(train_loader)
    scaling_factors_class = []
    scaling_factors_regr = []
    auroc_class_list = []
    pearson_r_regr_list = []
    
    for epoch in range(config.num_epochs):
        model.train()  # Set the model to training mode
        running_loss_class = 0.0
        running_loss_regr = 0.0

        # Training Step
        for i, (feats, coords, labels) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            class_labels_key, regr_labels_key = list(labels.keys())
            class_labels = torch.Tensor(labels[class_labels_key])
            regr_labels = torch.Tensor(labels[regr_labels_key]).to(torch.float32)
            feats, coords, class_labels, regr_labels = feats.to(config.device), \
                                                coords.to(config.device), \
                                                class_labels.to(config.device), \
                                                regr_labels.to(config.device)
            optimizer.zero_grad()  # Zero the gradients
            # Forward pass
            logits_classification, logits_regression = model(tile_tokens=feats, task_type=config.task_type, baseline=True)
            loss_class=torch.tensor(0.0)
            loss_regr=torch.tensor(0.0)
            # Compute the classification losses
            if logits_classification is not None:
                loss_class = criteron_class(logits_classification, class_labels).to(torch.float32)
                loss_class.backward(retain_graph=True) #backward pass regr loss, retain graph for another loss
            # Compute the regression losses
            if logits_regression is not None:
                loss_regr = criterion_regr(logits_regression, regr_labels).to(torch.float32)
                loss_regr.backward() #backward pass class loss


            #total_loss = (loss_class + loss_regr )/2
            #total_loss.backward()
            optimizer.step()  # Update weights

            running_loss_class += loss_class.item()
            running_loss_regr += loss_regr.item()

        scaling_factors_class.append(running_loss_class/ n_samples)
        scaling_factors_regr.append(running_loss_regr/ n_samples)

        print(f"[Epoch {epoch + 1}] \
                Cross-entropy loss: {running_loss_class / n_samples:.4f}, \
                MSE loss: {running_loss_regr / n_samples:.4f}")
        running_loss_class = 0.0
        running_loss_regr = 0.0
        
        # Validation Step
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
                    valid_loss_class = 0.0
                    valid_loss_regr = 0.0
                    y_true_class = []
                    y_pred_class = []
                    y_true_regr = []
                    y_pred_regr = []
                    for i, (feats, coords, labels) in tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False):
                        class_labels_key, regr_labels_key = list(labels.keys())
                        class_labels = torch.Tensor(labels[class_labels_key])
                        regr_labels = torch.Tensor(labels[regr_labels_key]).to(torch.float32)
                        feats, coords, class_labels, regr_labels = feats.to(config.device), \
                                                            coords.to(config.device), \
                                                            class_labels.to(config.device), \
                                                            regr_labels.to(config.device)

                        # Forward pass
                        logits_classification, logits_regression = model(tile_tokens=feats, task_type=config.task_type, baseline=True)
                        loss_class=torch.tensor(0.0)
                        loss_regr=torch.tensor(0.0)
                        # Compute the classification losses
                        if logits_classification is not None:
                            loss_class = criteron_class(logits_classification, class_labels).to(torch.float32)
                            y_true_class.extend(class_labels.cpu().numpy())
                            y_pred_class.extend(logits_classification[:, 1].cpu().numpy())  # Assuming 2-class classification
                        # Compute the regression losses
                        if logits_regression is not None:
                            loss_regr = criterion_regr(logits_regression, regr_labels).to(torch.float32)
                            y_true_regr.extend(regr_labels.cpu().numpy())
                            y_pred_regr.extend(logits_regression.cpu().numpy())

                        valid_loss_class += loss_class.item()
                        valid_loss_regr += loss_regr.item()

                    avg_valid_loss_class = valid_loss_class / len(valid_loader)
                    avg_valid_loss_regr = valid_loss_regr / len(valid_loader)

                    print(f"Validation Loss: Crossentropy={avg_valid_loss_class:.4f}, MSE={avg_valid_loss_regr:.4f}")
                    save_metrics=0.0
                    save_criteria_regr=False
                    save_criteria_class=False
                    # Calculate AUROC for classification
                    if logits_classification is not None:
                        metric = roc_auc_score(y_true_class, y_pred_class)
                        print(f"AUROC (Classification) on Validation: {metric:.4f}")
                        auroc_class_list.append(metric)
                        print(f"Valid Cross-entropy Loss: {avg_valid_loss_class:.4f}")
                        save_criteria_class = (avg_valid_loss_class < best_avg_loss_class)
                        #save_metrics = save_metrics + metric
                    # Calculate Pearson's r for regression
                    if logits_regression is not None:
                        metric, _ = pearsonr(y_true_regr, y_pred_regr)
                        print(f"Pearson's r (Regression) on Validation: {metric:.4f}")
                        pearson_r_regr_list.append(metric)
                        print(f"Valid MSE Loss: {avg_valid_loss_regr:.4f}")
                        save_criteria_regr = (avg_valid_loss_regr < best_avg_loss_regr)
                        #save_metrics = save_metrics + metric                             

                    #NOTE: optimized for lowest CE loss
                    if (save_criteria_class): # avg_valid_loss_class < best_metric: 
                        # best_metric = save_metrics
                        print(f"==== New best model found in {epoch+1}, loss = {avg_valid_loss_class} < {best_avg_loss_class} ===")
                        best_avg_loss_class = avg_valid_loss_class
                        best_avg_loss_regr = avg_valid_loss_regr
                        best_epoch = epoch+1
                        torch.save(model.state_dict(), f'{model_name}.pth')

                    if abs(epoch - best_epoch) == config.early_stopping:
                         print(f"Early stopping triggered, no improvement since epoch {best_epoch}...")
                         break
    
    ##################### Plotting loss / metrics over epochs
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    epochs = range(1, epoch+2)

    if logits_classification is not None:
        # Plot scaling factors
        ax1.plot(epochs, scaling_factors_class, label='Scaled Loss (Classification)', color='blue')
        ax2.plot(epochs, auroc_class_list, label='AUROC (Classification)', linestyle='--', color='red')


    if logits_regression is not None:
    # Create a second y-axis for AUROCs and Pearson's r
        ax1.plot(epochs, scaling_factors_regr, label='Scaled Loss (Regression)', color='green')
        ax2.plot(epochs, pearson_r_regr_list, label='Pearson\'s r (Regression)', linestyle='--', color='orange')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Scaled Loss', color='black')
    ax1.tick_params('y', colors='black')
    ax2.set_ylabel('AUROC / Pearson\'s r', color='black')
    ax2.tick_params('y', colors='black')

    # Add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Metrics and Scaled Losses After Each Epoch')
    plt.grid(True)
    fig.savefig(f'{model_name}-metrics.png')

    #####################
    #end of training
    # Test Step
    # Load the best weights before testing
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

        for i, (feats, coords, labels) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            class_labels_key, regr_labels_key = list(labels.keys())
            class_labels = torch.Tensor(labels[class_labels_key])
            regr_labels = torch.Tensor(labels[regr_labels_key]).to(torch.float32)
            feats, coords, class_labels, regr_labels = feats.to(config.device), \
                                                coords.to(config.device), \
                                                class_labels.to(config.device), \
                                                regr_labels.to(config.device)

            # Forward pass
            logits_classification, logits_regression = model(tile_tokens=feats, task_type=config.task_type, baseline=True)
            loss_class=torch.tensor(0.0)
            loss_regr=torch.tensor(0.0)
            # Compute the classification losses
            if logits_classification is not None:
                loss_class = criteron_class(logits_classification, class_labels).to(torch.float32)
                y_true_class.extend(class_labels.cpu().numpy())
                y_pred_class.extend(torch.softmax(logits_classification, dim=1).cpu().numpy())  # Assuming 2-class classification
            # Compute the regression losses
            if logits_regression is not None:
                loss_regr = criterion_regr(logits_regression, regr_labels).to(torch.float32)
                y_true_regr.extend(regr_labels.cpu().numpy())
                y_pred_regr.extend(logits_regression.cpu().numpy())
            
            test_loss_class += loss_class.item()
            test_loss_regr += loss_regr.item()

        avg_test_loss_class = test_loss_class / len(test_loader)
        avg_test_loss_regr = test_loss_regr / len(test_loader)
        
        if logits_classification is not None:
        # output
            print(f"Test Loss: Crossentropy={avg_test_loss_class:.4f}, MSE={avg_test_loss_regr:.4f}")
            softmax_class_logit=np.array(y_pred_class)
            auroc_class = roc_auc_score(y_true_class, softmax_class_logit[:, 1].tolist()) #first index of softmaxed logits
            print(f"AUROC (Classification) on Test: {auroc_class:.4f}")
        
        if logits_regression is not None:
            pearson_r_regr, _ = pearsonr(y_true_regr, y_pred_regr)
            print(f"Pearson's r (Regression) on Test: {pearson_r_regr:.4f}")
            # plot results 
            data = {'y_true_regr': y_true_regr, 'y_pred_regr': y_pred_regr}
            df = pd.DataFrame(data)
            plt.figure(figsize=(8, 6))
            sns.regplot(x='y_true_regr', y='y_pred_regr', data=df)
            plt.title(f'Regression Correlation Plot\nPearson\'s r: {pearson_r_regr:.4f}', fontsize=16)
            plt.savefig(f'{model_name}-regr_plot.png')

        # Save patient prediction files
        tasks = 0
        if logits_classification is not None:
            test_df_class = test_loader.dataset.targets[class_labels_key]
            test_df_class = pd.DataFrame(test_df_class).reset_index(col_level=0, col_fill='')
            test_df_class["class_logits_neg"] = softmax_class_logit[:, 0].tolist()
            test_df_class["class_logits_pos"] = softmax_class_logit[:, 1].tolist()
            test_df_class["class_pred"] = np.argmax(y_pred_class, axis=1)
            tasks+=1

        if logits_regression is not None:
            test_df_regr = test_loader.dataset.targets[regr_labels_key]
            test_df_regr = pd.DataFrame(test_df_regr).reset_index(col_level=0, col_fill='')
            test_df_regr["regr_logits"] = y_pred_regr
            tasks+=1

        if tasks == 2:
            preds_df = test_df_class.merge(test_df_regr, on='PATIENT', how='left')
        elif tasks == 1 and logits_classification is not None:
            preds_df = test_df_class.copy()
        elif tasks == 1 and logits_regression is not None:
            preds_df = test_df_regr.copy()
        
        preds_df.to_csv(f'{model_name}-patient_preds.csv')