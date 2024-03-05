from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def evaluate(model, test_loader, criterion, model_name, model_path, fold, config):
    criteron_class = criterion['classification']
    criterion_regr = criterion['regression']

    model.load_state_dict(torch.load(model_path))
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
            logits_classification, logits_regression = model(feats) #deploy joint by default

            # Compute the classification losses
            loss_class = criteron_class(logits_classification, class_labels).to(torch.float32)
            y_true_class.extend(class_labels.cpu().numpy())
            y_pred_class.extend(torch.softmax(logits_classification, dim=1).cpu().numpy())  # Assuming 2-class classification
            
            # Compute the regression losses
            loss_regr = criterion_regr(logits_regression, regr_labels).to(torch.float32)
            y_true_regr.extend(regr_labels.cpu().numpy())
            y_pred_regr.extend(logits_regression.cpu().numpy())
            
            test_loss_class += loss_class.item()
            test_loss_regr += loss_regr.item()

        avg_test_loss_class = test_loss_class / len(test_loader)
        avg_test_loss_regr = test_loss_regr / len(test_loader)
        

        # output
        print(f"Test Loss: Crossentropy={avg_test_loss_class:.4f}, MSE={avg_test_loss_regr:.4f}")
        softmax_class_logit=np.array(y_pred_class)
        auroc_class = roc_auc_score(y_true_class, softmax_class_logit[:, 1].tolist()) #first index of softmaxed logits
        print(f"AUROC (Classification) on Test: {auroc_class:.4f}")
        
        if not config.dummy_regr:
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
        test_df_class = test_loader.dataset.targets[class_labels_key]
        test_df_class = pd.DataFrame(test_df_class).reset_index(col_level=0, col_fill='')
        test_df_class["class_logits_neg"] = softmax_class_logit[:, 0].tolist()
        test_df_class["class_logits_pos"] = softmax_class_logit[:, 1].tolist()
        test_df_class["class_pred"] = np.argmax(y_pred_class, axis=1)

        if not config.dummy_regr:
            test_df_regr = test_loader.dataset.targets[regr_labels_key]
            test_df_regr = pd.DataFrame(test_df_regr).reset_index(col_level=0, col_fill='')
            test_df_regr["regr_logits"] = y_pred_regr

        if not config.dummy_regr:
            preds_df = test_df_class.merge(test_df_regr, on='PATIENT', how='left')
        else:
            preds_df = test_df_class.copy()
        
        save_path=f'{model_name}/fold-{fold}'
        os.makedirs(save_path, exist_ok=True)
        preds_df.to_csv(f'{save_path}/patient_preds_deploy.csv')