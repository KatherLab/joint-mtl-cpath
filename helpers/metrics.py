import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import numpy as np

def plot_classification_metrics(preds_files, config, file_path, is_joint=False):
    aurocs = []
    fprs = []
    tprs = []
    aps = []
    f1s = []

    j_aurocs = []
    j_fprs = []
    j_tprs = []
    j_aps = []
    j_f1s = []
    fig, ax = plt.subplots()
    j_fig, j_ax = plt.subplots()
    fig1, ax1 = plt.subplots()
    j_fig1, j_ax1 = plt.subplots()

    full_df = pd.DataFrame(pd.concat(map(pd.read_csv, preds_files)))        
    for fold, preds in enumerate(preds_files):
        df=pd.read_csv(preds)
        y_true_class = df[config.target_label_class].values
        y_pred_class = df["class_logits_pos"].values
        # Compute AUROC
        fpr, tpr, _ = roc_curve(y_true_class, y_pred_class)
        auroc = roc_auc_score(y_true_class, y_pred_class)
        ax.plot(fpr, tpr, label=f"{fold}-AUC = {auroc:0.2f}")
        aurocs.append(auroc)
        tprs.append(tpr)
        fprs.append(fpr)
        # Compute AUPRC
        precision, recall, _ = precision_recall_curve(y_true_class, y_pred_class)
        f1s.append(f1_score(y_true_class, df.class_pred.values))
        ap = average_precision_score(y_true_class, y_pred_class)
        aps.append(ap)
        ax1.plot(recall, precision, label=f"AUC = {ap:0.3f}")

        if is_joint:
            y_pred_regr = df["regr_logits"].values
            median_pred = np.median(df["regr_logits"].values)
            regr_pred = [1 if i > median_pred else 0 for i in y_pred_regr]
            fpr, tpr, _ = roc_curve(y_true_class, y_pred_regr)
            auroc = roc_auc_score(y_true_class, y_pred_regr)
            j_ax.plot(fpr, tpr, label=f"{fold}-AUC = {auroc:0.2f}")
            j_aurocs.append(auroc)
            j_tprs.append(tpr)
            j_fprs.append(fpr)
            # Compute AUPRC
            precision, recall, _ = precision_recall_curve(y_true_class, y_pred_regr)
            j_f1s.append(f1_score(y_true_class, regr_pred))
            ap = average_precision_score(y_true_class, y_pred_regr)
            j_aps.append(ap)
            j_ax1.plot(recall, precision, label=f"AUC = {ap:0.3f}")

        

    # Plot the AUROCs
    ax.legend()
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f'{config.target_label_class} AUROC: {np.mean(aurocs):.3f}, 95% CI: {np.percentile(aurocs, 2.5):.3f}-{np.percentile(aurocs, 97.5):.3f}')
    fig.savefig(f"{file_path}/AUROCS-5folds.svg")
    plt.close(fig)

    # Plot the AUPRCs
    baseline = (full_df[config.target_label_class].values).sum() / len(full_df[config.target_label_class].values)
    ax1.plot([0, 1], [0, 1], "r--", alpha=0)
    ax1.set_aspect("equal")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title(f'{config.target_label_class} AUPRC: {np.mean(aps):.3f}, 95% CI: {np.percentile(aps, 2.5):.3f}-{np.percentile(aps, 97.5):.3f}\n\
                 F1: {np.mean(f1s):.3f}, 95% CI: {np.percentile(f1s, 2.5):.3f}-{np.percentile(f1s, 97.5):.3f}')
    ax1.plot([0, 1], [baseline, baseline], "r--")
    fig1.savefig(f"{file_path}/AUPRCS-5folds.svg")
    plt.close(fig1)

    if is_joint:
        # Plot the AUROCs
        j_ax.legend()
        j_ax.plot([0, 1], [0, 1], "r--")
        j_ax.set_aspect("equal")
        j_ax.set_xlabel("False Positive Rate")
        j_ax.set_ylabel("True Positive Rate")
        j_ax.set_title(f'{config.target_label_class} AUROC: {np.mean(j_aurocs):.3f}, 95% CI: {np.percentile(j_aurocs, 2.5):.3f}-{np.percentile(j_aurocs, 97.5):.3f}')
        j_fig.savefig(f"{file_path}/AUROCS-5folds-regr.svg")
        plt.close(j_fig)

        # Plot the AUPRCs
        baseline = (full_df[config.target_label_class].values).sum() / len(full_df[config.target_label_class].values)
        j_ax1.plot([0, 1], [0, 1], "r--", alpha=0)
        j_ax1.set_aspect("equal")
        j_ax1.set_xlabel("Recall")
        j_ax1.set_ylabel("Precision")
        j_ax1.set_title(f'{config.target_label_class} AUPRC: {np.mean(j_aps):.3f}, 95% CI: {np.percentile(j_aps, 2.5):.3f}-{np.percentile(j_aps, 97.5):.3f}\n\
                    F1: {np.mean(j_f1s):.3f}, 95% CI: {np.percentile(j_f1s, 2.5):.3f}-{np.percentile(j_f1s, 97.5):.3f}')
        j_ax1.plot([0, 1], [baseline, baseline], "r--")
        j_fig1.savefig(f"{file_path}/AUPRCS-5folds-regr.svg")
        plt.close(j_fig1)

def plot_regression_pearson(y_true, y_pred, file_path):
    pearson_corr, pvalue = pearsonr(y_true, y_pred)

    data = {'y_true_regr': y_true, 'y_pred_regr': y_pred}
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.regplot(x='y_true_regr', y='y_pred_regr', data=df)
    plt.title(f'Regression Correlation Plot\nPearson\'s r: {pearson_corr:.4f}, p={pvalue:.2e}', fontsize=16)
    plt.savefig(f"{file_path}/regr_pearsonr_5-folds.svg")