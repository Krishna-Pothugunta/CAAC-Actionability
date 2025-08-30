import os, random
import json
import pandas as pd
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
from sklearn.model_selection import StratifiedKFold

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make torch ops deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
SEED = 42
set_seed(SEED)

#dfannotated = pd.read_csv('C:/Users/kpoth/Downloads/JOC/Performance/complete_set.csv')
dfannotated = pd.read_csv('/mnt/ffs24/home/pothugun/JOC_Workspace/complete_set.csv')
dfannotated = dfannotated.drop_duplicates(subset='video_id', keep='first')

#pathIn = 'C:/Users/kpoth/Downloads/JOC/Final_Results/Performance_Aug5/Stage3/Reg/BV/Mean'
pathIn = "/mnt/gs21/scratch/pothugun/Nov25_28_Output/Final_Output/Screen/BV/CO_ATTN/SoftMax/"

#attn_file = 'C:/Users/kpoth/Downloads/JOC/Final_Results/Reg/BV/attention_entropies.csv'
attn_file = "/mnt/gs21/scratch/pothugun/Nov25_28_Output/CO_ATTENTION_BVS/entropy_updated.csv"
dfattn = pd.read_csv(attn_file)
dffinal = pd.merge(dfattn, dfannotated, on = 'video_id')

videoidslist = []
clslist = []
for file in os.listdir(pathIn):
    videoidslist.append(file[:-3])
    tensor1 = torch.load(f'{pathIn}/{file}')
    clslist.append(tensor1)
    
labels = []
entropy_t2f = []
entropy_f2t = []
missingids = []
for i in range(len(videoidslist)):
    videoid = videoidslist[i]
    try:
        labels.append(dffinal[dffinal['video_id'] == videoid]['Actionable'].iloc[0])
        entropy_t2f.append(dffinal[dffinal['video_id'] == videoid]['entropy_t2f'].iloc[0])
        entropy_f2t.append(dffinal[dffinal['video_id'] == videoid]['entropy_f2t'].iloc[0])
    except IndexError:
        missingids.append(i)
        
for index in sorted(missingids, reverse=True):
    del videoidslist[index]
    del clslist[index] 

clstokens = torch.concat(clslist, dim = 0)
labelstest = torch.tensor(labels, dtype=torch.float32)
cos_sim = torch.tensor(entropy_t2f, dtype=torch.float32)
kld_dim = torch.tensor(entropy_f2t, dtype=torch.float32)

class VideoDataset(Dataset):
    def __init__(self, video_ids, tokens, labels, attention_cos, attention_kld):
        self.video_ids = video_ids
        self.tokens = tokens
        self.labels = labels
        self.attention_cos = attention_cos
        self.attention_kld = attention_kld

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        return self.video_ids[idx], self.tokens[idx], self.labels[idx], self.attention_cos[idx], self.attention_kld[idx]

test_videos = '/mnt/gs21/scratch/pothugun/Nov25_28_Output/test_videoids.json'
#test_videos = 'C:/Users/kpoth/Downloads/JOC/Performance/test_videoids.json'
with open(test_videos) as f:
    test_videoids = json.load(f)

test_idx = [i for i, vid in enumerate(videoidslist) if vid in test_videoids]
train_val_idx = [i for i, vid in enumerate(videoidslist) if vid not in test_videoids]

train_val_videoids = [videoidslist[i] for i in train_val_idx]
test_videoids = [videoidslist[i] for i in test_idx]

train_val_inputs = clstokens[train_val_idx]
test_inputs = clstokens[test_idx]

train_val_labels = labelstest[train_val_idx]
test_labels = labelstest[test_idx]

train_val_cos = cos_sim[train_val_idx]
test_cos = cos_sim[test_idx]

train_val_kld = kld_dim[train_val_idx]
test_kld = kld_dim[test_idx]

#train_videoids, test_videoids, train_inputs, test_inputs, train_labels, test_labels, train_cos, test_cos, train_kld, test_kld = train_test_split(videoidslist, clstokens, labelstest, cos_sim, kld_dim, test_size=0.2, random_state=42)
train_videoids, val_videoids, train_inputs, val_inputs, train_labels, val_labels, train_cos, val_cos, train_kld, val_kld = train_test_split(train_val_videoids, train_val_inputs, train_val_labels, train_val_cos, train_val_kld, test_size=0.25, random_state=42)

train_dataset = VideoDataset(train_videoids, train_inputs, train_labels, train_cos, train_kld)
val_dataset = VideoDataset(val_videoids, val_inputs, val_labels, val_cos, val_kld)
test_dataset = VideoDataset(test_videoids, test_inputs, test_labels, test_cos, test_kld)

#test_dataset_new = VideoDataset(newvideoidslist, clstokensnew, labelstestnew, cos_sim_new, kld_dim_new)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) 

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Sequential(
            #nn.Linear(input_dim, 1028),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            #nn.Dropout(0.5),
            # nn.Linear(1028, 512),
            # nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(128, 1),  
            #nn.Sigmoid()  
        )
    
    def forward(self, x):
        return self.fc(x)

def make_dataset_from_indices(idxs):
    idxs = np.array(idxs)
    return VideoDataset(
        np.array(train_val_videoids)[idxs],
        train_val_inputs[idxs],
        train_val_labels[idxs],
        train_val_cos[idxs],
        train_val_kld[idxs],
    )

def get_pos_weight_from_loader(loader):
    total_pos = 0
    total_neg = 0
    for batch in loader:
        labels = batch[2]
        labels = labels.view(-1)
        total_pos += (labels == 1).sum().item()
        total_neg += (labels == 0).sum().item()
    pos_weight = total_neg / max(1, total_pos)
    return torch.tensor([pos_weight], dtype=torch.float)    

def weighted_bce_with_attention_loss(predictions, targets, weights_cos, weights_kld, pos_weight, alpha, beta):
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_loss = bce_loss_fn(predictions, targets)

    penalty_cos = weights_cos.mean()
    penalty_kld = weights_kld.mean()

    total_loss = bce_loss + alpha * penalty_cos + beta * penalty_kld
    return total_loss

def train_one_run_fold(train_loader, val_loader, test_loader,
                       lr, weight_decay, excel_path, alpha, beta,
                       device="cuda" if torch.cuda.is_available() else "cpu"):
    set_seed(42)

    model = BinaryClassifier(input_dim=768).to(device)
    pos_weight = get_pos_weight_from_loader(train_loader).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_epochs = 50
    patience = 5
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_path = os.path.join(excel_path,
                                   f"best_model_lr{lr}_wd{weight_decay}_alpha{alpha}_beta{beta}.pth")

    for epoch in range(num_epochs):
        # --- training loop ---
        model.train()
        train_loss = 0.0
        for video_ids, tokens, labels, attn_cos, attn_kld in train_loader:
            optimizer.zero_grad()
            tokens = tokens.to(device)
            labels = labels.to(device).float()
            attn_cos = attn_cos.to(device)
            attn_kld = attn_kld.to(device)

            outputs = model(tokens).squeeze()
            loss = weighted_bce_with_attention_loss(outputs, labels, attn_cos, attn_kld,
                                                    pos_weight, alpha, beta)
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        # --- validation loop ---
        model.eval()
        val_loss = 0.0
        val_true, val_scores = [], []
        with torch.no_grad():
            for video_ids, tokens, labels, attn_cos, attn_kld in val_loader:
                tokens = tokens.to(device)
                labels = labels.to(device).float()
                attn_cos = attn_cos.to(device)
                attn_kld = attn_kld.to(device)

                outputs = model(tokens).squeeze()
                loss = weighted_bce_with_attention_loss(outputs, labels, attn_cos, attn_kld,
                                                        pos_weight, alpha, beta)
                val_loss += loss.item()
                val_true.extend(labels.cpu().numpy())
                val_scores.extend(outputs.cpu().numpy())

        val_loss /= max(1, len(val_loader))
        val_auc = roc_auc_score(val_true, val_scores)
        val_preds = (np.array(val_scores) >= 0.5).astype(int)
        val_f1 = f1_score(val_true, val_preds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                break

    # reload best checkpoint and evaluate on test
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    y_true_test, y_scores_test, test_video_ids = [], [], []
    with torch.no_grad():
        for video_ids, tokens, labels, attn_cos, attn_kld in test_loader:
            tokens = tokens.to(device)
            labels = labels.to(device).float()
            outputs = model(tokens).squeeze()
            y_true_test.extend(labels.cpu().numpy())
            y_scores_test.extend(outputs.cpu().numpy())
            test_video_ids.extend(list(video_ids))

    y_true_test = np.array(y_true_test)
    y_scores_test = np.array(y_scores_test)
    y_pred_test = (y_scores_test >= 0.5).astype(int)

    test_auc = roc_auc_score(y_true_test, y_scores_test)
    test_f1 = f1_score(y_true_test, y_pred_test, average="weighted", zero_division=0)
    test_precision = precision_score(y_true_test, y_pred_test, average="weighted", zero_division=0)
    test_recall = recall_score(y_true_test, y_pred_test, average="weighted", zero_division=0)
    
    report = classification_report(y_true_test, y_pred_test, output_dict=True, zero_division=0)

    return {
        "val_auc": val_auc, "val_f1": val_f1,
        "test_auc": test_auc, "test_f1": test_f1,
        "test_precision": test_precision, "test_recall": test_recall,
        "classification_report": report,
        "test_video_ids": test_video_ids,
        "y_true_test": y_true_test.tolist(),
        "y_scores_test": y_scores_test.tolist(),
        "model_path": best_model_path
    }
    
def performance_accuracy_cv5(excel_root,
                             lrs=(1e-4, 1e-3, 1e-2, 1e-1),
                             wds=(0.0, 1e-4, 1e-3, 1e-2, 1e-1),
                             alphas=(0.0, 0.5, 1.0),
                             betas=(0.0, 0.5, 1.0),
                             device="cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(excel_root, exist_ok=True)

    # fixed test set
    test_dataset = VideoDataset(test_videoids, test_inputs, test_labels, test_cos, test_kld)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    X_pool = np.arange(len(train_val_videoids))
    y_pool = np.array(train_val_labels)

    cv_dir = os.path.join(excel_root, "cv5")
    os.makedirs(cv_dir, exist_ok=True)
    print(f"\n================ CV = 5 ================\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_rows = []       
    summary_rows = []    
    all_preds_rows = []  

    for lr in lrs:
        for wd in wds:
            for alpha in alphas:
                for beta in betas:
                    fold_rows = []
                    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_pool, y_pool), 1):
                        train_dataset = make_dataset_from_indices(tr_idx)
                        val_dataset   = make_dataset_from_indices(va_idx)
                        train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
                        val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=True)

                        fold_out_dir = os.path.join(cv_dir, f"lr{lr}_wd{wd}_a{alpha}_b{beta}", f"fold{fold_idx}")
                        os.makedirs(fold_out_dir, exist_ok=True)

                        res = train_one_run_fold(
                            train_loader, val_loader, test_loader,
                            lr=lr, weight_decay=wd,
                            excel_path=fold_out_dir,
                            alpha=alpha, beta=beta, device=device
                        )

                        preds_df = pd.DataFrame({
                            "video_id": res["test_video_ids"],
                            "y_true":   res["y_true_test"],
                            "y_score":  res["y_scores_test"],
                        })
                        preds_df["y_pred"] = (preds_df["y_score"] >= 0.5).astype(int)
                        preds_csv_path = os.path.join(fold_out_dir, "test_predictions.csv")
                        preds_df.to_csv(preds_csv_path, index=False)

                        preds_with_meta = preds_df.copy()
                        preds_with_meta.insert(0, "fold", fold_idx)
                        preds_with_meta.insert(0, "cv", 5)
                        preds_with_meta["lr"] = lr
                        preds_with_meta["weight_decay"] = wd
                        preds_with_meta["alpha"] = alpha
                        preds_with_meta["beta"] = beta
                        all_preds_rows.append(preds_with_meta)

                        cls_df = pd.DataFrame(res["classification_report"]).T
                        cls_df.to_csv(os.path.join(fold_out_dir, "classification_report.csv"))

                        row = {
                            "cv": 5, "fold": fold_idx,
                            "lr": lr, "weight_decay": wd, "alpha": alpha, "beta": beta,
                            **{k: v for k, v in res.items()
                               if k in ["val_auc","val_f1","test_auc","test_f1","test_precision","test_recall","model_path"]}
                        }
                        fold_rows.append(row)
                        all_rows.append(row)

                    df_combo = pd.DataFrame(fold_rows)
                    avg_row = {
                        "cv": 5,
                        "lr": lr, "weight_decay": wd, "alpha": alpha, "beta": beta,
                        "val_auc_mean":  df_combo["val_auc"].mean(),  "val_auc_std":  df_combo["val_auc"].std(ddof=1),
                        "val_f1_mean":   df_combo["val_f1"].mean(),   "val_f1_std":   df_combo["val_f1"].std(ddof=1),
                        "test_auc_mean": df_combo["test_auc"].mean(), "test_auc_std": df_combo["test_auc"].std(ddof=1),
                        "test_f1_mean":  df_combo["test_f1"].mean(),  "test_f1_std":  df_combo["test_f1"].std(ddof=1),
                    }
                    summary_rows.append(avg_row)

    df_folds = pd.DataFrame(all_rows)
    df_folds.to_csv(os.path.join(cv_dir, "fold_metrics_all_combos.csv"), index=False)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(cv_dir, "grid_results_cv_avg.csv"), index=False)

    if all_preds_rows:
        df_all_preds = pd.concat(all_preds_rows, ignore_index=True)
        df_all_preds.to_csv(os.path.join(cv_dir, "all_test_predictions.csv"), index=False)

    best = df_summary.sort_values(by=["val_f1_mean", "val_auc_mean"], ascending=False).iloc[0]
    print(f"Best (CV=5) -> lr={best['lr']} wd={best['weight_decay']} alpha={best['alpha']} beta={best['beta']}")
    print(f"Val AUC: {best['val_auc_mean']:.4f} +/- {best['val_auc_std']:.4f} | "
          f"Val F1: {best['val_f1_mean']:.4f} +/- {best['val_f1_std']:.4f}")
    print(f"Test AUC: {best['test_auc_mean']:.4f} +/- {best['test_auc_std']:.4f} | "
          f"Test F1: {best['test_f1_mean']:.4f} +/- {best['test_f1_std']:.4f}")

excel_root = "/mnt/gs21/scratch/pothugun/Performance/Aug29/SC/Best_Method"
performance_accuracy_cv5(
    excel_root,
    lrs=(1e-4, 1e-3, 1e-2, 1e-1),
    wds=(0.0, 1e-4, 1e-3, 1e-2, 1e-1),
    alphas=(0.25, 0.5, 0.75, 1.0),
    betas=(0.25, 0.5, 0.75, 1.0),
    device="cpu" 
)