# %%
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from tqdm import tqdm
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# %%
# Check if GPU is available, otherwise use CPU
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

# %%
learn_w = 4
lead_w = 0
pred_w = 4

test_ratio = 0.2                # Test set ratio
valid_ratio = 0.1               # Validation set ratio

input_dim = 41
hidden_dim = 64
num_layers = 6
output_dim = 2

seed = 42                       # Random seed
batch_size = 64                 # Batch size
num_epoch = 300                 # Number of training epochs
learning_rate = 0.00001         # Learning rate

pre_model_path = '/home/wen/Sepsis_MODS/Not_pretrained_but_optimized_structure/transformer_eicu_mimic_pre_dead_rate.ckpt'
model_path = '/home/wen/Sepsis_MODS/Not_pretrained_but_optimized_structure/transformer_all_params.ckpt'  # Path to save model checkpoint

# %%
def same_seeds(seed):  # Fix random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(seed)

# %%
# Load datasets
sepsis_mimic_train_data = pd.read_csv('/home/wen/Sepsis_MODS/sepsis_mimic_train_data.csv')
sepsis_mimic_test_data = pd.read_csv('/home/wen/Sepsis_MODS/sepsis_mimic_test_data.csv')
sepsis_mimic_valid_data = pd.read_csv('/home/wen/Sepsis_MODS/sepsis_mimic_valid_data.csv')
sepsis_eicu_data = pd.read_csv('/home/wen/Sepsis_MODS/sepsis_eicu_data.csv')

# %%
class SepsisData(Dataset):
    def __init__(self, data, learn_w, lead_w, pred_w):
        x1s = []  # Non-invasive data
        x2s = []  # Invasive data
        ys = []
        row_id = list(set(data['stay_id']))
        for i in row_id:
            data_row = data[data['stay_id'] == i]
            data_row = data_row.sort_values(by='hr')  # Sort by time
            data_row = data_row.reset_index(drop=True)
            end_boundary = data_row.shape[0]  # If no positive label, full table as boundary

            if (end_boundary <= learn_w):  # Skip if data shorter than learning window
                continue
            
            for j in range(end_boundary - learn_w - lead_w):
                label = []
                x_data = data_row.iloc[j: j + learn_w, 1:-1]
                x1_data = x_data.loc[:, ['heart_rate', 'mbp', 'temperature', 'spo2', 'resp_rate', 'sbp', 'dbp']]
                x2_data = x_data.drop(['heart_rate', 'mbp', 'temperature', 'spo2', 'resp_rate', 'sbp', 'dbp'], axis=1).iloc[-1, :]

                # Labeling for 6h, 12h, 24h prediction windows
                positive_flag_6 = 0
                for k in range(j + learn_w + lead_w, min(j + lead_w + learn_w + 6, data_row.shape[0])):
                    if (data_row.iloc[k, -1] == 1):
                        positive_flag_6 = 1
                        label.append(1)
                        break
                if (positive_flag_6 == 0):
                    label.append(0)

                positive_flag_12 = 0
                for k in range(j + learn_w + lead_w, min(j + lead_w + learn_w + 12, data_row.shape[0])):
                    if (data_row.iloc[k, -1] == 1):
                        positive_flag_12 = 1
                        label.append(1)
                        break
                if (positive_flag_12 == 0):
                    label.append(0)

                positive_flag_24 = 0
                for k in range(j + learn_w + lead_w, min(j + lead_w + learn_w + 24, data_row.shape[0])):
                    if (data_row.iloc[k, -1] == 1):
                        positive_flag_24 = 1
                        label.append(1)
                        break
                if (positive_flag_24 == 0):
                    label.append(0)

                x1s.append(torch.tensor(x1_data.values, dtype=torch.float32))
                x2s.append(torch.tensor(x2_data.values, dtype=torch.float32))
                ys.append(torch.tensor(label, dtype=torch.float32))
        self.x1 = torch.stack(x1s)
        self.x2 = torch.stack(x2s)
        self.y = torch.stack(ys)
        
    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.y[index]
    
    def __len__(self):
        return len(self.x1)

# %%
# train_set = SepsisData(sepsis_mimic_train_data, learn_w, lead_w, pred_w)
test_set = SepsisData(sepsis_mimic_test_data, learn_w, lead_w, pred_w)
# valid_set = SepsisData(sepsis_mimic_valid_data, learn_w, lead_w, pred_w)
eicu_set = SepsisData(sepsis_eicu_data, learn_w, lead_w, pred_w)

# %%
# train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
# valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
eicu_loader = DataLoader(eicu_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# %%
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
    def forward(self, x):
        x = self.block(x)
        return x
    
class GRU(nn.Module):
    def __init__(self, x1_input_dim, x1_hidden_dim, x1_num_layers, x2_input_dim, x2_hidden_dim, x2_num_layers, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.x1_hidden_dim = x1_hidden_dim
        self.x1_num_layers = x1_num_layers
        self.x1_fc_input = nn.Sequential(
            nn.Linear(x1_input_dim, x1_hidden_dim),
            nn.LayerNorm(x1_hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(0.3)
        ) 
        self.gru = nn.GRU(x1_hidden_dim, x1_hidden_dim, x1_num_layers, batch_first=True, dropout=0.5)

        self.gln = nn.Sequential(
            nn.LayerNorm(x1_hidden_dim)
        )
        self.x1_fc_output = nn.Linear(x1_hidden_dim, x1_hidden_dim)
        self.x2_fc = nn.Sequential(
            BasicBlock(x2_input_dim, x2_hidden_dim),
            *[BasicBlock(x2_hidden_dim, x2_hidden_dim) for _ in range(x2_num_layers - 1)],
        )
        self.fc = nn.Sequential(
            BasicBlock(x1_hidden_dim + x2_hidden_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x1, x2):  # x1: non-invasive, x2: invasive
        out_x1 = self.x1_fc_input(x1)
        h0 = torch.zeros(self.x1_num_layers, out_x1.size(0), self.x1_hidden_dim).requires_grad_().to(device)
        out_x1, hn = self.gru(out_x1, h0.detach())
        out_x1 = self.gln(out_x1)
        out_x1 = self.x1_fc_output(out_x1[:, -1, :])
        out_x2 = self.x2_fc(x2)
        out = torch.cat((out_x1, out_x2), dim=1)
        out = self.fc(out)
        return out

# %%
model = GRU(x1_input_dim=7, x1_hidden_dim=8, x1_num_layers=8, x2_input_dim=34, x2_hidden_dim=64, x2_num_layers=10, hidden_dim=64, num_layers=8, output_dim=3).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
print(model)

# %%
from sklearn.metrics import roc_auc_score
best_acc = 0.0
best_auc = 0.0
stale = 0
patience = 30
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    train_loss = []
    train_accs = []
    train_aucs = []
    
    train_prob_all_6 = []
    train_label_all_6 = []
    train_prob_all_12 = []
    train_label_all_12 = []
    train_prob_all_24 = []
    train_label_all_24 = []
    for batch in tqdm(train_loader):
        x1, x2, labels = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(x1, x2)
 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        train_prob_all_6.extend(outputs[:, 0].detach().cpu().numpy())
        train_label_all_6.extend(labels[:, 0].detach().cpu().numpy())
        train_prob_all_12.extend(outputs[:, 1].detach().cpu().numpy())
        train_label_all_12.extend(labels[:, 1].detach().cpu().numpy())
        train_prob_all_24.extend(outputs[:, 2].detach().cpu().numpy())
        train_label_all_24.extend(labels[:, 2].detach().cpu().numpy())

    mean_train_loss = sum(train_loss)/len(train_loss)
    train_auc_6 = roc_auc_score(train_label_all_6, train_prob_all_6)
    train_auc_12 = roc_auc_score(train_label_all_12, train_prob_all_12)
    train_auc_24 = roc_auc_score(train_label_all_24, train_prob_all_24)
    print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {mean_train_loss:.5f}, auc = {((train_auc_6 + train_auc_12 + train_auc_24) / 3):.5f}")
    
    
    valid_loss = []
    valid_accs = []
    valid_aucs = []
    valid_prob_all_6 = []
    valid_label_all_6 = []
    valid_prob_all_12 = []
    valid_label_all_12 = []
    valid_prob_all_24 = []
    valid_label_all_24 = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x1, x2, labels = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)
            outputs = model(x1, x2)

            loss = criterion(outputs, labels)
            valid_loss.append(loss.item())

            valid_prob_all_6.extend(outputs[:, 0].detach().cpu().numpy())
            valid_label_all_6.extend(labels[:, 0].detach().cpu().numpy())
            valid_prob_all_12.extend(outputs[:, 1].detach().cpu().numpy())
            valid_label_all_12.extend(labels[:, 1].detach().cpu().numpy())
            valid_prob_all_24.extend(outputs[:, 2].detach().cpu().numpy())
            valid_label_all_24.extend(labels[:, 2].detach().cpu().numpy())
    
    mean_valid_loss = sum(valid_loss) / len(valid_loss)
    valid_auc_6 = roc_auc_score(valid_label_all_6, valid_prob_all_6)
    valid_auc_12 = roc_auc_score(valid_label_all_12, valid_prob_all_12)
    valid_auc_24 = roc_auc_score(valid_label_all_24, valid_prob_all_24)
    mean_valid_auc = ((valid_auc_6 + valid_auc_12 + valid_auc_24) / 3)
    print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {mean_valid_loss:.5f}, auc = {mean_valid_auc:.5f}")
    # Save best model
    if mean_valid_auc > best_auc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), model_path)
        best_auc = mean_valid_auc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvement for {patience} consecutive epochs, early stopping")
            break

# %%
# Load the best model
model = GRU(
    x1_input_dim=7, x1_hidden_dim=8, x1_num_layers=8,
    x2_input_dim=34, x2_hidden_dim=64, x2_num_layers=10,
    hidden_dim=64, num_layers=8, output_dim=3
).to(device)
model.load_state_dict(torch.load(model_path))

# %%
from sklearn.metrics import roc_auc_score, confusion_matrix
torch.backends.cudnn.enabled = True
model.eval()

test_acc = 0.0
threshold = 0.10
test_prob_all_6 = []
test_label_all_6 = []
test_prob_all_12 = []
test_label_all_12 = []
test_prob_all_24 = []
test_label_all_24 = []

# Evaluation
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        x1, x2, labels = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)
        outputs = model(x1, x2)
        outputs = torch.sigmoid(outputs)
        
        test_prob_all_6.extend(outputs[:, 0].detach().cpu().numpy())
        test_label_all_6.extend(labels[:, 0].detach().cpu().numpy())
        test_prob_all_12.extend(outputs[:, 1].detach().cpu().numpy())
        test_label_all_12.extend(labels[:, 1].detach().cpu().numpy())
        test_prob_all_24.extend(outputs[:, 2].detach().cpu().numpy())
        test_label_all_24.extend(labels[:, 2].detach().cpu().numpy())

# Convert to numpy arrays
test_prob_all_6 = np.array(test_prob_all_6)
test_label_all_6 = np.array(test_label_all_6)
test_prob_all_12 = np.array(test_prob_all_12)
test_label_all_12 = np.array(test_label_all_12)
test_prob_all_24 = np.array(test_prob_all_24)
test_label_all_24 = np.array(test_label_all_24)

# Compute metrics
test_pred_all = (test_prob_all_6 >= threshold).astype(int)
test_auc_6 = roc_auc_score(test_label_all_6, test_prob_all_6)
test_auc_12 = roc_auc_score(test_label_all_12, test_prob_all_12)
test_auc_24 = roc_auc_score(test_label_all_24, test_prob_all_24)
mean_auc = (test_auc_6 + test_auc_12 + test_auc_24) / 3
print(f"acc = {(test_label_all_6 == test_pred_all).sum() / len(test_pred_all):.5f}, auc = {mean_auc:.5f}")

# Confusion matrix and TPR/TNR
cm = confusion_matrix(test_label_all_6, test_pred_all)
TN, FP, FN, TP = cm.ravel()

TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Sensitivity / Recall
TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # Specificity

print(f"Confusion Matrix:\n{cm}")
print(f"TPR (Sensitivity): {TPR:.4f}")
print(f"TNR (Specificity): {TNR:.4f}")

# %%
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

# Example test arrays
y_true = test_label_all_24
y_score = test_prob_all_24
y_pred = (y_score >= threshold).astype(int)

# Find threshold that balances TPR and TNR
min_gap = 1
min_i = -1
for i in np.arange(0.10, 0.3, 0.01):
    threshold = i
    y_pred = (y_score >= threshold).astype(int)
    n_samples = len(y_true)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    if (abs(np.mean(tpr) - np.mean(tnr)) < min_gap):
        min_i = i
        min_gap = abs(np.mean(tpr) - np.mean(tnr))

threshold = min_i
y_pred = (y_score >= threshold).astype(int)
n_samples = len(y_true)
n_bootstraps = 1000  # Adjustable
rng = np.random.RandomState(42)

# Bootstrap evaluation metrics
auc_list = []
acc_list = []
tpr_list = []
tnr_list = []

from multiprocessing import Pool
from tqdm import tqdm

def bootstrap_once(seed):
    rng = np.random.RandomState(seed)
    idxs = rng.randint(0, n_samples, n_samples)
    if len(np.unique(y_true[idxs])) < 2:
        return None  # Skip invalid samples

    y_true_b = y_true[idxs]
    y_score_b = y_score[idxs]
    y_pred_b = y_pred[idxs]

    auc_b = roc_auc_score(y_true_b, y_score_b)
    acc_b = accuracy_score(y_true_b, y_pred_b)
    cm_b = confusion_matrix(y_true_b, y_pred_b).ravel()
    TN, FP, FN, TP = cm_b
    tpr_b = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    tnr_b = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return auc_b, acc_b, tpr_b, tnr_b

# Parallel bootstrap
seeds = list(range(n_bootstraps))
results = []
with Pool() as pool:
    for res in tqdm(pool.imap(bootstrap_once, seeds), total=n_bootstraps, desc="Bootstrapping"):
        if res is not None:
            results.append(res)

# Split results
auc_list, acc_list, tpr_list, tnr_list = zip(*results)

# Convert to numpy arrays
auc_arr = np.array(auc_list)
acc_arr = np.array(acc_list)
tpr_arr = np.array(tpr_list)
tnr_arr = np.array(tnr_list)

# Compute mean, std, and 95% CI
def summarize(arr, name):
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    ci_lower, ci_upper = np.percentile(arr, [2.5, 97.5])
    print(f"{name}: mean={mean:.4f}, std={std:.4f}, 95% CI=({ci_lower:.4f}, {ci_upper:.4f})")

summarize(auc_arr, "AUC")
summarize(acc_arr, "Accuracy")
summarize(tpr_arr, "TPR (Sensitivity)")
summarize(tnr_arr, "TNR (Specificity)")

# %%
# ROC curves
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr_6, tpr_6, thresholds_6 = roc_curve(test_label_all_6, test_prob_all_6)
fpr_12, tpr_12, thresholds_12 = roc_curve(test_label_all_12, test_prob_all_12)
fpr_24, tpr_24, thresholds_24 = roc_curve(test_label_all_24, test_prob_all_24)
roc_auc_6 = auc(fpr_6, tpr_6)
roc_auc_12 = auc(fpr_12, tpr_12)

# Plot ROC curves
plt.figure()
plt.plot(fpr_6, tpr_6, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_6:.2f})')
plt.plot(fpr_12, tpr_12, color='darkgreen', lw=2)
plt.plot(fpr_24, tpr_24, color='darkblue', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# %%
# Save ROC curve data
name = 'eicu_MLPLSTM'
np.save('{}_fpr_6.npy'.format(name), fpr_6)
np.save('{}_tpr_6.npy'.format(name), tpr_6)
np.save('{}_fpr_12.npy'.format(name), fpr_12)
np.save('{}_tpr_12.npy'.format(name), tpr_12)
np.save('{}_fpr_24.npy'.format(name), fpr_24)
np.save('{}_tpr_24.npy'.format(name), tpr_24)

# %%
# Compute confidence interval for AUC
n_bootstraps = 10000
rng_seed = 42  # Fixed random seed
bootstrapped_aucs = []
rng = np.random.default_rng(rng_seed)

for i in range(n_bootstraps):
    indices = rng.integers(0, len(test_prob_all_6), len(test_prob_all_6))
    if len(np.unique(test_label_all_6[indices])) < 2:
        continue

    fpr_boot, tpr_boot, _ = roc_curve(test_label_all_6[indices], test_prob_all_6[indices])
    roc_auc_boot = auc(fpr_boot, tpr_boot)
    bootstrapped_aucs.append(roc_auc_boot)

# 95% CI
alpha = 0.95
lower = np.percentile(bootstrapped_aucs, (1 - alpha) / 2 * 100)
upper = np.percentile(bootstrapped_aucs, (alpha + (1 - alpha) / 2) * 100)

# Print AUC and confidence interval
print(f"AUC = {roc_auc_6:.2f}")
print(f"95% CI = [{lower:.2f}, {upper:.2f}]")
