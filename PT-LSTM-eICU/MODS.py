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
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
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

seed = 42                        # Random seed
batch_size = 64                 # Batch size
num_epoch = 300                  # Number of training epochs
learning_rate = 0.00001         # Learning rate

pre_model_path = '/home/wen/Sepsis_MODS/unoptimized_pretrained/eicu_mimic_pre_dead_rate.ckpt'
model_path = '/home/wen/Sepsis_MODS/unoptimized_pretrained/lstm_all_params.ckpt'  # Path to save model checkpoint

# %%
def same_seeds(seed):  # Fix random seed (CPU)
    torch.manual_seed(seed)     # Fix random seed (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For current GPU
        torch.cuda.manual_seed_all(seed)  # For all GPUs
    np.random.seed(seed)  # Ensure consistent random numbers for numpy
    torch.backends.cudnn.benchmark = False  # Can set to True if GPU and network structure fixed
    torch.backends.cudnn.deterministic = True  # Fix network structure

same_seeds(seed)

# %%

sepsis_mimic_train_data = pd.read_csv('/home/wen/Sepsis_MODS/sepsis_mimic_train_data.csv')
sepsis_mimic_test_data = pd.read_csv('/home/wen/Sepsis_MODS/sepsis_mimic_test_data.csv')
sepsis_mimic_valid_data = pd.read_csv('/home/wen/Sepsis_MODS/sepsis_mimic_valid_data.csv')

sepsis_eicu_data = pd.read_csv('/home/wen/Sepsis_MODS/sepsis_eicu_data.csv')

# %%
class SepsisData(Dataset):
    def __init__(self, data, learn_w, lead_w, pred_w):
        xs = []
        ys = []
        row_id = list(set(data['stay_id']))
        for i in row_id:
            data_row = data[data['stay_id'] == i]  # Extract data for each patient ID
            data_row = data_row.sort_values(by='hr')  # Sort by time
            data_row = data_row.reset_index(drop=True)
            data_row = data_row[['stay_id', 'heart_rate', 'mbp', 'temperature', 'spo2', 'resp_rate', 'sbp', 'dbp', 'hr', 'gender', 'age', 'pao2', 'hematocrit', 'wbc', 'creatinine', 'bun', 'sodium', 'albumin', 'bilirubin', 'glucose', 'ph', 'pco2', 'gcs', 'pao2fio2ratio', 'platelet', 'pt', 'potassium', 'epinephrine', 'norepinephrine', 'dopamine', 'dobutamine', 'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'alt', 'ast', 'baseexcess', 'chloride', 'totalco2', 'lactate', 'free_calcium', 'fio2', 'mods']]
            end_boundary = data_row.shape[0]  # When no positive data, boundary = whole table
            # for j in range(data_row.shape[0]):
            #     if data_row.iloc[j, -1] == 1:  # Boundary = the positive row
            #         end_boundary = j + 1
            #         break

            if (end_boundary <= learn_w):  # If boundary smaller than learning window, skip this patient
                continue
            
            for j in range(end_boundary - learn_w - lead_w):
                label = []
                x_data = data_row.iloc[j: j + learn_w, 1:-1]  # Data within learning window
                positive_flag_6 = 0
                for k in range(j + learn_w + lead_w, min(j + lead_w + learn_w + 6, data_row.shape[0])):  # Label positive samples in prediction window
                    if (data_row.iloc[k, -1] == 1):
                        positive_flag_6 = 1
                        label.append(1)
                        break
                if (positive_flag_6 == 0):
                    label.append(0)
                positive_flag_12 = 0
                for k in range(j + learn_w + lead_w, min(j + lead_w + learn_w + 12, data_row.shape[0])):  # Label positive samples
                    if (data_row.iloc[k, -1] == 1):
                        positive_flag_12 = 1
                        label.append(1)
                        break
                if (positive_flag_12 == 0):
                    label.append(0)
                positive_flag_24 = 0
                for k in range(j + learn_w + lead_w, min(j + lead_w + learn_w + 24, data_row.shape[0])):  # Label positive samples
                    if (data_row.iloc[k, -1] == 1):
                        positive_flag_24 = 1
                        label.append(1)
                        break
                if (positive_flag_24 == 0):
                    label.append(0)

                xs.append(torch.tensor(x_data.values, dtype=torch.float32))
                ys.append(torch.tensor(label, dtype=torch.float32))
        self.x = torch.stack(xs)
        self.y = torch.stack(ys)
        
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

# %%
# train_set = SepsisData(sepsis_mimic_train_data, learn_w, lead_w, pred_w)
# test_set = SepsisData(sepsis_mimic_test_data, learn_w, lead_w, pred_w)
# valid_set = SepsisData(sepsis_mimic_valid_data, learn_w, lead_w, pred_w)
eicu_set = SepsisData(sepsis_eicu_data, learn_w, lead_w, pred_w)

# %%
# train_loader = DataLoader(train_set, batch_size=1024, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2)
# test_loader = DataLoader(test_set, batch_size=1024, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2)
# valid_loader = DataLoader(valid_set, batch_size=1024, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2)
eicu_loader = DataLoader(eicu_set, batch_size=1024, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2)

# %%
class GRUWithLayerNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUWithLayerNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, hn = self.gru(x, h0.detach())
        
        # for i in range(self.num_layers):
        #     out = self.layer_norms[i](out)

        out = self.fc1(out[:, -1, :])
        out = self.fc(out)
        return out

# %%
pre_model = GRUWithLayerNorm(input_dim=41, hidden_dim=128, num_layers=12, output_dim=2).to(device)
pre_model.load_state_dict(torch.load(pre_model_path))
print(pre_model.modules)

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
class MyModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers = 1):
        super(MyModel, self).__init__()
        #self.pre_x1_fc_input = pre_model.x1_fc_input
        self.pre_gru = pre_model.gru
        self.pre_fc1 = pre_model.fc1
        #self.pre_fc = pre_model.fc  # Don't import pretrained final layers to avoid dimension mismatch
        num_features = pre_model.hidden_dim
        self.fc = nn.Sequential(
            BasicBlock(num_features, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        h0 = torch.zeros(pre_model.num_layers, x.size(0), pre_model.hidden_dim).requires_grad_().to(device)
        out_x1, hn = self.pre_gru(x, h0.detach())
        out_x1 = self.pre_fc1(out_x1[:, -1, :])
        out = self.fc(out_x1)
        return out

# %%
# for param in pre_model.fc_input.parameters():
#     param.requires_grad = False  # Freeze LSTM layer
# for param in pre_model.gru.parameters():
#     param.requires_grad = False  # Freeze LSTM layer
# for param in pre_model.gln.parameters():
#     param.requires_grad = False  # Freeze LSTM layer

# %%
model = MyModel(hidden_dim=32, output_dim=3).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
print(model.children)

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
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(x)
 
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
        
    mean_train_loss = sum(train_loss) / len(train_loss)
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
        for batch in tqdm(valid_loader):
            x, labels = batch
            x = x.to(device)
            labels = labels.to(device)
            outputs = model(x)

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
    
    # Save the best model
    if mean_valid_auc > best_auc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), model_path)  # Save only the best model to avoid memory issues
        best_auc = mean_valid_auc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvement for {patience} consecutive epochs, early stopping")
            break

# %%
model = MyModel(hidden_dim=32, output_dim=3).to(device)
model.load_state_dict(torch.load(model_path))

# %%
from sklearn.metrics import roc_auc_score, confusion_matrix
torch.backends.cudnn.enabled = True
model.eval()
test_acc = 0.0
threshold = 0.25
test_prob_all_6 = []
test_label_all_6 = []
test_prob_all_12 = []
test_label_all_12 = []
test_prob_all_24 = []
test_label_all_24 = []
with torch.no_grad():
    for i, batch in enumerate(tqdm(eicu_loader)):
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)
        outputs = model(x)
        outputs = torch.sigmoid(outputs)
        
        test_prob_all_6.extend(outputs[:, 0].detach().cpu().numpy())
        test_label_all_6.extend(labels[:, 0].detach().cpu().numpy())
        test_prob_all_12.extend(outputs[:, 1].detach().cpu().numpy())
        test_label_all_12.extend(labels[:, 1].detach().cpu().numpy())
        test_prob_all_24.extend(outputs[:, 2].detach().cpu().numpy())
        test_label_all_24.extend(labels[:, 2].detach().cpu().numpy())

test_prob_all_6 = np.array(test_prob_all_6)
test_label_all_6 = np.array(test_label_all_6)
test_prob_all_12 = np.array(test_prob_all_12)
test_label_all_12 = np.array(test_label_all_12)
test_prob_all_24 = np.array(test_prob_all_24)
test_label_all_24 = np.array(test_label_all_24)
test_pred_all = (test_prob_all_6 >= threshold).astype(int)
test_auc_6 = roc_auc_score(test_label_all_6, test_prob_all_6)
test_auc_12 = roc_auc_score(test_label_all_12, test_prob_all_12)
test_auc_24 = roc_auc_score(test_label_all_24, test_prob_all_24)
mean_auc = (test_auc_6 + test_auc_12 + test_auc_24) / 3
print(f"acc = {(test_label_all_6 == test_pred_all).sum() / len(test_pred_all):.5f}, auc = {mean_auc:.5f}")

cm = confusion_matrix(test_label_all_6, test_pred_all)
TN, FP, FN, TP = cm.ravel()

# Calculate TPR and TNR
TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Sensitivity / Recall
TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # Specificity
print(f"Confusion Matrix:\n{cm}")
print(f"TPR (Sensitivity): {TPR:.4f}")
print(f"TNR (Specificity): {TNR:.4f}")

# %%
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

# Assume these arrays are obtained from your test code
# test_prob_all_6, test_label_all_6, test_pred_all (binary classification results)
y_true = test_label_all_24
y_score = test_prob_all_24
y_pred = (y_score >= threshold).astype(int)
min_gap = 1
min_i = -1
for i in np.arange(0.10, 0.5, 0.01):
    threshold = i
    y_pred = (y_score >= threshold).astype(int)
    n_samples = len(y_true)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn)  # Sensitivity, Recall
    tnr = tn / (tn + fp)  # Specificity

    if (abs(np.mean(tpr) - np.mean(tnr)) < min_gap):
        min_i = i
        min_gap = abs(np.mean(tpr) - np.mean(tnr))

threshold = min_i
y_pred = (y_score >= threshold).astype(int)
n_samples = len(y_true)
n_bootstraps = 1000  # Adjustable as needed
rng = np.random.RandomState(42)

# Store metrics from each bootstrap iteration
auc_list = []
acc_list = []
tpr_list = []
tnr_list = []

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from multiprocessing import Pool
from tqdm import tqdm

def bootstrap_once(seed):
    rng = np.random.RandomState(seed)
    idxs = rng.randint(0, n_samples, n_samples)
    if len(np.unique(y_true[idxs])) < 2:
        return None  # Skip invalid samples

    y_true_b = y_true[idxs]
    y_score_b = y_score[idxs]
    y_pred_b  = y_pred[idxs]

    auc_b = roc_auc_score(y_true_b, y_score_b)
    acc_b = accuracy_score(y_true_b, y_pred_b)
    cm_b  = confusion_matrix(y_true_b, y_pred_b).ravel()
    TN, FP, FN, TP = cm_b
    tpr_b = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    tnr_b = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return auc_b, acc_b, tpr_b, tnr_b

# Create seed list for each task
seeds = list(range(n_bootstraps))

results = []
with Pool() as pool:
    for res in tqdm(pool.imap(bootstrap_once, seeds), total=n_bootstraps, desc="Bootstrapping"):
        if res is not None:
            results.append(res)

# Split results
auc_list, acc_list, tpr_list, tnr_list = zip(*results)

# Convert to NumPy arrays
auc_arr = np.array(auc_list)
acc_arr = np.array(acc_list)
tpr_arr = np.array(tpr_list)
tnr_arr = np.array(tnr_list)

# Compute mean, std, and 95% confidence intervals
def summarize(arr, name):
    mean = np.mean(arr)
    std  = np.std(arr, ddof=1)
    ci_lower, ci_upper = np.percentile(arr, [2.5, 97.5])
    print(f"{name}: mean={mean:.4f}, std={std:.4f}, 95% CI=({ci_lower:.4f}, {ci_upper:.4f})")

summarize(auc_arr, "AUC")
summarize(acc_arr, "Accuracy")
summarize(tpr_arr, "TPR (Sensitivity)")
summarize(tnr_arr, "TNR (Specificity)")

# %%
min_i

# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr_6, tpr_6, thresholds_6 = roc_curve(test_label_all_6, test_prob_all_6)
fpr_12, tpr_12, thresholds_12 = roc_curve(test_label_all_12, test_prob_all_12)
fpr_24, tpr_24, thresholds_24 = roc_curve(test_label_all_24, test_prob_all_24)
roc_auc_6 = auc(fpr_6, tpr_6)
roc_auc_12 = auc(fpr_12, tpr_12)

# Plot ROC curve
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
# %cd Sepsis_MODS/unoptimized_pretrained/
name = 'eicu_PT-LSTM-eICU'
np.save('{}_fpr_6.npy'.format(name), fpr_6)
np.save('{}_tpr_6.npy'.format(name), tpr_6)
np.save('{}_fpr_12.npy'.format(name), fpr_12)
np.save('{}_tpr_12.npy'.format(name), tpr_12)
np.save('{}_fpr_24.npy'.format(name), fpr_24)
np.save('{}_tpr_24.npy'.format(name), tpr_24)

# %% [markdown]
# ## SHAP Analysis

# 