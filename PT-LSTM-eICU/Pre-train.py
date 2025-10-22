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
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

# %%
learn_w = 4
lead_w = 0
pred_w = 24

test_ratio = 0.2                # Test set ratio
valid_ratio = 0.1               # Validation set ratio

input_dim = 41
hidden_dim = 128
num_layers = 12
output_dim = 2

seed = 42                        # Random seed
batch_size = 64                 # Batch size
num_epoch = 300                 # Number of training epochs
learning_rate = 0.00001         # Learning rate

model_pre_path = './eicu_mimic_pre_dead_rate.ckpt'     # Path to save checkpoint (model saving location)

# %%
def same_seeds(seed):  # Fix random seeds (CPU)
    torch.manual_seed(seed)  # Fix random seeds (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set for current GPU
        torch.cuda.manual_seed_all(seed)  # Set for all GPUs
    np.random.seed(seed)  # Ensure consistent random numbers when using random functions later
    torch.backends.cudnn.benchmark = False  # Set to True if GPU and network structure are fixed
    torch.backends.cudnn.deterministic = True  # Fix network structure

same_seeds(seed)

# %%

valid_data = dd.read_csv('../valid_data.csv', blocksize=64e6)
train_data = dd.read_csv('../train_data.csv', blocksize=64e6)
test_data = dd.read_csv('../test_data.csv', blocksize=64e6)

sepsis_mimic_train_data = dd.read_csv('../sepsis_mimic_train_data.csv', blocksize=64e6)
sepsis_mimic_test_data = dd.read_csv('../sepsis_mimic_test_data.csv', blocksize=64e6)
sepsis_mimic_valid_data = dd.read_csv('../sepsis_mimic_valid_data.csv', blocksize=64e6)

sepsis_eicu_data = dd.read_csv('../sepsis_eicu_data.csv', blocksize=64e6)
learn_w = 4
lead_w = 0
pred_w = 24

# %%

class DeathData(Dataset):
    def __init__(self, data, learn_w, lead_w, pred_w):
        self.xs = []
        self.ys = []
        self.learn_w = learn_w
        self.lead_w = lead_w
        result = data.groupby('stay_id').apply(
            self.datapressing, meta=(None, 'object')
        )
        with ProgressBar():
            batches = result.compute()
        for batch in batches:
            if (len(batch) != 2):
                continue
            x_batch, y_batch = batch
            self.xs.extend(x_batch)
            self.ys.extend(y_batch)
        self.x = torch.stack(self.xs)
        self.y = torch.stack(self.ys)


    def datapressing(self, group):
        data_row = group.sort_values(by='hr').reset_index(drop=True)
        data_row = data_row[['stay_id', 'heart_rate', 'mbp', 'temperature', 'spo2', 'resp_rate', 'sbp', 'dbp', 'hr', 'gender', 'age', 'pao2', 'hematocrit', 'wbc', 'creatinine', 'bun', 'sodium', 'albumin', 'bilirubin', 'glucose', 'ph', 'pco2', 'gcs', 'pao2fio2ratio', 'platelet', 'pt', 'potassium', 'epinephrine', 'norepinephrine', 'dopamine', 'dobutamine', 'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'alt', 'ast', 'baseexcess', 'chloride', 'totalco2', 'lactate', 'free_calcium', 'fio2', 'death_label']]
        data_row = data_row.to_numpy()
        end_boundary = len(data_row)
        if end_boundary <= self.learn_w:
            return []

        batch_x, batch_y = [], []
        for j in range(end_boundary - self.learn_w - self.lead_w + 1):
            x_data = data_row[j:j + self.learn_w, 1:-1]
            positive_flag = int(data_row[j + self.learn_w - 1, -1])
            batch_x.append(torch.tensor(x_data, dtype=torch.float32))
            batch_y.append(torch.tensor(positive_flag, dtype=torch.long))
        return batch_x, batch_y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


# %%
# train_set = DeathData(train_data, learn_w, lead_w, pred_w)
test_set = DeathData(test_data, learn_w, lead_w, pred_w)
# valid_set = DeathData(valid_data, learn_w, lead_w, pred_w)

# %%
# train_loader = DataLoader(train_set, batch_size=4096, shuffle=True, pin_memory=True, num_workers=16)
test_loader = DataLoader(test_set, batch_size=4096, shuffle=True, pin_memory=True, num_workers=16)
# valid_loader = DataLoader(valid_set, batch_size=4096, shuffle=True, pin_memory=True, num_workers=16)

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

model = GRUWithLayerNorm(input_dim=41, hidden_dim=128, num_layers=12, output_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
print(model)

# %%
from sklearn.metrics import roc_auc_score
best_acc = 0.0
best_auc = 0.0
stale = 0
patience = 300
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    train_loss = []
    train_accs = []
    train_aucs = []
    
    train_prob_all = []
    train_label_all = []
    for batch in tqdm(train_loader):
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(x)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()
        # auc = roc_auc_score(labels.detach(), outputs.cpu().numpy(), multi_class='ovr', average='macro')
        train_loss.append(loss.item())
        train_accs.append(acc.detach().item())
        # train_aucs.append(auc.detach().item())
        train_prob_all.extend(outputs[:, 1].detach().cpu().numpy())
        train_label_all.extend(labels.detach().cpu().numpy())
        

    mean_train_acc = sum(train_accs) / len(train_accs)
    mean_train_loss = sum(train_loss) / len(train_loss)
    # mean_train_auc = sum(train_aucs) / len(train_aucs)
    train_auc = roc_auc_score(train_label_all, train_prob_all)
    print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {mean_train_loss:.5f}, acc = {mean_train_acc:.5f}, auc = {train_auc:.5f}")
    
    valid_loss = []
    valid_accs = []
    valid_aucs = []
    valid_prob_all = []
    valid_label_all = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            x, labels = batch
            x = x.to(device)
            labels = labels.to(device)
            outputs = model(x)

            loss = criterion(outputs, labels)

            acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()
            # auc = roc_auc_score(labels.detach(), outputs.detach(), multi_class='ovr', average='macro')
            valid_loss.append(loss.item())
            valid_accs.append(acc.detach().item())
            # valid_aucs.append(auc.detach().item())
            valid_prob_all.extend(outputs[:, 1].detach().cpu().numpy())
            valid_label_all.extend(labels.detach().cpu().numpy())
    
    mean_valid_loss = sum(valid_loss) / len(valid_loss)
    mean_valid_acc = sum(valid_accs) / len(valid_accs)
    valid_auc = roc_auc_score(valid_label_all, valid_prob_all)
    print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {mean_valid_loss:.5f}, acc = {mean_valid_acc:.5f}, auc = {valid_auc:.5f}")
    
    # Save models
    if valid_auc > best_auc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), model_pre_path)  # Only save the best model to prevent memory overflow
        best_auc = valid_auc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvement for {patience} consecutive epochs, early stopping")
            break



# %%
model = GRUWithLayerNorm(input_dim=41, hidden_dim=128, num_layers=12, output_dim=2).to(device)
model.load_state_dict(torch.load(model_pre_path))

# %%
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch.nn.functional as F
model.eval()
test_acc = 0.0
test_prob_all = []
test_label_all = []
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        x, labels = batch
        x = x.to(device)
        labels = labels.to(device)
        outputs = model(x)
        outputs = F.softmax(outputs, dim = -1)
        _, test_pred = torch.max(outputs, 1)
        test_acc += (test_pred.cpu() == labels.cpu()).sum().item()
        
        test_prob_all.extend(outputs[:, 1].detach().cpu().numpy())
        test_label_all.extend(labels.detach().cpu().numpy())
threshold = 0.12
test_prob_all = np.array(test_prob_all)
test_label_all = np.array(test_label_all)
test_pred_all = (test_prob_all >= threshold).astype(int)
test_auc = roc_auc_score(test_label_all, test_prob_all)
print(f"acc = {(test_acc/len(test_set)):.5f}, auc = {test_auc:.5f}")
cm = confusion_matrix(test_label_all, test_pred_all)
TN, FP, FN, TP = cm.ravel()

# Calculate TPR and TNR
TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Sensitivity / Recall
TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # Specificity
print(f"Confusion Matrix:\n{cm}")
print(f"TPR (Sensitivity): {TPR:.4f}")
print(f"TNR (Specificity): {TNR:.4f}")
