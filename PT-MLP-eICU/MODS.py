# %%
# %cd Sepsis_MODS/pretrained_optimization_structure/
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
# %ls

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

seed = 42                        # Random seed
batch_size = 64                 # Batch size
num_epoch = 300                 # Number of training epochs
learning_rate = 0.00001         # Learning rate

pre_model_path = './eicu_mimic_pre_dead_rate.ckpt'
model_path = './lstm_all_params.ckpt'     # Path to save checkpoint

# %%
def same_seeds(seed):  # Fix random seeds (CPU)
    torch.manual_seed(seed)  # Fix random seeds (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For current GPU
        torch.cuda.manual_seed_all(seed)  # For all GPUs
    np.random.seed(seed)  # Ensure fixed random numbers
    torch.backends.cudnn.benchmark = False  # Can set True if GPU/network fixed
    torch.backends.cudnn.deterministic = True  # Fix network structure

same_seeds(seed)

# %%

sepsis_mimic_train_data = pd.read_csv('../sepsis_mimic_train_data.csv')
sepsis_mimic_test_data = pd.read_csv('../sepsis_mimic_test_data.csv')
sepsis_mimic_valid_data = pd.read_csv('../sepsis_mimic_valid_data.csv')

sepsis_eicu_data = pd.read_csv('../sepsis_eicu_data.csv')

# %%
class SepsisData(Dataset):
    def __init__(self, data, learn_w, lead_w, pred_w):
        x1s = []  # Store non-invasive data
        x2s = []  # Store invasive data
        ys = []
        row_id = list(set(data['stay_id']))
        for i in row_id:
            data_row = data[data['stay_id'] == i]  # Extract data for each patient
            data_row = data_row.sort_values(by='hr')  # Sort by time
            data_row = data_row.reset_index(drop=True)
            end_boundary = data_row.shape[0]  # Default boundary is the whole table if no positive data

            if (end_boundary <= learn_w):  # Skip if window smaller than learning window
                continue
            
            for j in range(end_boundary - learn_w - lead_w):
                label = []
                x_data = data_row.iloc[j: j + learn_w, 1:-1]  # Data within learning window
                x1_data = x_data.loc[x_data.index[-1], ['heart_rate', 'mbp', 'temperature', 'spo2', 'resp_rate', 'sbp', 'dbp']]
                x2_data = x_data.drop(['heart_rate', 'mbp', 'temperature', 'spo2', 'resp_rate', 'sbp', 'dbp'], axis=1).iloc[-1, :]

                positive_flag_6 = 0
                for k in range(j + learn_w + lead_w, min(j + lead_w + learn_w + 6, data_row.shape[0])):  # Label positive within prediction window
                    if (data_row.iloc[k, -1] == 1):
                        positive_flag_6 = 1
                        label.append(1)
                        break
                if (positive_flag_6 == 0):
                    label.append(0)

                positive_flag_12 = 0
                for k in range(j + learn_w + lead_w, min(j + lead_w + learn_w + 12, data_row.shape[0])):  # Label positive within 12-hour window
                    if (data_row.iloc[k, -1] == 1):
                        positive_flag_12 = 1
                        label.append(1)
                        break
                if (positive_flag_12 == 0):
                    label.append(0)

                positive_flag_24 = 0
                for k in range(j + learn_w + lead_w, min(j + lead_w + learn_w + 24, data_row.shape[0])):  # Label positive within 24-hour window
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
# train_loader = DataLoader(train_set, batch_size=1024, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_set, batch_size=1024, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2)
# valid_loader = DataLoader(valid_set, batch_size=1024, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2)
eicu_loader = DataLoader(eicu_set, batch_size=1024, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2)

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
    
class LR(nn.Module):
    def __init__(self, x1_input_dim, x1_hidden_dim, x1_num_layers, x2_input_dim, x2_hidden_dim, x2_num_layers, hidden_dim, num_layers, output_dim):
        super(LR, self).__init__()
        self.x1_hidden_dim = x1_hidden_dim
        self.x1_num_layers = x1_num_layers
        self.x1_fc_input = nn.Sequential(
            BasicBlock(x1_input_dim, x1_hidden_dim),
            *[BasicBlock(x1_hidden_dim, x1_hidden_dim) for _ in range(x1_num_layers -1)],
        ) 

        self.x2_fc = nn.Sequential(
            BasicBlock(x2_input_dim, x2_hidden_dim),
            *[BasicBlock(x2_hidden_dim, x2_hidden_dim) for _ in range(x2_num_layers -1)],
        )
        self.fc = nn.Sequential(
            BasicBlock(x1_hidden_dim + x2_hidden_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(num_layers -1)],
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x1, x2):  # x1: non-invasive, x2: invasive
        out_x1 = self.x1_fc_input(x1)
        out_x2 = self.x2_fc(x2)
        out = torch.cat((out_x1, out_x2), dim=1)
        out = self.fc(out)
        return out

# %%
pre_model = LR(x1_input_dim=7, x1_hidden_dim=8, x1_num_layers=8, x2_input_dim=34, x2_hidden_dim=64, x2_num_layers=10, hidden_dim=64, num_layers=8, output_dim=2).to(device)
pre_model.load_state_dict(torch.load(pre_model_path))
print(pre_model.modules)

# %%
# (Rest of the code remains the same—no Chinese comments below this point)
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
        self.pre_x1_fc_input = pre_model.x1_fc_input
        self.pre_x2_fc = pre_model.x2_fc
        # self.pre_fc = pre_model.fc  # Do not import last layers, output dimension mismatch
        num_features = pre_model.x1_hidden_dim + 64
        self.fc = nn.Sequential(
            BasicBlock(num_features, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x1, x2):
        out_x1 = self.pre_x1_fc_input(x1)
        out_x2 = self.pre_x2_fc(x2)
        out = torch.cat((out_x1, out_x2), dim=1)
        out = self.fc(out)
        return out

# %%
# The following lines were used to freeze pretrained layers
# for param in pre_model.fc_input.parameters():
#     param.requires_grad = False  # Freeze LSTM layers
# for param in pre_model.gru.parameters():
#     param.requires_grad = False  # Freeze LSTM layers
# for param in pre_model.gln.parameters():
#     param.requires_grad = False  # Freeze LSTM layers

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
model = MyModel(hidden_dim=32, output_dim=3).to(device)
model.load_state_dict(torch.load(model_path))

# %%
from sklearn.metrics import roc_auc_score, confusion_matrix
torch.backends.cudnn.enabled = True
model.eval()
test_acc = 0.0
threshold = 0.15
test_prob_all_6 = []
test_label_all_6 = []
test_prob_all_12 = []
test_label_all_12 = []
test_prob_all_24 = []
test_label_all_24 = []
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

# Compute TPR and TNR
TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Sensitivity / Recall
TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # Specificity
print(f"Confusion Matrix:\n{cm}")
print(f"TPR (Sensitivity): {TPR:.4f}")
print(f"TNR (Specificity): {TNR:.4f}")
