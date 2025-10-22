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
hidden_dim = 64
num_layers = 6
output_dim = 2

seed = 42                        # Random seed
batch_size = 64                 # Batch size
num_epoch = 300                   # Number of training epochs
learning_rate = 0.00001          # Learning rate

model_pre_path = './eicu_mimic_pre_dead_rate.ckpt'     # Path to save checkpoint (i.e., the location where the model saving function is called below)

# %%
def same_seeds(seed): # Fix random seed (CPU)
    torch.manual_seed(seed) # Fix random seed (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # Set for current GPU
        torch.cuda.manual_seed_all(seed)  # Set for all GPUs
    np.random.seed(seed)  # Ensure fixed random numbers when using random function subsequently
    torch.backends.cudnn.benchmark = False # GPU and network structure fixed, can be set to True
    torch.backends.cudnn.deterministic = True # Fix network structure

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
        self.x1s = []
        self.x2s = []
        self.ys = []
        self.learn_w = learn_w
        self.lead_w = lead_w
        result = data.groupby('stay_id').apply(
            self.datapressing, meta=(None, 'object')
        )
        with ProgressBar():
            batches = result.compute()
        for batch in batches:
            if (len(batch) != 3):
                continue
            x1_batch, x2_batch, y_batch = batch
            self.x1s.extend(x1_batch)
            self.x2s.extend(x2_batch)
            self.ys.extend(y_batch)
        self.x1 = torch.stack(self.x1s)
        self.x2 = torch.stack(self.x2s)
        self.y = torch.stack(self.ys)


    def datapressing(self, group):
        data_row = group.sort_values(by='hr').reset_index(drop=True)
        data_row = data_row[['stay_id', 'heart_rate', 'mbp', 'temperature', 'spo2', 'resp_rate', 'sbp', 'dbp', 'hr', 'gender', 'age', 'pao2', 'hematocrit', 'wbc', 'creatinine', 'bun', 'sodium', 'albumin', 'bilirubin', 'glucose', 'ph', 'pco2', 'gcs', 'pao2fio2ratio', 'platelet', 'pt', 'potassium', 'epinephrine', 'norepinephrine', 'dopamine', 'dobutamine', 'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'alt', 'ast', 'baseexcess', 'chloride', 'totalco2', 'lactate', 'free_calcium', 'fio2', 'death_label']]
        data_row = data_row.to_numpy()
        end_boundary = len(data_row)
        if end_boundary <= self.learn_w:
            return []

        batch_x1, batch_x2, batch_y = [], [], []
        for j in range(end_boundary - self.learn_w - self.lead_w + 1):
            x_data = data_row[j:j + self.learn_w, 1:-1]
            x1_data = x_data[:, :7]
            x2_data = x_data[-1, 7:]
            positive_flag = int(data_row[j + self.learn_w - 1, -1])
            batch_x1.append(torch.tensor(x1_data, dtype=torch.float32))
            batch_x2.append(torch.tensor(x2_data, dtype=torch.float32))
            batch_y.append(torch.tensor(positive_flag, dtype=torch.long))
        return batch_x1, batch_x2, batch_y

    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.y[index]

    def __len__(self):
        return len(self.x1)


# %%
train_set = DeathData(train_data, learn_w, lead_w, pred_w)
test_set = DeathData(test_data, learn_w, lead_w, pred_w)
valid_set = DeathData(valid_data, learn_w, lead_w, pred_w)

# %%
# train_loader = DataLoader(train_set, batch_size=8192, shuffle=True, pin_memory=True, num_workers=16)
test_loader = DataLoader(test_set, batch_size=8192, shuffle=True, pin_memory=True, num_workers=16)
# valid_loader = DataLoader(valid_set, batch_size=8192, shuffle=True, pin_memory=True, num_workers=16)

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
            *[BasicBlock(x2_hidden_dim, x2_hidden_dim) for _ in range(x2_num_layers -1)],
        )
        self.fc = nn.Sequential(
            BasicBlock(x1_hidden_dim + x2_hidden_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(num_layers -1)],
        )
        self.last_fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x1, x2):  # x1 non-invasive, x2 invasive
        out_x1 = self.x1_fc_input(x1)
        h0 = torch.zeros(self.x1_num_layers, out_x1.size(0), self.x1_hidden_dim).requires_grad_().to(device)
        out_x1, hn = self.gru(out_x1, h0.detach())
        out_x1 = self.gln(out_x1)
        out_x1 = self.x1_fc_output(out_x1[:, -1, :])
        out_x2 = self.x2_fc(x2)
        out = torch.cat((out_x1, out_x2), dim=1)
        out = self.fc(out)
        out = self.last_fc(out)
        return out

# %%

model = GRU(x1_input_dim=7, x1_hidden_dim=32, x1_num_layers=8, x2_input_dim=34, x2_hidden_dim=64, x2_num_layers=10, hidden_dim=64, num_layers=8, output_dim=2).to(device)
criterion = nn.CrossEntropyLoss()
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
    
    train_prob_all = []
    train_label_all = []
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

        acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()
        #auc = roc_auc_score(labels.detach(), outputs.cpu().numpy(), multi_class='ovr', average='macro')
        train_loss.append(loss.item())
        train_accs.append(acc.detach().item())
        #train_aucs.append(auc.detach().item())
        train_prob_all.extend(outputs[:, 1].detach().cpu().numpy())
        train_label_all.extend(labels.detach().cpu().numpy())
        

    mean_train_acc = sum(train_accs) / len(train_accs)
    mean_train_loss = sum(train_loss)/len(train_loss)
    #mean_train_auc = sum(train_aucs) / len(train_aucs)
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
            x1, x2, labels = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)
            outputs = model(x1, x2)

            loss = criterion(outputs, labels)

            acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()
            #auc = roc_auc_score(labels.detach(), outputs.detach(), multi_class='ovr', average='macro')
            valid_loss.append(loss.item())
            valid_accs.append(acc.detach().item())
            #valid_aucs.append(auc.detach().item())
            valid_prob_all.extend(outputs[:, 1].detach().cpu().numpy())
            valid_label_all.extend(labels.detach().cpu().numpy())
    
    mean_valid_loss = sum(valid_loss) / len(valid_loss)
    mean_valid_acc = sum(valid_accs) / len(valid_accs)
    valid_auc = roc_auc_score(valid_label_all, valid_prob_all)
    print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {mean_valid_loss:.5f}, acc = {mean_valid_acc:.5f}, auc = {valid_auc:.5f}")
    
    # Save models
    if valid_auc > best_auc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), model_pre_path) # only save best to prevent output memory exceed error
        best_auc = valid_auc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvement {patience} consecutive epochs, early stopping")
            break



# %%
model = GRU(x1_input_dim=7, x1_hidden_dim=32, x1_num_layers=8, x2_input_dim=34, x2_hidden_dim=64, x2_num_layers=10, hidden_dim=64, num_layers=8, output_dim=2).to(device)
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
        x1, x2, labels = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)
        outputs = model(x1, x2)
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