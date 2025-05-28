# %%
from torch.utils.data import DataLoader, WeightedRandomSampler
from resnet import ResNet2 # Assuming resnet.py contains your ResNet2 model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
import numpy as np
from custom_dataset import FinalDataset, BeatsDataset # Assuming custom_dataset.py is available
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from FocalLoss import FocalLoss # Assuming FocalLoss.py is available
import copy
import itertools # Import itertools for product of hyperparameters

# %%
file_path= "./data/average_beats.hdf5"
dataset2 = BeatsDataset(file_path ,downsample=True,majority_ratio=0.95)

# %%
# --- Hyperparameter Grids ---
learning_rates_grid = [1e-4, 5e-4, 1e-3]
weight_decays_grid = [1e-5, 1e-4, 1e-3]
l1_lambda_grid = [0, 1e-6, 1e-5, 1e-4]

# ResNet parameters
in_channels = 12
out_channels = 64

# Optimizer parameters
momentum = 0.9 # Fixed for AdamW, relevant if using SGD

random_state = 42

# Training parameters
epochs = 200 # Each configuration runs for this many epochs
batch_size = 256

# Early stopping parameters removed:
# patience, count = 10,0

# %%
train_val_idx_2, test_idx_2 = train_test_split(
    np.arange(len(dataset2)),
    test_size=0.15,
    random_state=random_state,
    stratify=dataset2.get_labels()
)
train_val_idx_2.sort()
train_idx_2, val_idx_2 = train_test_split(
    train_val_idx_2,
    test_size=0.17647,  # ~15% of total
    random_state=random_state,
    stratify=dataset2.labels[train_val_idx_2]
)
train_loader_2 = DataLoader(BeatsDataset("./data/average_beats.hdf5", indices = dataset2.indices[train_idx_2]), batch_size=batch_size, shuffle=True)
val_loader_2   = DataLoader(BeatsDataset("./data/average_beats.hdf5", indices = dataset2.indices[val_idx_2]),   batch_size=batch_size, shuffle=False)
test_loader_2  = DataLoader(BeatsDataset("./data/average_beats.hdf5", indices = dataset2.indices[test_idx_2]),  batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# %%
def train_loop(dataloader, model, loss_fn, optimizer, device, l1_lambda=0):
    size = len(dataloader.dataset.indices)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)

        bce_loss = loss_fn(pred, y).mean()
        
        loss = bce_loss
        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = bce_loss + l1_lambda * l1_norm
        
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return total_loss / len(dataloader)


def test_loop(dataloader, model, loss_fn, device, val=False, current_hyperparams_str=""):
    size = len(dataloader.dataset.indices)
    total_loss = 0
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y).mean()
            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    probabilities = torch.sigmoid(all_preds)
    targets = all_targets.int().numpy()

    if val:
        RocCurveDisplay.from_predictions(all_targets, probabilities, name=f"ROC Curve ({current_hyperparams_str})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f'ROC Curve - Best Model ({current_hyperparams_str})')
        plt.savefig(f'roc_curve_RETE_BEATS_BEST_MODEL.png')
        plt.show()
    return total_loss / len(dataloader)

# %%
# --- Grid Search Setup ---
param_grid = list(itertools.product(learning_rates_grid, 
                                   weight_decays_grid, 
                                   l1_lambda_grid))

overall_best_val_loss = float('inf')
best_hyperparams = None
overall_best_model_state = None
all_results = []

# --- Main Grid Search Loop ---
for lr, wd, current_l1_lambda in param_grid:
    hyperparams_str = f"LR_{lr}_WD_{wd}_L1_{current_l1_lambda}"
    print(f"\n\n===== Training with Hyperparameters: {hyperparams_str} =====")

    model = ResNet2(in_channels, out_channels).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # Scheduler patience is for ReduceLROnPlateau, not for early stopping.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) # Adjusted scheduler patience slightly

    train_losses_curr_run = []
    val_losses_curr_run = []
    best_loss_curr_run = float('inf')
    best_model_state_curr_run = None # To store the best model state for this specific configuration

    for epoch in range(epochs): # Will run for all epochs
        train_loss = train_loop(train_loader_2, model, loss_fn, optimizer, device, l1_lambda=current_l1_lambda)
        val_loss = test_loop(val_loader_2, model, loss_fn, device) 
        scheduler.step(val_loss)
        
        train_losses_curr_run.append(train_loss)
        val_losses_curr_run.append(val_loss)
        
        if epoch % 10 == 0 or epoch == epochs - 1 : # Print less frequently
             print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e} | Params: {hyperparams_str}")

        # Check if current validation loss is the best for this configuration run
        if val_loss < best_loss_curr_run:
            best_loss_curr_run = val_loss
            best_model_state_curr_run = copy.deepcopy(model.state_dict())
            # No early stopping counter increment or check
    
    print(f"Finished training for {hyperparams_str}. Best Val Loss for this run: {best_loss_curr_run:.4f} (achieved over {epochs} epochs).")
    all_results.append({
        'lr': lr,
        'wd': wd,
        'l1_lambda': current_l1_lambda,
        'best_val_loss': best_loss_curr_run, # This is the min val_loss for this config
        'epochs_trained': epochs # Always trained for full epochs
    })

    # If the best model from this configuration is better than the overall best so far
    if best_loss_curr_run < overall_best_val_loss:
        overall_best_val_loss = best_loss_curr_run
        best_hyperparams = {'lr': lr, 'wd': wd, 'l1_lambda': current_l1_lambda}
        overall_best_model_state = best_model_state_curr_run # Save the best model state from this run
        print(f"\n*** New Overall Best Validation Loss: {overall_best_val_loss:.4f} with params: {best_hyperparams} ***\n")

# --- End of Grid Search ---

print("\n\n===== Grid Search Completed =====")
sorted_results = sorted(all_results, key=lambda x: x['best_val_loss'])
print("Top 5 results:")
for i, res in enumerate(sorted_results[:5]):
    print(f"{i+1}. Params: { {k:v for k,v in res.items() if k not in ['best_val_loss','epochs_trained']} }, Val Loss: {res['best_val_loss']:.4f}, Epochs: {res['epochs_trained']}")

print(f"\nBest hyperparameters found: {best_hyperparams}")
print(f"Best validation loss: {overall_best_val_loss:.4f}")

if overall_best_model_state is not None:
    # Load the best model state found across all configurations and their epochs
    model.load_state_dict(overall_best_model_state)
    print("\nLoaded overall best model for final testing.")

    final_test_loss = test_loop(test_loader_2, model, loss_fn, device, val=True, current_hyperparams_str=str(best_hyperparams))
    print(f"Final Test Loss with best model ({best_hyperparams}): {final_test_loss:.4f}")

    torch.save(overall_best_model_state, f"model_RETE_BEATS_BEST_GRID_SEARCH_FULL_EPOCHS_95.pth") # Updated filename
    print(f"Saved best model with params: {best_hyperparams}")
else:
    print("No model was successfully trained or no improvement found during grid search.")

print("Grid search and final evaluation completed!")
# %%
# Reload the old model
from torch.utils.data import DataLoader, WeightedRandomSampler
from resnet import ResNet2 # Assuming resnet.py contains your ResNet2 model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay # Added precision_recall_curve and PrecisionRecallDisplay

import numpy as np
from custom_dataset import FinalDataset, BeatsDataset # Assuming custom_dataset.py is available
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from FocalLoss import FocalLoss # Assuming FocalLoss.py is available
import copy
import itertools
file_path= "./data/average_beats.hdf5"
dataset2 = BeatsDataset(file_path ,downsample=True,majority_ratio=0.95)

# --- Hyperparameter Grids ---
learning_rates_grid = [1e-4, 5e-4, 1e-3]
weight_decays_grid = [1e-5, 1e-4, 1e-3]
l1_lambda_grid = [0, 1e-6, 1e-5, 1e-4]

# ResNet parameters
in_channels = 12
out_channels = 64

# Optimizer parameters
momentum = 0.9 # Fixed for AdamW, relevant if using SGD

random_state = 42

# Training parameters
epochs = 200 # Each configuration runs for this many epochs
batch_size = 256

# Early stopping parameters removed:
# patience, count = 10,0
train_val_idx_2, test_idx_2 = train_test_split(
    np.arange(len(dataset2)),
    test_size=0.15,
    random_state=random_state,
    stratify=dataset2.get_labels()
)
train_val_idx_2.sort()
train_idx_2, val_idx_2 = train_test_split(
    train_val_idx_2,
    test_size=0.17647,  # ~15% of total
    random_state=random_state,
    stratify=dataset2.labels[train_val_idx_2]
)
train_loader_2 = DataLoader(BeatsDataset("./data/average_beats.hdf5", indices = dataset2.indices[train_idx_2]), batch_size=batch_size, shuffle=True)
val_loader_2   = DataLoader(BeatsDataset("./data/average_beats.hdf5", indices = dataset2.indices[val_idx_2]),   batch_size=batch_size, shuffle=False)
test_loader_2  = DataLoader(BeatsDataset("./data/average_beats.hdf5", indices = dataset2.indices[test_idx_2]),  batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
old_model = ResNet2(in_channels, out_channels).to(device)
old_model.load_state_dict(torch.load("model_RETE_BEATS_BEST_GRID_SEARCH_FULL_EPOCHS_95.pth"))
old_model.eval()

# Generate predictions and plot ROC curve
all_preds, all_targets = [], []

with torch.no_grad():
    for X, y in test_loader_2:
        X, y = X.to(device), y.to(device)
        pred = old_model(X)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

probabilities = torch.sigmoid(all_preds)
from sklearn.metrics import roc_curve

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(all_targets.numpy(), probabilities.numpy())

# Find the point on the ROC curve with the smallest distance to (0, 1)
distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
min_distance_idx = np.argmin(distances)
optimal_fpr, optimal_tpr = fpr[min_distance_idx], tpr[min_distance_idx]

best_threshold = thresholds[min_distance_idx]
print("Best threshold:", best_threshold)
# Plot the ROC curve and mark the optimal point
RocCurveDisplay.from_predictions(all_targets, probabilities, name=f"ROC Curve BT: {best_threshold:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.scatter(optimal_fpr, optimal_tpr, color='red')

plt.title('ROC Curve')
plt.legend()
plt.savefig('optimalThreshold_RETE_BEATS_BEST_GRID_SEARCH_FULL_EPOCHS_95.png')
plt.show()
precision, recall, pr_thresholds = precision_recall_curve(all_targets.numpy(), probabilities.numpy())

# Plot the Precision-Recall curve
plt.figure(figsize=(10, 5))
PrecisionRecallDisplay.from_predictions(all_targets, probabilities, name=f"Curva PR")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.grid(True)
plt.savefig('PR_Curve_RETE_BEATS_BEST_GRID_SEARCH_FULL_EPOCHS_95.png')
plt.show()

# Optionally, find a "best" threshold from the PR curve by maximizing the F1-score
# This threshold might differ from the ROC-derived one because it directly optimizes
# for a balance between precision and recall.
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10) # Added a small epsilon to prevent division by zero for numerical stability
optimal_f1_idx = np.argmax(f1_scores)
optimal_pr_threshold = pr_thresholds[optimal_f1_idx]
print(f"Best threshold (from PR, maximizing F1-score): {optimal_pr_threshold:.2f}")
print(f"Precision at optimal PR threshold: {precision[optimal_f1_idx]:.2f}")
print(f"Recall at optimal PR threshold: {recall[optimal_f1_idx]:.2f}")
print(f"F1-score at optimal PR threshold: {f1_scores[optimal_f1_idx]:.2f}")

# %%
probabilities_numpy = probabilities.numpy()
y_true_numpy = all_targets.numpy()

target_specificity = 0.9
closest_threshold_specificity = None
min_specificity_diff = float('inf')  # Large initial value

for t in thresholds:
    predicted = (probabilities_numpy > t).astype(int)
    cm = confusion_matrix(y_true_numpy, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    current_specificity = tn / (tn + fp + 1e-10)  # handle zero-division
    specificity_diff = abs(current_specificity - target_specificity)

    if specificity_diff < min_specificity_diff:
        min_specificity_diff = specificity_diff
        closest_threshold_specificity = t

if closest_threshold_specificity is not None:
    print(f"\n--- Metrics with Threshold Closest to {target_specificity:.1f} Specificity ({closest_threshold_specificity:.4f}) ---")
    predicted_closest_specificity = (probabilities_numpy > closest_threshold_specificity).astype(int)
    cm_closest_specificity = confusion_matrix(y_true_numpy, predicted_closest_specificity, labels=[0, 1])
    print("Confusion Matrix:")
    print(cm_closest_specificity)
    report_closest_specificity = classification_report(y_true_numpy, predicted_closest_specificity, digits=4, zero_division=0)
    print("Classification Report:")
    print(report_closest_specificity)

    tn, fp, fn, tp = cm_closest_specificity.ravel()
    actual_specificity = tn / (tn + fp + 1e-10)
    print(f"Actual Specificity at this threshold: {actual_specificity:.4f}")
# %%
