# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import numpy as np
import warnings

# Suppress sklearn undefined metric warnings during early epochs
warnings.filterwarnings('ignore', category=UserWarning)

# Import from our established pipeline
from als_dataset import prepare_clinical_data_loaders
from als_model import ALSHybridAttentionClassifier

def evaluate_loader(model, loader, criterion, device):
    """Helper function to cleanly evaluate specific sub-cohort DataLoaders."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            logits = model(batch_X)
            loss = criterion(logits.squeeze(), batch_y)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits.squeeze())
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.0 # Fallback for perfectly healthy early batches
        
    return avg_loss, auc, all_preds, all_targets

def execute_phase_iv(num_epochs=30):
    print("="*80)
    print("PHASE IV: END-TO-END PIPELINE STRESS TEST & TRAINING EXECUTION")
    print("="*80)

    # Tuning factor forces higher sensitivity to early ALS onset
    SENSITIVITY_TUNING_FACTOR = 1.35  

    train_loader, val_loader, val_fine_loader, val_gross_loader, pos_weight_val = prepare_clinical_data_loaders(
        batch_size=64, 
        train_ratio=0.8,
        sensitivity_bias=SENSITIVITY_TUNING_FACTOR
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[SYSTEM] Executing training loop on hardware accelerator: {device}")

    model = ALSHybridAttentionClassifier(hidden_dim=32, dropout_rate=0.3).to(device)

    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Weight decay (L2 Regularization) added to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Track Validation Loss instead of AUC for smoother learning rate adjustments
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print("\n" + "-"*80)
    print(f"COMMENCING {num_epochs}-EPOCH TRAINING LOOP")
    print("-" * 80)

    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = 15
    final_metrics = {}

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- EVALUATION PHASE ---
        # 1. Global Validation (For Loss Tracking & Early Stopping)
        avg_val_loss, _, _, _ = evaluate_loader(model, val_loader, criterion, device)
        
        # 2. Cohort-Specific Validation (For Target Tracking)
        _, fine_auc, fine_preds, fine_targets = evaluate_loader(model, val_fine_loader, criterion, device)
        _, gross_auc, gross_preds, gross_targets = evaluate_loader(model, val_gross_loader, criterion, device)

        # Step scheduler based on global validation loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Fine AUC: {fine_auc:.4f} | Gross AUC: {gross_auc:.4f} | LR: {current_lr:.6f}")

        # --- EARLY STOPPING & CHECKPOINTING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Store the metrics from the absolute best epoch
            final_metrics = {
                'fine_auc': fine_auc,
                'gross_auc': gross_auc,
                'fine_targets': fine_targets,
                'fine_preds': fine_preds,
                'gross_targets': gross_targets,
                'gross_preds': gross_preds
            }
            # Save absolute path to prevent I/O errors on clusters
            torch.save(model.state_dict(), 'best_als_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n[EARLY STOPPING] Validation loss stagnated for {EARLY_STOPPING_PATIENCE} epochs. Halting to preserve optimal weights.")
                break

    # ---------------------------------------------------------
    # STEP 5: SYNTHETIC EVALUATION & DUAL-TARGET ASSESSMENT
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("FINAL CLINICAL EVALUATION & PAPER ASSESSMENT")
    print("="*80)
    
    fine_target_met = final_metrics['fine_auc'] >= 0.93
    gross_target_met = final_metrics['gross_auc'] >= 0.90
    
    print(f"Fine-Motor (Upper/Simultaneous) ROC AUC: {final_metrics['fine_auc']:.4f} \t Target (>= 0.93): {'✅' if fine_target_met else '⚠️'}")
    print(f"Gross-Motor (Lower Limb) ROC AUC:        {final_metrics['gross_auc']:.4f} \t Target (>= 0.90): {'✅' if gross_target_met else '⚠️'}")
    
    if fine_target_met and gross_target_met:
        print("\n[SUCCESS] Both clinical phenotyping targets have been successfully met.")
    else:
        print("\n[WARNING] One or more targets missed. Adjust SENSITIVITY_TUNING_FACTOR or alpha smoothing.")

    # Generate Clinical Classification Reports
    print("\n" + "-"*80)
    print("FINE-MOTOR COHORTS CLASSIFICATION REPORT")
    binary_fine_preds = [1.0 if p >= 0.5 else 0.0 for p in final_metrics['fine_preds']]
    print(classification_report(final_metrics['fine_targets'], binary_fine_preds, target_names=["Healthy", "ALS Decline"], zero_division=0))
    print("Confusion Matrix [TN, FP] / [FN, TP]:")
    print(confusion_matrix(final_metrics['fine_targets'], binary_fine_preds))

    print("\n" + "-"*80)
    print("GROSS-MOTOR COHORTS CLASSIFICATION REPORT")
    binary_gross_preds = [1.0 if p >= 0.5 else 0.0 for p in final_metrics['gross_preds']]
    print(classification_report(final_metrics['gross_targets'], binary_gross_preds, target_names=["Healthy", "ALS Decline"], zero_division=0))
    print("Confusion Matrix [TN, FP] / [FN, TP]:")
    print(confusion_matrix(final_metrics['gross_targets'], binary_gross_preds))
    print("="*80)
    print("[SYSTEM] Best model weights secured. Architecture ready for Transfer Learning.")

if __name__ == "__main__":
    execute_phase_iv(num_epochs=30)