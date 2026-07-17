# als_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from data_generator import ALS_Synthetic_Dataset_Generator
from calibration_engine import PersonalizedCalibrationEngine

class ALSDigitalPhenotypeDataset(Dataset):
    """
    Ingests synthetic cohorts, applies AR(1) temporal smoothing, normalizes via 
    PersonalizedCalibrationEngine, and extracts 7-day sliding windows with dynamic 
    temporal labeling to prevent pre-onset noise contamination.
    """
    def __init__(self, cohort_dataset, alpha=0.75, window_size=7):
        self.window_sequences = []
        self.window_labels = []
        self.alpha = alpha
        self.window_size = window_size

        print(f"[PYTORCH] Initializing dataset wrapper...")
        print(f"[PYTORCH] Extracting {self.window_size}-day sliding windows with dynamic labeling...")
        
        for cohort_name, patient_list in cohort_dataset.items():
            for patient_profile in patient_list:
                raw_matrix = patient_profile['matrix']     # Shape: (180, 6)
                true_onset = patient_profile['onset']      # Day the decline actually begins
                
                # --- AR(1) TEMPORAL AUTOCORRELATION LAYER ---
                smoothed_matrix = np.zeros_like(raw_matrix)
                smoothed_matrix[0] = raw_matrix[0]
                for t in range(1, len(raw_matrix)):
                    smoothed_matrix[t] = (self.alpha * smoothed_matrix[t-1]) + ((1 - self.alpha) * raw_matrix[t])
                
                # --- PASSIVE CALIBRATION ENGINE LAYER ---
                engine = PersonalizedCalibrationEngine(calibration_days=30)
                patient_z_sequence = []
                
                for day_vector in smoothed_matrix:
                    z_scores = engine.process_day(day_vector)
                    if z_scores is None:
                        patient_z_sequence.append(np.zeros(6))
                    else:
                        patient_z_sequence.append(z_scores)
                
                full_z_matrix = np.array(patient_z_sequence)
                
                # --- SLIDING WINDOW EXTRACTION & DYNAMIC LABELING ---
                # Yields 174 overlapping windows per patient
                for start_idx in range(len(full_z_matrix) - self.window_size + 1):
                    end_idx = start_idx + self.window_size
                    window_tensor = full_z_matrix[start_idx:end_idx, :]
                    
                    # Dynamic Label: 1.0 only if the window ends AFTER the true onset day
                    label = 1.0 if end_idx >= true_onset else 0.0
                    
                    self.window_sequences.append(window_tensor)
                    self.window_labels.append(label)
            
        # Convert aggregated lists into unified PyTorch tensors
        self.X = torch.tensor(np.array(self.window_sequences), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.window_labels), dtype=torch.float32)
        
        print(f"[PYTORCH] Tensor generation complete. Normalized Input Shape: {self.X.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_clinical_data_loaders(batch_size=64, train_ratio=0.8, sensitivity_bias=1.60):
    """
    Constructs overlapping window data loaders for training and validation,
    ensuring strict patient-level isolation to prevent data leakage.
    Now exports cohort-specific validation loaders for granular AUC tracking.
    """
    print("\n" + "="*80)
    print("PREPARING PYTORCH DATA PIPELINE GENERATION")
    print("="*80)
    
    generator = ALS_Synthetic_Dataset_Generator(total_days=180)
    cohort_dataset = generator.build_unified_dataset(samples_per_cohort=100)
    
    train_cohort_dataset = {}
    val_cohort_dataset = {}
    
    np.random.seed(42)
    
    for cohort_name, patient_list in cohort_dataset.items():
        np.random.shuffle(patient_list)
        split_idx = int(len(patient_list) * train_ratio)
        
        train_cohort_dataset[cohort_name] = patient_list[:split_idx]
        val_cohort_dataset[cohort_name] = patient_list[split_idx:]
    
    # ---------------------------------------------------------
    # GENERATE ISOLATED DATASETS
    # ---------------------------------------------------------
    print("[PYTORCH] Generating Training Dataset...")
    # Updated alpha to 0.65 to allow more high-frequency signal jitter through
    train_dataset = ALSDigitalPhenotypeDataset(train_cohort_dataset, alpha=0.60, window_size=7)
    
    print("[PYTORCH] Generating Global Validation Dataset...")
    val_dataset = ALSDigitalPhenotypeDataset(val_cohort_dataset, alpha=0.60, window_size=7)
    
    print("[PYTORCH] Generating Cohort-Specific Validation Datasets...")
    # Isolate Fine-Motor Cohorts (Upper & Simultaneous) vs Healthy
    val_fine_motor_cohorts = {
        'Healthy': val_cohort_dataset['Healthy'], 
        'Upper': val_cohort_dataset['Upper'], 
        'Simultaneous': val_cohort_dataset['Simultaneous']
    }
    val_fine_dataset = ALSDigitalPhenotypeDataset(val_fine_motor_cohorts, alpha=0.60, window_size=7)
    
    # Isolate Gross-Motor Cohorts (Lower) vs Healthy
    val_gross_motor_cohorts = {
        'Healthy': val_cohort_dataset['Healthy'], 
        'Lower': val_cohort_dataset['Lower']
    }
    val_gross_dataset = ALSDigitalPhenotypeDataset(val_gross_motor_cohorts, alpha=0.60, window_size=7)
    
    # ---------------------------------------------------------
    # BUILD DATALOADERS
    # ---------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_fine_loader = DataLoader(val_fine_dataset, batch_size=batch_size, shuffle=False)
    val_gross_loader = DataLoader(val_gross_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate dynamic imbalance weight with sensitivity bias
    num_neg = (train_dataset.y == 0.0).sum().item()
    num_pos = (train_dataset.y == 1.0).sum().item()
    pos_weight_val = (num_neg / max(1, num_pos)) * sensitivity_bias
    
    print(f"\n[PIPELINE SUCCESS] Data Loaders built successfully.")
    print(f" -> Training Batches: {len(train_loader)} (Total Samples: {len(train_dataset)})")
    print(f" -> Global Val Batches: {len(val_loader)}")
    print(f" -> Fine-Motor Val Batches: {len(val_fine_loader)}")
    print(f" -> Gross-Motor Val Batches: {len(val_gross_loader)}")
    print(f" -> Effective pos_weight for Training: {pos_weight_val:.4f} (Bias: {sensitivity_bias})")
    print("="*80 + "\n")
    
    return train_loader, val_loader, val_fine_loader, val_gross_loader, pos_weight_val

if __name__ == "__main__":
    # Updated to catch the new pos_weight_val variable
    train_loader, val_loader, pos_weight_val = prepare_clinical_data_loaders()
    sample_batch_X, sample_batch_y = next(iter(train_loader))
    print("Data Loader Verification Audit:")
    print(f" * Mini-Batch Input Tensor Shape: {sample_batch_X.shape}")
    print(f" * Mini-Batch Label Tensor Shape: {sample_batch_y.shape}")
    print(f" * Dynamic Imbalance Scalar: {pos_weight_val:.4f}")
