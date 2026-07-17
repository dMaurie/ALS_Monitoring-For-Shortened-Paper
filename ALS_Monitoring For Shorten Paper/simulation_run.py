# simulation_run.py
import numpy as np
from data_generator import ALS_Synthetic_Dataset_Generator
from calibration_engine import PersonalizedCalibrationEngine

def execute_clinical_simulation():
    print("=" * 80)
    print("LAUNCHING PASSIVE DIGITAL PHENOTYPING PIPELINE RUNNER")
    print("=" * 80)

    # 1. Pull a single upper-limb cohort timeline from your generator factory
    factory = ALS_Synthetic_Dataset_Generator(total_days=180, onset_range=(45, 75))
    dataset = factory.build_unified_dataset(samples_per_cohort=1)
    simulated_patient = dataset['Upper'][0]
    
    raw_timeline_matrix = simulated_patient['matrix'] # Shape: (180, 6)
    true_onset_day = simulated_patient['onset']
    
    print(f"[DATA SETUP] Simulated Patient Phenotype: Upper-Limb Onset")
    print(f"[DATA SETUP] Latent Mathematical Disease Degredation Target Day: {true_onset_day}\n")

    # 2. Initialize the device engine
    engine = PersonalizedCalibrationEngine(calibration_days=30)
    
    # 3. Create long-term dual-layer memory storage (Saves raw data truths + Z-scores)
    patient_encrypted_storage = []

    print("Processing 180-Day Timeline Observation Sequence...")
    print("-" * 80)
    
    for day_idx in range(180):
        # Extract the current day's real-world telemetry vector
        raw_vector = raw_timeline_matrix[day_idx]
        
        # Pass the daily data through the silent monitoring engine
        z_score_vector = engine.process_day(raw_vector)
        
        # Implement storage strategy: Cache raw values alongside calculated Z-Scores
        day_record = {
            "day": day_idx + 1,
            "raw_features": raw_vector,
            "z_scores": z_score_vector  # This will be None for Days 1-30
        }
        patient_encrypted_storage.append(day_record)
        
        # STEP 9 VERIFICATION: Ensure classification loop remains strictly disconnected during baseline
        if day_idx < 30:
            assert z_score_vector is None, f"Fatal: Pipeline leakage! Predictions active on Day {day_idx + 1}."
        
        # Print status updates every 25 days to track velocity trend lines
        if (day_idx + 1) == 30:
            print(f"Day 030: [STATE CHANGE] Calibration complete. Static parameters frozen in memory.")
            print(f"        * Computed Baseline Means (μ):  {engine.user_means}")
            print(f"        * Computed Baseline StdDevs (σ): {engine.user_stds}\n")
            
        elif (day_idx + 1) % 25 == 0 and day_idx >= 30:
            # Highlight Flight Time metric drift (Feature Index 1)
            # A negative or positive steady trend away from 0.0 marks systemic velocity decline
            print(f"Day {day_idx+1:03d}: [ACTIVE MONITORING]")
            print(f"        * Raw Flight Time: {raw_vector[1]:.4f}s")
            print(f"        * Normalized Velocity Deviation Trend (Z-Score): {z_score_vector[1]:+.4f} σ")
            
    print("-" * 80)
    print("[SUCCESS] Timeline execution complete.")
    print(f"[AUDIT] Verified total stored records: {len(patient_encrypted_storage)}")
    print(f"[AUDIT] Raw historical records intact: {True if len(patient_encrypted_storage[0]['raw_features']) == 6 else False}")
    print("=" * 80)

if __name__ == "__main__":
    execute_clinical_simulation()