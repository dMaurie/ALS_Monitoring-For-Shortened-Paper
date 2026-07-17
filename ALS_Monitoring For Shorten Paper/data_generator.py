import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# STEP 3: LATENT PROGRESSION MATH (Helper Function)
# =====================================================================
def calculate_tau(t, onset_day, total_days):
    """
    Calculates the latent disease progression variable (tau).
    Locked at 0.0 before clinical onset, accelerates quadratically afterwards.
    """
    if t < onset_day:
        return 0.0
    else:
        return ((t - onset_day) / (total_days - onset_day)) ** 2


# =====================================================================
# STEP 7: HEALTHY CONTROL COHORT GENERATOR
# =====================================================================
def generate_healthy_day(t, onset_day=999, total_days=180):
    """Generates a day of telemetry for a healthy control (No progression)."""
    # Fine-Motor Dynamics (Step 1 Baseline)
    ht = np.random.normal(0.10, 0.012) 
    ft = np.random.normal(0.65, 0.06)
    sd = np.random.normal(22.72, 2.5)
    
    # =====================================================================
    # GROUNDED HEALTHY CONTROL BASES (Matthews 2021 & NHAPS 2001)
    # =====================================================================
    # act: NHAPS active locomotion/ambulatory time averages ~11.3% of the waking day
    act = np.random.normal(11.3, 2.5)
    
    # inact: Matthews et al. reports a mean sedentary window of ~9.8 hours/day 
    # for age-matched cohorts, which maps to ~61.3% of a 16-hour waking day
    inact = np.random.normal(61.3, 4.5)
    
    # ind: Spatial navigation independence is unconstrained in controls (high & tight)
    ind = np.random.normal(95.0, 1.5)
    
    return [ht, ft, sd, np.clip(act, 0.0, 100.0), np.clip(inact, 0.0, 100.0), np.clip(ind, 0.0, 100.0)]


# =====================================================================
# STEP 4: UPPER-LIMB COHORT GENERATOR
# =====================================================================
def generate_upper_limb_day(t, onset_day, total_days=180):
    """Fine-motor typing kinetics degrade; gross-motor remains healthy."""
    tau = calculate_tau(t, onset_day, total_days)
        
    # Fine-Motor Dynamics (degrades)
    ht = np.random.normal(0.10 + (0.01 * tau), 0.01)     
    ft = np.random.normal(0.65 + (0.15 * tau), 0.05)     
    sd = np.random.normal(22.72 + (15.67 * tau), 2.0)   
    
    # =====================================================================
    # GROUNDED HEALTHY CONTROL BASES (Matthews 2021 & NHAPS 2001)
    # =====================================================================
    # act: NHAPS active locomotion/ambulatory time averages ~11.3% of the waking day
    act = np.random.normal(11.3, 2.5)
    
    # inact: Matthews et al. reports a mean sedentary window of ~9.8 hours/day 
    # for age-matched cohorts, which maps to ~61.3% of a 16-hour waking day
    inact = np.random.normal(61.3, 4.5)
    
    # ind: Spatial navigation independence is unconstrained in controls (high & tight)
    ind = np.random.normal(95.0, 1.5)
    
    return [ht, ft, sd, np.clip(act, 0.0, 100.0), np.clip(inact, 0.0, 100.0), np.clip(ind, 0.0, 100.0)]


# =====================================================================
# STEP 5: LOWER-LIMB COHORT GENERATOR
# =====================================================================
def generate_lower_limb_day(t, onset_day, total_days=180):
    """Gross-motor mobility degrades; fine-motor typing remains healthy."""
    tau = calculate_tau(t, onset_day, total_days)
        
    # Fine-Motor Dynamics (healthy)
    ht = np.random.normal(0.10, 0.01)
    ft = np.random.normal(0.65, 0.05)
    sd = np.random.normal(22.72, 2.0)
    
    # =====================================================================
    # GROUP B GROSS-MOTOR DECLINE MODELING (HomeSenseALS Group B Triggers)
    # =====================================================================
    # Active locomotion plummets as lower-limb weakness or fatigue sets in
    act = np.random.normal(11.3 - (7.5 * tau), 2.5)     
    
    # Sedentary/Stationary time escalates significantly to compensate for loss of mobility
    inact = np.random.normal(61.3 + (18.0 * tau), 4.5)   
    
    # FIXED: Independence/Community mobility collapses as spatial autonomy shrinks
    ind = np.random.normal(95.0 - (25.0 * tau), 1.5)   
    
    return [ht, ft, sd, np.clip(act, 0.0, 100.0), np.clip(inact, 0.0, 100.0), np.clip(ind, 0.0, 100.0)]


# =====================================================================
# STEP 6: SIMULTANEOUS DECLINE COHORT GENERATOR
# =====================================================================
def generate_simultaneous_day(t, onset_day, total_days=180):
    """Both fine-motor and gross-motor metrics degrade simultaneously."""
    tau = calculate_tau(t, onset_day, total_days)
        
    # Fine-Motor Dynamics (degrades)
    ht = np.random.normal(0.10 + (0.01 * tau), 0.01)
    ft = np.random.normal(0.65 + (0.15 * tau), 0.05)
    sd = np.random.normal(22.72 + (15.67 * tau), 2.0)
    
   # =====================================================================
    # GROUP B GROSS-MOTOR DECLINE MODELING (HomeSenseALS Group B Triggers)
    # =====================================================================
    # Active locomotion plummets as lower-limb weakness or fatigue sets in
    act = np.random.normal(11.3 - (7.5 * tau), 2.5)     
    
    # Sedentary/Stationary time escalates significantly to compensate for loss of mobility
    inact = np.random.normal(61.3 + (18.0 * tau), 4.5)   
    
    # FIXED: Independence/Community mobility collapses as spatial autonomy shrinks
    ind = np.random.normal(95.0 - (25.0 * tau), 1.5)
    
    return [ht, ft, sd, np.clip(act, 0.0, 100.0), np.clip(inact, 0.0, 100.0), np.clip(ind, 0.0, 100.0)]


# =====================================================================
# UNIFIED DATASET GENERATOR CLASS
# =====================================================================
class ALS_Synthetic_Dataset_Generator:
    def __init__(self, total_days=360, onset_range=(30, 120)):
        self.total_days = total_days
        self.onset_range = onset_range

    def build_unified_dataset(self, samples_per_cohort=100):
        dataset = {c: [] for c in ['Healthy', 'Upper', 'Lower', 'Simultaneous']}
        for cohort in dataset:
            for _ in range(samples_per_cohort):
                onset_day = np.random.randint(*self.onset_range) if cohort != 'Healthy' else 999
                traj = []
                for t in range(self.total_days):
                    if cohort == 'Healthy': traj.append(generate_healthy_day(t))
                    elif cohort == 'Upper': traj.append(generate_upper_limb_day(t, onset_day, self.total_days))
                    elif cohort == 'Lower': traj.append(generate_lower_limb_day(t, onset_day, self.total_days))
                    elif cohort == 'Simultaneous': traj.append(generate_simultaneous_day(t, onset_day, self.total_days))
                dataset[cohort].append({'onset': onset_day, 'matrix': np.array(traj)})
        return dataset

# =====================================================================
# EXECUTION AND VERIFICATION
# =====================================================================
if __name__ == "__main__":
    generator = ALS_Synthetic_Dataset_Generator()
    data = generator.build_unified_dataset(samples_per_cohort=100)
    
    print("Dataset generated successfully. Plotting trajectory means...")
    
    plt.figure(figsize=(10, 6))
    for cohort, samples in data.items():
        # Average across all patients for the "Active" metric (Index 3)
        mean_traj = np.mean([s['matrix'][:, 3] for s in samples], axis=0)
        plt.plot(mean_traj, label=f"{cohort} Mean Active%")
    
    plt.axvline(x=30, color='r', linestyle='--', label="Min Onset Threshold")
    plt.title("Progression Verification: Active% Metric vs Time")
    plt.legend()
    plt.show()