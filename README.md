# Passive Digital Phenotyping Pipeline for Early-Stage ALS Detection

An end-to-end machine learning and behavioral telemetry pipeline designed to passively monitor fine-motor and gross-motor metrics via smartphone sensors for the early detection and classification of Amyotrophic Lateral Sclerosis (ALS)[cite: 3, 6, 9].

---

## 🏗️ Architectural Overview

The system features a dual-branch hybrid deep learning architecture built in PyTorch, structured to isolate neurodegenerative trajectories while safeguarding user privacy through local-first processing and strict regulatory design[cite: 5, 9].

* **Branch A (Fine-Motor Kinetics):** Processes high-frequency typing dynamics and touch pressure metrics (Features 0–2: Hold Time, Flight Time, Stroke Deviation)[cite: 3, 9].
* **Branch B (Gross-Motor Mobility):** Evaluates macro-behavioral shifts and spatial autonomy (Features 3–5: Active %, Inactive/Sedentary %, Independence/Mobility %)[cite: 3, 9].
* **Cross-Attention & Gating Mechanism:** Projects both branches into a shared attention space where fine-motor features query gross-motor indicators, managed by a dynamic logical gating layer and classification head.
* **Clinical Circuit Breaker:** Acts as a hard filter against natural, non-pathological daily noise by evaluating absolute Z-scores across the temporal window before allowing disease classification triggers[cite: 9].

---

## 📂 Repository & File Structure

```text
.
├── als_model.py                # Dual-branch LSTM model with cross-attention & circuit breaker
├── data_generator.py           # Synthetic cohort timeline generator grounded in clinical baselines
├── calibration_engine.py       # Longitudinal baseline calibration engine and Z-score normalizer
├── als_dataset.py              # AR(1) temporal smoothing & patient-isolated PyTorch data loaders
├── train_model.py              # Phase IV training loop, evaluation metrics, and checkpointing
├── simulation_run.py           # Pipeline runner to simulate continuous 180-day patient timelines
├── deploy_transfer_learning.py # Phase V protocol for parameter freezing & transfer learning audit
└── best_als_model.pth          # Saved pre-trained model weights checkpoint
Module Descriptionsals_model.py: Defines the PyTorch ALSHybridAttentionClassifier architecture, incorporating dual-branch LSTMs, cross-attention fusion, logical gating, and a clinical circuit breaker[cite: 9].data_generator.py: Generates grounded synthetic multi-cohort timelines (Healthy, Upper-Limb, Lower-Limb, and Simultaneous) using quadratic degradation mathematics (tau) post-onset.  calibration_engine.py: Manages the silent baseline calibration window, computes individualized baseline parameters ($\mu$ and $\sigma$), and transforms raw telemetry into static Z-scores.  als_dataset.py: Applies AR(1) temporal autocorrelation smoothing, builds PyTorch Dataset wrappers, and splits cohorts strictly at the patient level to eliminate data leakage across overlapping sliding windows.  train_model.py: Executes the training loop using dynamically weighted BCEWithLogitsLoss, AdamW optimization, learning rate decay, and early stopping.  simulation_run.py: Executes end-to-end clinical timeline observation simulations over a 180-day period.  deploy_transfer_learning.py: Implements Phase V deployment protocol to freeze recurrent feature extractors and fine-tune classification heads for live data ingestion.  best_als_model.pth: Pre-trained PyTorch state dictionary checkpoint.  ⚙️ Execution Instructions1. Requirements & SetupEnsure Python 3.10+ and PyTorch are installed in your environment:Bashpython -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib scikit-learn
2. Verify Baseline Synthetic GenerationRun data_generator.py to inspect synthetic trajectory plots across healthy and clinical cohorts:  Bashpython data_generator.py
3. Run Pipeline SimulationExecute simulation_run.py to simulate a single patient moving through 30 days of silent baseline calibration followed by active monitoring:  Bashpython simulation_run.py
4. Train the Neural NetworkTrain the dual-branch hybrid attention classifier across all patient cohorts:  Bashpython train_model.py
5. Execute Deployment & Transfer Learning AuditRun the deployment script to verify sub-network layer freezing and gradient isolation:  Bashpython deploy_transfer_learning.py
🛡️ Data Leakage Prevention & PrivacyPatient-Level Splitting: Dataset window extraction occurs after patient profiles are randomly assigned to train or validation splits, ensuring zero overlap of patient identities across train/val sets.  Regulatory Design: Telemetry parameters use temporal deviation metrics rather than continuous GPS speed or coordinate tracking to comply with IRB privacy guidelines.
