# deploy_transfer_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
from als_model import ALSHybridAttentionClassifier

def execute_deployment_protocol():
    print("=" * 80)
    print("PHASE V: CLINICAL DEPLOYMENT & TRANSFER LEARNING PROTOCOL")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Initialize Architecture & Load Secured Weights
    model = ALSHybridAttentionClassifier(hidden_dim=32, dropout_rate=0.5).to(device)
    
    try:
        model.load_state_dict(torch.load('best_als_model.pth', map_location=device))
        print("[SYSTEM] Pre-trained clinical weights successfully loaded.")
    except FileNotFoundError:
        print("[WARNING] best_als_model.pth not found. Running mock protocol with initialized weights.")

    # 2. Layer-Freezing Protocol
    print("\nExecuting Sub-network Freeze...")
    frozen_layers = ['branch_A_lstm', 'branch_B_lstm', 'cross_attention']
    
    for name, param in model.named_parameters():
        if any(layer in name for layer in frozen_layers):
            param.requires_grad = False
            
    # Verify Freeze
    active_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"[AUDIT] Frozen components: {frozen_layers}")
    print(f"[AUDIT] Trainable parameters limited to: {active_params}")

    # 3. Fine-Tuning Optimizer Implementation
    # Only passes parameters where requires_grad == True
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4 # Conservative LR for fine-tuning
    )
    criterion = nn.BCEWithLogitsLoss()

    # 4. Mock Live Validation Sequence
    print("\n" + "-" * 80)
    print("SIMULATING LIVE HUMAN DATA INGESTION (FINE-TUNING)")
    print("-" * 80)
    
    model.train() # Set to train to allow classification head to adapt
    
    # Simulate an incoming 7-day window from a newly calibrated patient
    mock_live_data = torch.randn(1, 7, 6).to(device) 
    mock_live_label = torch.tensor([1.0]).to(device) # Simulated target
    
    optimizer.zero_grad()
    logits = model(mock_live_data)
    loss = criterion(logits.squeeze(0), mock_live_label)
    loss.backward()
    
    # 5. Gradient Audit (Crucial for Paper Methodology)
    lstm_grad = model.branch_A_lstm.weight_ih_l0.grad
    classifier_grad = model.classifier[0].weight.grad
    
    if lstm_grad is None and classifier_grad is not None:
        print("[SUCCESS] Gradients successfully isolated to classification head.")
        print("[SUCCESS] Pre-trained physiological feature extractors remained frozen.")
    else:
        raise RuntimeError("Fatal: Gradient leakage detected in frozen layers.")
        
    optimizer.step()
    print("\n[PIPELINE READY] System is fully prepared for secure university server deployment.")
    print("=" * 80)

if __name__ == "__main__":
    execute_deployment_protocol()