# als_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ALSHybridAttentionClassifier(nn.Module):
    """
    Dual-branch LSTM network with cross-attention and logical circuit breakers
    designed to detect progressive fine-motor and gross-motor neurodegeneration.
    """
    def __init__(self, hidden_dim=32, dropout_rate=0.5):
        super(ALSHybridAttentionClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # ==========================================
        # BRANCH A: Fine-Motor Kinetics (Features 0-2)
        # ==========================================
        self.branch_A_lstm = nn.LSTM(input_size=3, hidden_size=hidden_dim, 
                                     batch_first=True, dropout=dropout_rate if dropout_rate > 0 else 0)
        
        # ==========================================
        # BRANCH B: Gross-Motor Mobility (Features 3-5)
        # ==========================================
        self.branch_B_lstm = nn.LSTM(input_size=3, hidden_size=hidden_dim, 
                                     batch_first=True, dropout=dropout_rate if dropout_rate > 0 else 0)
        
        # ==========================================
        # CROSS-ATTENTION & FUSION LAYERS
        # ==========================================
        # Projects Branch A and Branch B into a shared attention space
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # Logical Gating: Decides how much to trust Branch A vs Branch B
        self.gate_layer = nn.Linear(hidden_dim * 2, 1)
        
        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1) # Outputs raw logit
        )

    def forward(self, x):
        """
        Input x shape: (Batch, Window_Size, Total_Features) -> (B, 7, 6)
        """
        # Split the tensor into modalities
        x_fine = x[:, :, 0:3]  # Branch A inputs: [HT, FT, SD]
        x_gross = x[:, :, 3:6] # Branch B inputs: [Act%, Inact%, Ind%]
        
        # Extract temporal features via LSTMs
        # Output shapes: (Batch, Window, Hidden_Dim)
        out_A, (h_A, _) = self.branch_A_lstm(x_fine)
        out_B, (h_B, _) = self.branch_B_lstm(x_gross)
        
        # Cross-Attention: Let Fine-Motor (Query) attend to Gross-Motor (Key/Value)
        attn_out, _ = self.cross_attention(query=out_A, key=out_B, value=out_B)
        
        # Extract the final timestep representation for both branches
        final_A = out_A[:, -1, :]       # Shape: (Batch, Hidden_Dim)
        final_attn = attn_out[:, -1, :] # Shape: (Batch, Hidden_Dim)
        
        # Concatenate features for gating and classification
        fused_features = torch.cat((final_A, final_attn), dim=1) # Shape: (Batch, Hidden_Dim * 2)
        
        # Logical Gating Mechanism
        gate = torch.sigmoid(self.gate_layer(fused_features)) # Value between 0 and 1
        gated_features = fused_features * gate
        
        # Calculate final logit prediction
        logits = self.classifier(gated_features) # Shape: (Batch, 1)
        
        # ==========================================
        # CLINICAL CIRCUIT BREAKER
        # ==========================================
        # If the highest absolute Z-score in the entire 7-day window is < 0.8 std deviations, 
        # force the network to output a highly negative logit (predicting healthy).
        # This acts as a hard filter against natural, non-pathological daily noise.
        max_z_scores, _ = torch.max(torch.abs(x), dim=1) # Max over time
        max_z_scores_global, _ = torch.max(max_z_scores, dim=1, keepdim=True) # Max over features
        
        # Create a boolean mask (True if anomaly exists, False if purely noise)
        circuit_breaker_mask = (max_z_scores_global > 0.8).float()
        
        # Apply mask: Valid signals pass through, weak noise is driven to -10.0 (near 0 probability)
        final_logits = (logits * circuit_breaker_mask) + (-10.0 * (1 - circuit_breaker_mask))
        
        return final_logits

# =====================================================================
# VERIFICATION SUITE
# =====================================================================
if __name__ == "__main__":
    print("="*80)
    print("VERIFYING PYTORCH HYBRID ATTENTION ARCHITECTURE")
    print("="*80)
    
    # Instantiate the model
    model = ALSHybridAttentionClassifier(hidden_dim=32)
    print("[SUCCESS] Model instantiated.")
    print(model)
    
    # Create a dummy batch tensor matching the Dataset DataLoader output
    # Shape: (Batch Size = 16, Temporal Window = 7 Days, Features = 6)
    dummy_input = torch.randn(16, 7, 6)
    
    # Execute a forward pass
    output_logits = model(dummy_input)
    output_probs = torch.sigmoid(output_logits)
    
    print("\nArchitecture Matrix Audit:")
    print(f" * Input Tensor Shape:  {dummy_input.shape}")
    print(f" * Output Logits Shape: {output_logits.shape}")
    print(f" * Final Probabilities: {output_probs.shape}")
    
    # Verify outputs are successfully gated and flattened
    assert output_logits.shape == (16, 1), "Error: Final classification dimension mismatch."
    print("\n[SUCCESS] Forward pass and mathematical tensor fusion completed without errors.")
    print("="*80)