import torch
import torch.nn as nn
import torch.nn.functional as F

class ICUFusionModel(nn.Module):
    """
    Multimodal Fusion model for Predicting ICU Speech/Swallow Dysfunction.
    Inputs:
    - Physiological time-series (SpO2, RR, Airway Pressure)
    - Acoustic embeddings (Speech, Swallow, Cough)
    """
    def __init__(self, physio_dim=8, acoustic_dim=128, hidden_dim=64):
        super(ICUFusionModel, self).__init__()
        
        # Physiological Branch (e.g., LSTM for vitals)
        self.physio_rnn = nn.LSTM(input_size=physio_dim, hidden_size=hidden_dim, 
                                  num_layers=2, batch_first=True, dropout=0.2)
        
        # Acoustic Branch (e.g., MLP for pre-extracted embeddings)
        self.acoustic_net = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion Layer
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Final Risk Prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, physio_seq, acoustic_feat):
        # physio_seq shape: (batch, seq_len, physio_dim)
        # acoustic_feat shape: (batch, acoustic_dim)
        
        # Process physiological sequence
        _, (h_n, _) = self.physio_rnn(physio_seq)
        physio_out = h_n[-1] # Take last hidden state
        
        # Process acoustic features
        acoustic_out = self.acoustic_net(acoustic_feat)
        
        # Concatenate fusion
        combined = torch.cat((physio_out, acoustic_out), dim=1)
        fused = F.relu(self.fusion_layer(combined))
        
        # Output probability
        risk_score = self.classifier(fused)
        return risk_score

if __name__ == "__main__":
    # Example test
    model = ICUFusionModel()
    test_physio = torch.randn(1, 24, 8) # 24 hours of 8 vital signs
    test_acoustic = torch.randn(1, 128) # Pre-extracted acoustic embedding
    
    out = model(test_physio, test_acoustic)
    print(f"Predicted Risk Score: {out.item():.4f}")
