#!/usr/bin/env python3
"""
FRNN Path B - Pure PyTorch Demo Implementation
This is a standalone demo that runs WITHOUT C++ extensions.
Use this to test the hardened training workflow.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class FRNNPathB(nn.Module):
    """Pure PyTorch implementation of FRNN Path B for demonstration."""
    
    def __init__(self, input_dim, output_dim, num_states, memory_dim, 
                 hidden_dim, bank_size, use_gumbel=False, tau=1.0, 
                 stickiness=0.0, ema_decay=0.99):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = num_states
        self.Dm = memory_dim
        self.H = hidden_dim
        self.bank_size = bank_size
        self.use_gumbel = use_gumbel
        self.tau = tau
        self.stickiness = stickiness
        self.ema_decay = ema_decay
        
        # Transition weights
        self.Wtr = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.btr = nn.Parameter(torch.zeros(hidden_dim))
        
        # Mode selection
        self.Wms = nn.Parameter(torch.randn(hidden_dim, num_states) * 0.01)
        self.bms = nn.Parameter(torch.zeros(num_states))
        
        # Memory matrix
        self.M = nn.Parameter(torch.randn(num_states, memory_dim) * 0.01)
        
        # Readout
        self.Wrd = nn.Parameter(torch.randn(memory_dim, output_dim) * 0.01)
        self.brd = nn.Parameter(torch.zeros(output_dim))
        
        # Bank buffers (not parameters)
        self.register_buffer('bank_keys', torch.zeros(bank_size, memory_dim))
        self.register_buffer('bank_vals', torch.zeros(bank_size, memory_dim))
        self.register_buffer('bank_ptr', torch.zeros(1, dtype=torch.long))
        
    def forward(self, x: torch.Tensor, prev_mode: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through FRNN Path B.
        
        Args:
            x: [B, S, Di] input sequence
            prev_mode: [B, K] previous mode (optional)
            
        Returns:
            y: [B, S, Do] output sequence
            modes: [B, S, K] mode selections
        """
        B, S, Di = x.shape
        device = x.device
        
        # Initialize previous mode
        if prev_mode is None:
            prev_mode = torch.zeros(B, self.K, device=device)
            prev_mode[:, 0] = 1.0  # Start in mode 0
        
        outputs = []
        modes = []
        
        for t in range(S):
            x_t = x[:, t, :]  # [B, Di]
            
            # Transition: x_t -> hidden
            v_t = torch.relu(x_t @ self.Wtr + self.btr)  # [B, H]
            
            # Mode selection logits
            logits = v_t @ self.Wms + self.bms  # [B, K]
            
            # Add stickiness
            if self.stickiness > 0:
                logits = logits + self.stickiness * prev_mode
            
            # Mode selection (deterministic or stochastic)
            if self.use_gumbel and self.training:
                # Gumbel-Softmax (stochastic)
                m_t = torch.nn.functional.gumbel_softmax(logits, tau=self.tau, hard=True)
            else:
                # Argmax (deterministic)
                indices = logits.argmax(dim=-1)  # [B]
                m_t = torch.nn.functional.one_hot(indices, self.K).float()  # [B, K]
            
            # Memory lookup
            mem_t = m_t @ self.M  # [B, Dm]
            
            # Readout
            y_t = mem_t @ self.Wrd + self.brd  # [B, Do]
            
            # Update bank (EMA)
            if self.training:
                ptr = self.bank_ptr.item()
                self.bank_keys[ptr] = mem_t.mean(dim=0).detach()
                self.bank_vals[ptr] = v_t.mean(dim=0).detach()
                self.bank_ptr[0] = (ptr + 1) % self.bank_size
            
            outputs.append(y_t)
            modes.append(m_t)
            prev_mode = m_t
        
        y = torch.stack(outputs, dim=1)  # [B, S, Do]
        mode_seq = torch.stack(modes, dim=1)  # [B, S, K]
        
        return y, mode_seq


def train_demo():
    """Demonstration training loop."""
    print("="*80)
    print("FRNN PATH B - PURE PYTORCH DEMO")
    print("="*80)
    print()
    
    # Config
    B = 4
    S = 256
    Di = 128
    Do = 64
    K = 64
    Dm = 256
    H = 256
    bank_size = 32
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: B={B}, S={S}, Di={Di}, Do={Do}, K={K}, Dm={Dm}")
    print()
    
    # Determinism
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("Determinism: TF32 disabled, deterministic algorithms enabled")
    print()
    
    # Model
    model = FRNNPathB(
        input_dim=Di,
        output_dim=Do,
        num_states=K,
        memory_dim=Dm,
        hidden_dim=H,
        bank_size=bank_size,
        use_gumbel=False,  # Deterministic
        tau=1.0,
        stickiness=0.1,
        ema_decay=0.99
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    
    print("Training for 100 steps...")
    print()
    
    for step in range(100):
        # Random data
        x = torch.randn(B, S, Di, device=device)
        target = torch.randint(0, 2, (B, S), device=device)
        
        # Forward
        y, modes = model(x)
        
        # Loss (simple binary classification)
        loss = nn.functional.cross_entropy(
            y.reshape(-1, Do), 
            target.reshape(-1).long() % Do
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient check
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        # Update
        optimizer.step()
        
        if step % 10 == 0:
            mode_entropy = -(modes.mean(dim=(0,1)) * torch.log(modes.mean(dim=(0,1)) + 1e-8)).sum()
            print(f"Step {step:3d}: loss={loss.item():.4f}, "
                  f"grad_norm={total_norm:.4f}, "
                  f"mode_entropy={mode_entropy.item():.4f}")
    
    print()
    print("[OK] Training complete!")
    print()
    
    # Save weights
    weights = {
        'Wtr': model.Wtr.detach().cpu().numpy(),
        'btr': model.btr.detach().cpu().numpy(),
        'Wms': model.Wms.detach().cpu().numpy(),
        'bms': model.bms.detach().cpu().numpy(),
        'M': model.M.detach().cpu().numpy(),
        'Wrd': model.Wrd.detach().cpu().numpy(),
        'brd': model.brd.detach().cpu().numpy(),
    }
    
    config = {
        'input_dim': Di,
        'output_dim': Do,
        'num_states': K,
        'memory_dim': Dm,
        'hidden_dim': H,
        'bank_size': bank_size,
        'use_gumbel': False,
        'gumbel_temp': 1.0,
        'gumbel_hard': True,
        'stickiness': 0.1,
        'use_bank': True,
        'ema_decay': 0.99,
    }
    
    np.savez('frnn_demo_weights.npz', config=config, **weights)
    print("[OK] Saved weights to: frnn_demo_weights.npz")
    print()
    
    # Test determinism
    print("Testing determinism...")
    model.eval()
    x_test = torch.randn(2, 64, Di, device=device)
    
    with torch.no_grad():
        y1, _ = model(x_test)
        y2, _ = model(x_test)
    
    diff = (y1 - y2).abs().max().item()
    print(f"Max difference between runs: {diff:.6e}")
    
    if diff < 1e-6:
        print("[OK] Deterministic! (diff < 1e-6)")
    else:
        print("[WARN] Non-deterministic (diff >= 1e-6)")
    
    print()
    print("="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print()
    print("This demonstrates the FRNN Path B architecture in pure PyTorch.")
    print("For production deployment, use the C++ extensions with train_hardened.py")
    print()


if __name__ == "__main__":
    train_demo()
