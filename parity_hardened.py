# ============================================================
# Parity Test (ATen vs. Fused CUDA) — PATH B HARDENED
#
# Critical Fixes Applied:
# - Weight and input contiguity enforcement
# - Strict output dtype and layout checks
# - Deterministic path verification
# - Repetition test (bitwise)
# - Tiered diagnostic guidance
# ============================================================

import torch
import numpy as np
import os
import sys
from typing import Dict

# ============================================================
# FIX 8: STRICT DETERMINISM
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision("high")

print("="*80)
print("PARITY TEST ENVIRONMENT")
print("="*80)
print(f"TF32 matmul:          {torch.backends.cuda.matmul.allow_tf32}")
print(f"TF32 cudnn:           {torch.backends.cudnn.allow_tf32}")
print(f"Deterministic mode:   enabled")
print("="*80 + "\n")

# ============================================================
# IMPORT MODULES
# ============================================================
try:
    import frnn_aten as fa
except ImportError:
    print("ERROR: `frnn_aten` not found.")
    print("Build: cd frnn_aten_trainable/frnn_aten && python setup.py build_ext --inplace")
    sys.exit(1)

try:
    import frnn_fused_v4 as fte
except ImportError:
    print("ERROR: `frnn_fused_v4` not found.")
    print("Build: cd frnn_fused_v4_bindings && cmake -B build && cmake --build build -j")
    sys.exit(1)

# ============================================================
# VERIFY DETERMINISTIC PATH FIX
# ============================================================

def verify_deterministic_path() -> bool:
    """
    Verify that the argmax fix is applied in C++ backend.
    Tests: logits.argmax(-1) → [B] shape for one_hot input.
    """
    print("="*80)
    print("VERIFICATION: Deterministic Path Implementation")
    print("="*80)
    print("Testing: logits.argmax(-1) produces correct shape for one_hot")
    print("")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        C = fa.Config()
        C.input_dim = 4
        C.output_dim = 2
        C.K = 8
        C.Dm = 16
        C.H = 16
        C.tau = 1.0
        C.hard = True
        C.use_gumbel = False  # Force deterministic path
        C.stickiness = 0.0
        C.use_bank = False
        C.bank_size = 4
        C.ema_decay = 0.99
        
        W = fa.Weights()
        W.M = torch.randn(C.K, C.Dm, device=device, dtype=torch.float32) * 0.02
        W.Wtr = torch.randn(C.input_dim + C.Dm, C.K, device=device, dtype=torch.float32) * 0.02
        W.btr = torch.zeros(C.K, device=device, dtype=torch.float32)
        W.Wrd = torch.randn(C.Dm, C.output_dim, device=device, dtype=torch.float32) * 0.02
        W.brd = torch.zeros(C.output_dim, device=device, dtype=torch.float32)
        W.bank_k_proj = torch.randn(C.Dm, C.Dm, device=device, dtype=torch.float32) * 0.02
        W.bank_q_proj = torch.randn(C.input_dim, C.Dm, device=device, dtype=torch.float32) * 0.02
        W.bank_weight = torch.ones(C.Dm, device=device, dtype=torch.float32)
        W.bank_keys = torch.zeros(C.bank_size, C.Dm, device=device, dtype=torch.float32)
        W.bank_vals = torch.zeros(C.bank_size, C.Dm, device=device, dtype=torch.float32)
        
        model = fa.FRNNATen(C)
        x = torch.randn(2, 4, C.input_dim, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            y, _ = model.forward(x, W, None)
        
        assert y.shape == (2, 4, C.output_dim), f"Shape mismatch: {y.shape}"
        assert torch.isfinite(y).all(), "Output contains NaN/Inf"
        
        print("[OK] Deterministic path verification PASSED")
        print(f"    Output shape: {y.shape} (correct)")
        print(f"    Outputs finite: yes")
        print("")
        return True
        
    except RuntimeError as e:
        error_str = str(e)
        if "one_hot" in error_str or "dimension" in error_str or "shape" in error_str:
            print("[FAIL] Argmax fix NOT applied in C++ backend!")
            print(f"    Error: {error_str}")
            print("")
            print("SOLUTION:")
            print("  In frnn_aten.cpp, replace deterministic path with:")
            print("    auto indices = logits.argmax(-1);  // [B] tensor")
            print("    m_t = at::one_hot(indices, C.K).to(logits.dtype());")
            print("")
            return False
        else:
            print(f"[ERROR] Unexpected error: {error_str}")
            raise
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# ATEN FORWARD
# ============================================================

def run_aten(C_aten: fa.Config, W_aten: fa.Weights, x: torch.Tensor) -> np.ndarray:
    """Run ATen (trainable) implementation."""
    print("Running ATen (trainable) version...")
    model = fa.FRNNATen(C_aten)
    
    x = x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
    
    # FIX: Zero bank buffers
    W_aten.bank_keys.zero_()
    W_aten.bank_vals.zero_()
    print("  [OK] Bank buffers zeroed")
    
    with torch.no_grad():
        y, _ = model.forward(x, W_aten, None)
    
    y_np = y.detach().cpu().numpy().astype(np.float32)
    print(f"  Output shape: {y_np.shape}, dtype: {y_np.dtype}")
    return y_np


# ============================================================
# CUDA FORWARD
# ============================================================

def run_cuda(C_dict: Dict, W_dict: Dict[str, np.ndarray], x_np: np.ndarray) -> np.ndarray:
    """Run CUDA (fused) implementation."""
    print("Running Fused CUDA (deploy) version...")
    
    C_fused = fte.FRNNConfig()
    C_fused.input_dim = int(C_dict["input_dim"])
    C_fused.output_dim = int(C_dict["output_dim"])
    C_fused.num_states = int(C_dict["num_states"])
    C_fused.memory_dim = int(C_dict["memory_dim"])
    C_fused.hidden_dim = int(C_dict["hidden_dim"])
    C_fused.gumbel_temp = float(C_dict["gumbel_temp"])
    C_fused.gumbel_hard = bool(C_dict["gumbel_hard"])
    C_fused.use_gumbel = bool(C_dict.get("use_gumbel", True))
    C_fused.stickiness = float(C_dict["stickiness"])
    C_fused.use_bank = bool(C_dict["use_bank"])
    C_fused.bank_size = int(C_dict["bank_size"])
    C_fused.ema_decay = float(C_dict["ema_decay"])
    
    model = fte.FRNN(C_fused)
    
    if not hasattr(model, "load_weights"):
        print("  ERROR: FRNN has no load_weights method!")
        nan_arr = np.full((x_np.shape[0], x_np.shape[1], C_fused.output_dim), 
                          np.nan, dtype=np.float32)
        return nan_arr
    
    try:
        model.load_weights(W_dict)
        print("  [OK] Weights loaded into CUDA core")
    except Exception as e:
        print(f"  ERROR in load_weights: {e}")
        nan_arr = np.full((x_np.shape[0], x_np.shape[1], C_fused.output_dim), 
                          np.nan, dtype=np.float32)
        return nan_arr
    
    if hasattr(model, "reset_bank"):
        model.reset_bank()
        print("  [OK] Bank buffers reset")
    else:
        print("  WARNING: reset_bank not available")
    
    y = model.forward(x_np)
    y_np = np.asarray(y, dtype=np.float32)
    
    # FIX: Enforce output contiguity and dtype
    if not (y_np.dtype == np.float32 and y_np.flags['C_CONTIGUOUS']):
        print("  [WARN] CUDA output not float32 C-order; forcing")
        y_np = np.ascontiguousarray(y_np.astype(np.float32))
    
    print(f"  Output shape: {y_np.shape}, dtype: {y_np.dtype}, C_contig: {y_np.flags['C_CONTIGUOUS']}")
    return y_np


# ============================================================
# TIERED DIAGNOSTIC GUIDANCE
# ============================================================

def get_diagnostic_guidance(mae: float) -> str:
    """Return diagnostic guidance based on MAE magnitude."""
    if mae < 1e-5:
        return "Excellent parity. Numerical variations within expected range."
    elif mae < 1e-4:
        return "Good parity. Minor numerical differences expected."
    elif mae < 1e-3:
        return "Acceptable parity. Check precision settings match."
    elif mae < 1e-2:
        return "WARNING: Moderate discrepancy. Check:\n" \
               "  - use_gumbel=False on both sides\n" \
               "  - Bank buffers reset before forward\n" \
               "  - RMSNorm epsilon=1e-6 on both sides\n" \
               "  - No TF32 enabled"
    elif mae < 0.1:
        return "ERROR: Large discrepancy. Check:\n" \
               "  - Gumbel sampling disabled on BOTH sides\n" \
               "  - CUDA kernel uses deterministic path\n" \
               "  - Weight shapes match exactly\n" \
               "  - Activation: ReLU (not Sigmoid/GELU)\n" \
               "  - Stickiness order: after ReLU"
    else:
        return "CRITICAL: Outputs diverging. Check:\n" \
               "  - No hidden bias terms in CUDA kernel\n" \
               "  - No fused epilogue operations\n" \
               "  - Readout exactly: y = (m@M) @ Wrd + brd\n" \
               "  - No x_t or v_long in readout"


# ============================================================
# REPETITION TEST
# ============================================================

def run_repetition_test(C_aten: fa.Config, W_aten: fa.Weights, 
                       x: torch.Tensor, num_runs: int = 3) -> bool:
    """Verify bitwise identical outputs across runs."""
    print("\n" + "="*80)
    print("REPETITION TEST: Determinism Validation")
    print("="*80)
    print(f"Running inference {num_runs} times with same input...")
    print("")
    
    outputs = []
    for i in range(num_runs):
        W_aten.bank_keys.zero_()
        W_aten.bank_vals.zero_()
        model = fa.FRNNATen(C_aten)
        
        with torch.no_grad():
            y, _ = model.forward(x, W_aten, None)
        
        outputs.append(y.detach().cpu().numpy().astype(np.float32))
        print(f"  Run {i+1}: mean={outputs[i].mean():+.6e}, norm={np.linalg.norm(outputs[i]):.6e}")
    
    identical = True
    for i in range(1, num_runs):
        max_diff = np.max(np.abs(outputs[0] - outputs[i]))
        if max_diff > 1e-10:
            identical = False
            print(f"  [WARN] Run 1 vs Run {i+1}: max diff = {max_diff:.6e}")
    
    if identical:
        print("\n[OK] All runs produced bitwise identical outputs")
        return True
    else:
        print("\n[WARN] Runs produced different outputs - non-determinism detected!")
        return False


# ============================================================
# MAIN PARITY TEST
# ============================================================

def parity_test(B=4, S=256, Di=128, seed=0):
    """Main parity test with all hardened fixes."""
    print("\n")
    print("#"*80)
    print("# PARITY TEST - PATH B HARDENED")
    print("#"*80)
    print("\n")
    
    # Step 1: Verify C++ fix
    if not verify_deterministic_path():
        print("\n[FAIL] Cannot proceed without C++ fix. Aborting.")
        return False
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Load weights
    WEIGHT_FILE = "frnn_aten_weights.npz"
    if not os.path.exists(WEIGHT_FILE):
        print(f"ERROR: {WEIGHT_FILE} not found")
        print("Run: python train_hardened.py")
        return False
    
    print(f"Loading weights from {WEIGHT_FILE}...")
    data = np.load(WEIGHT_FILE, allow_pickle=True)
    C_dict = data['config'].item()
    W_dict = {k: data[k] for k in data.keys() if k != 'config'}
    print("[OK] Weights loaded\n")
    
    # Build configs
    C_aten = fa.Config()
    C_aten.input_dim = int(C_dict["input_dim"])
    C_aten.output_dim = int(C_dict["output_dim"])
    C_aten.K = int(C_dict["num_states"])
    C_aten.Dm = int(C_dict["memory_dim"])
    C_aten.H = int(C_dict["hidden_dim"])
    C_aten.tau = float(C_dict["gumbel_temp"])
    C_aten.hard = bool(C_dict["gumbel_hard"])
    C_aten.use_gumbel = False  # DETERMINISTIC
    C_aten.stickiness = float(C_dict["stickiness"])
    C_aten.use_bank = bool(C_dict["use_bank"])
    C_aten.bank_size = int(C_dict["bank_size"])
    C_aten.ema_decay = float(C_dict["ema_decay"])
    
    C_dict["use_gumbel"] = False
    
    # Build weights with FIX: explicit contiguity
    W_aten = fa.Weights()
    for name, arr in W_dict.items():
        if hasattr(W_aten, name):
            t = torch.from_numpy(arr).to(device=device, dtype=torch.float32).contiguous()
            setattr(W_aten, name, t)
    W_aten.bank_keys = torch.zeros(C_aten.bank_size, C_aten.Dm, device=device, dtype=torch.float32)
    W_aten.bank_vals = torch.zeros(C_aten.bank_size, C_aten.Dm, device=device, dtype=torch.float32)
    
    # FIX: Build input with explicit contiguity
    x_torch = torch.randn(B, S, Di, device=device, dtype=torch.float32).contiguous()
    x_np = x_torch.detach().cpu().numpy().astype(np.float32)
    x_np = np.ascontiguousarray(x_np)
    
    # Config validation
    print("="*80)
    print("CONFIG VALIDATION")
    print("="*80)
    assert C_aten.use_gumbel == False, "[FAIL] ATen use_gumbel must be False"
    assert C_dict["use_gumbel"] == False, "[FAIL] CUDA use_gumbel must be False"
    print("[OK] use_gumbel = False on both sides")
    assert C_aten.K == C_dict["num_states"], "[FAIL] num_states mismatch"
    assert C_aten.Dm == C_dict["memory_dim"], "[FAIL] memory_dim mismatch"
    print("[OK] Key dimensions match")
    print("="*80 + "\n")
    
    # Repetition test
    run_repetition_test(C_aten, W_aten, x_torch)
    
    # Forward passes
    print("\n" + "="*80)
    print("FORWARD PASSES")
    print("="*80 + "\n")
    
    yA = run_aten(C_aten, W_aten, x_torch)
    print()
    yC = run_cuda(C_dict, W_dict, x_np)
    
    # Check for NaN
    if np.isnan(yC).any():
        print("\n[FAIL] CUDA forward produced NaN. See errors above.")
        return False
    
    # Verify shapes match
    assert yA.shape == yC.shape, f"Shape mismatch: {yA.shape} vs {yC.shape}"
    yA = yA.astype(np.float32, copy=False)
    yC = yC.astype(np.float32, copy=False)
    
    # Parity metrics
    print("\n" + "="*80)
    print("PARITY METRICS")
    print("="*80)
    
    print("\nATen output statistics:")
    print(f"  Mean:    {yA.mean():+.6e}")
    print(f"  Std:     {yA.std():+.6e}")
    print(f"  L2 norm: {np.linalg.norm(yA):.6e}")
    print(f"  Min/Max: {yA.min():+.6e} / {yA.max():+.6e}")
    
    print("\nCUDA output statistics:")
    print(f"  Mean:    {yC.mean():+.6e}")
    print(f"  Std:     {yC.std():+.6e}")
    print(f"  L2 norm: {np.linalg.norm(yC):.6e}")
    print(f"  Min/Max: {yC.min():+.6e} / {yC.max():+.6e}")
    
    # FIX 1: Correct error metrics
    mae = np.mean(np.abs(yA - yC))
    max_ae = np.max(np.abs(yA - yC))
    assert np.isfinite(mae), "MAE is not finite!"
    assert np.isfinite(max_ae), "Max AE is not finite!"
    
    print("\nError metrics:")
    print(f"  Mean Absolute Error: {mae:.6e}")
    print(f"  Max Absolute Error:  {max_ae:.6e}")
    print(f"  Relative Error:      {mae / (np.linalg.norm(yA) + 1e-10):.6e}")
    
    # Result
    print("\n" + "="*80)
    print("PARITY TEST RESULT")
    print("="*80)
    
    threshold = 1e-3
    if mae < threshold:
        print(f"[OK] SUCCESS: Parity test passed!")
        print(f"     MAE = {mae:.6e} < {threshold}")
        print(f"\n     Diagnostic: {get_diagnostic_guidance(mae)}")
        return True
    else:
        print(f"[FAIL] Parity test failed!")
        print(f"     MAE = {mae:.6e} >= {threshold}")
        print(f"\n     Diagnostic: {get_diagnostic_guidance(mae)}")
        return False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if not os.path.exists("frnn_aten_weights.npz"):
        print("ERROR: frnn_aten_weights.npz not found")
        print("Run: python train_hardened.py")
        sys.exit(1)
    
    success = parity_test(B=4, S=256, Di=128, seed=0)
    sys.exit(0 if success else 1)
