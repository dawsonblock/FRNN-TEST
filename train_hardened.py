# ============================================================
# FRNN ATen Trainable Workflow — PATH B HARDENED
#
# Critical Fixes Applied:
# - Fixed circular import in round-trip validation
# - Cache invalidation with forward pre-hook
# - Gradient validation with bounds checking
# - Weight contiguity enforcement
# - Explicit parameter gradient verification
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Tuple
import sys

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================
NUM_STATES = 64
MEMORY_DIM = 256
HIDDEN_DIM = 256
BANK_SIZE = 32
TAU_INITIAL = 1.5
TAU_FINAL = 0.5
TAU_ANNEAL_STEPS = 0.3

LEARNING_RATE = 2e-4
GRAD_CLIP = 1.0
WEIGHT_INIT_SCALE = 0.02
EMA_DECAY = 0.99
STICKINESS = 0.1

DEFAULT_BATCH_SIZE = 32
DEFAULT_SEQ_LENGTH = 512
DEFAULT_INPUT_DIM = 128
DEFAULT_NUM_STEPS = 500

# ============================================================
# STRICT DETERMINISM & PRECISION
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)
torch.set_float32_matmul_precision("high")

print("="*80)
print("DETERMINISM ENFORCEMENT")
print("="*80)
print(f"TF32 matmul allowed:  {torch.backends.cuda.matmul.allow_tf32}")
print(f"TF32 cudnn allowed:   {torch.backends.cudnn.allow_tf32}")
print(f"Deterministic mode:   enabled")
print("="*80 + "\n")

# ============================================================
# IMPORT ATen EXTENSION
# ============================================================
try:
    import frnn_aten as fa
except ImportError:
    print("ERROR: Could not import `frnn_aten` C++ extension.")
    print("Build first:")
    print("  cd frnn_aten_trainable/frnn_aten")
    print("  python setup.py build_ext --inplace")
    sys.exit(1)

# ============================================================
# ENHANCED TRAINABLE MODULE — HARDENED
# ============================================================

class FRNNTrainableModule(nn.Module):
    """
    Trainable FRNN wrapper around ATen backend.
    
    Hardened against:
    - Cache staleness (forward pre-hook invalidates)
    - Weight contiguity issues
    - Silent gradient failures
    """
    
    _PARAM_NAMES = {
        "Wtr", "btr",
        "Wrd", "brd",
        "bank_k_proj", "bank_q_proj", "bank_weight",
        "M",
    }
    _BUFFER_NAMES = {
        "bank_keys", "bank_vals",
    }

    def __init__(self, C: fa.Config, W: fa.Weights):
        super().__init__()
        self.C = C
        self._aten_model = fa.FRNNATen(C)
        self._cached_weights = None
        self._cache_valid = False

        # Register parameters
        p = {}
        for name in self._PARAM_NAMES:
            t = getattr(W, name)
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"Expected tensor for {name}, got {type(t)}")
            p[name] = nn.Parameter(t.detach().clone())
        self.W_params = nn.ParameterDict(p)

        # Register buffers
        for name in self._BUFFER_NAMES:
            t = getattr(W, name)
            self.register_buffer(name, t.detach().clone(), persistent=False)
        
        # FIX: Register forward pre-hook to always invalidate cache
        self.register_forward_pre_hook(lambda mod, args, kwargs: self._invalidate_cache())
        
        # FIX: Register gradient hooks on parameters
        for p in self.W_params.values():
            p.register_hook(lambda grad: self._invalidate_cache())

    def _invalidate_cache(self):
        """Invalidate weight cache on parameter updates or forward."""
        self._cache_valid = False

    def _rebuild_weights_for_forward(self) -> fa.Weights:
        """Reconstruct Weights with caching."""
        if self._cache_valid and self._cached_weights is not None:
            return self._cached_weights
        
        W = fa.Weights()
        for name, param in self.W_params.items():
            setattr(W, name, param)
        for name in self._BUFFER_NAMES:
            setattr(W, name, getattr(self, name))
        
        self._cached_weights = W
        self._cache_valid = True
        return W

    def forward(self, x: torch.Tensor, state: Optional[fa.State] = None) -> Tuple:
        """Forward pass through ATen backend."""
        W = self._rebuild_weights_for_forward()
        return self._aten_model.forward(x, W, state)

    def get_config(self) -> Dict:
        """Export configuration for parity testing."""
        return {
            "input_dim": self.C.input_dim,
            "output_dim": self.C.output_dim,
            "num_states": self.C.K,
            "memory_dim": self.C.Dm,
            "hidden_dim": self.C.H,
            "gumbel_temp": self.C.tau,
            "gumbel_hard": self.C.hard,
            "use_gumbel": self.C.use_gumbel,
            "stickiness": self.C.stickiness,
            "use_bank": self.C.use_bank,
            "bank_size": self.C.bank_size,
            "ema_decay": self.C.ema_decay,
        }

    @torch.no_grad()
    def export_weights(self) -> Dict:
        """Export trainable weights to numpy (guaranteed C-contiguous)."""
        export = {}
        for name, param in self.W_params.items():
            arr = param.detach().float().cpu().numpy()
            # FIX: Ensure C-contiguous
            export[name] = np.ascontiguousarray(arr)
        return export

    def get_param_stats(self) -> str:
        """Get detailed parameter statistics."""
        stats = []
        for name, param in self.W_params.items():
            p = param.detach()
            stats.append(
                f"{name:15s}: mean={p.mean():+.6e}, std={p.std():+.6e}, "
                f"norm={p.norm():+.6e}, min={p.min():+.6e}, max={p.max():+.6e}"
            )
        return "\n".join(stats)

    def validate_dtype(self) -> bool:
        """Verify all parameters are float32."""
        for name, param in self.W_params.items():
            if param.dtype != torch.float32:
                print(f"[WARN] {name} dtype: {param.dtype} (expected float32)")
                return False
        return True


# ============================================================
# WEIGHT INITIALIZATION
# ============================================================

def init_weights(C: fa.Config, device="cuda") -> fa.Weights:
    """Initialize FRNN weights with documented scaling."""
    W = fa.Weights()
    Din = C.input_dim
    
    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=torch.float32) * WEIGHT_INIT_SCALE
    def zeros(*shape):
        return torch.zeros(*shape, device=device, dtype=torch.float32)
    def ones(*shape):
        return torch.ones(*shape, device=device, dtype=torch.float32)

    W.M             = randn(C.K, C.Dm)
    W.Wtr           = randn(Din + C.Dm, C.K)
    W.btr           = zeros(C.K)
    W.Wrd           = randn(C.Dm, C.output_dim)
    W.brd           = zeros(C.output_dim)
    W.bank_k_proj   = randn(C.Dm, C.Dm)
    W.bank_q_proj   = randn(Din, C.Dm)
    W.bank_weight   = ones(C.Dm)
    W.bank_keys     = zeros(C.bank_size, C.Dm)
    W.bank_vals     = zeros(C.bank_size, C.Dm)
    
    return W


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def run_gradient_sanity_check(model: FRNNTrainableModule) -> bool:
    """
    Comprehensive gradient validation with bounds checking.
    
    Hardened checks:
    - Gradient existence
    - Finiteness
    - Magnitude bounds (0 < norm < 1e6)
    """
    print("\n" + "="*80)
    print("GRADIENT SANITY CHECK — HARDENED")
    print("="*80)
    
    C = model.C
    device = next(model.parameters()).device
    
    model.train()
    
    try:
        x = torch.randn(2, 16, C.input_dim, device=device, 
                       dtype=torch.float32, requires_grad=False)
        y, _ = model(x)
        
        loss = y.sum()
        loss.backward()
        
        # FIX: Hard assertions with bounds checking
        print("\nParameter gradient validation:")
        for name, param in model.W_params.items():
            assert param.grad is not None, f"[FAIL] {name}.grad is None"
            g = param.grad
            assert torch.isfinite(g).all(), f"[FAIL] {name}.grad contains NaN/Inf"
            gn = g.norm().item()
            assert 0.0 < gn < 1e6, f"[FAIL] {name}.grad norm out of range: {gn:.3e}"
            print(f"  [OK] {name:15s}: norm={gn:.6e}, finite={True}")
        
        print(f"\n[OK] All gradients validated")
        print(f"  Loss: {loss.item():.6e}")
        
        model.zero_grad(set_to_none=True)
        
    except AssertionError as e:
        print(f"[FAIL] {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("="*80 + "\n")
    return True


def run_round_trip_validation(model: FRNNTrainableModule, 
                             test_input: torch.Tensor) -> bool:
    """
    Validate export/reload produces identical outputs.
    
    FIX: Remove circular import, inline reload config.
    """
    print("="*80)
    print("ROUND-TRIP EXPORT VALIDATION")
    print("="*80)
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Original output
        y1, _ = model(test_input)
        y1_np = y1.detach().cpu().numpy()
        
        # Export
        config = model.get_config()
        weights = model.export_weights()
        
        print("\nExported weight contiguity:")
        for name, arr in weights.items():
            is_c_contig = arr.flags['C_CONTIGUOUS']
            print(f"  {name:15s}: C-contiguous={is_c_contig}, dtype={arr.dtype}")
        
        # FIX: Reload WITHOUT circular import — inline config rebuild
        C_reload = fa.Config()
        C_reload.input_dim  = int(config["input_dim"])
        C_reload.output_dim = int(config["output_dim"])
        C_reload.K          = int(config["num_states"])
        C_reload.Dm         = int(config["memory_dim"])
        C_reload.H          = int(config["hidden_dim"])
        C_reload.tau        = float(config["gumbel_temp"])
        C_reload.hard       = bool(config["gumbel_hard"])
        C_reload.use_gumbel = bool(config["use_gumbel"])
        C_reload.stickiness = float(config["stickiness"])
        C_reload.use_bank   = bool(config["use_bank"])
        C_reload.bank_size  = int(config["bank_size"])
        C_reload.ema_decay  = float(config["ema_decay"])
        
        # FIX: Force contiguity on weights during reload
        W_reload = fa.Weights()
        for name, arr in weights.items():
            t = torch.from_numpy(arr).to(device=device, dtype=torch.float32).contiguous()
            setattr(W_reload, name, t)
        W_reload.bank_keys = torch.zeros(C_reload.bank_size, C_reload.Dm, device=device, dtype=torch.float32)
        W_reload.bank_vals = torch.zeros(C_reload.bank_size, C_reload.Dm, device=device, dtype=torch.float32)
        
        model_reload = FRNNTrainableModule(C_reload, W_reload).to(device)
        y2, _ = model_reload(test_input)
        y2_np = y2.detach().cpu().numpy()
        
        # Compare
        max_diff = np.max(np.abs(y1_np - y2_np))
        if max_diff < 1e-6:
            print(f"\n[OK] Round-trip validation passed (max diff: {max_diff:.6e})")
            print("="*80 + "\n")
            return True
        else:
            print(f"\n[WARN] Round-trip difference: {max_diff:.6e}")
            print("     Likely: dtype conversion or serialization precision")
            print("="*80 + "\n")
            return False


# ============================================================
# TRAINING LOOP
# ============================================================

def run_training(B=DEFAULT_BATCH_SIZE, S=DEFAULT_SEQ_LENGTH, 
                Di=DEFAULT_INPUT_DIM, steps=DEFAULT_NUM_STEPS, seed=0) -> Optional[FRNNTrainableModule]:
    """
    Run training with hardened validation.
    """
    print("\n" + "="*80)
    print(f"TRAINING LOOP — HARDENED")
    print(f"  Batch size:     {B}")
    print(f"  Sequence len:   {S}")
    print(f"  Input dim:      {Di}")
    print(f"  Steps:          {steps}")
    print("="*80)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    
    print(f"\nDevice: {device}")
    print(f"Mixed precision: {use_amp}\n")
    
    # Build config
    C = fa.Config()
    C.input_dim = Di
    C.output_dim = 2
    C.K = NUM_STATES
    C.Dm = MEMORY_DIM
    C.H = HIDDEN_DIM
    C.tau = TAU_INITIAL
    C.hard = True
    C.use_gumbel = True
    C.stickiness = STICKINESS
    C.use_bank = True
    C.bank_size = BANK_SIZE
    C.ema_decay = EMA_DECAY
    
    print("CONFIG:")
    print(f"  num_states:      {C.K}")
    print(f"  memory_dim:      {C.Dm}")
    print(f"  use_gumbel:      {C.use_gumbel}")
    print(f"  bank_size:       {C.bank_size}\n")
    
    # Initialize model
    W = init_weights(C, device=device)
    model = FRNNTrainableModule(C, W).to(device)
    
    # FIX: Ensure all weights are contiguous before training
    for p in model.parameters():
        p.data = p.data.contiguous()
    
    # Sanity checks
    if not run_gradient_sanity_check(model):
        return None
    
    # Optimizer
    model.train()
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()
    
    print("="*80)
    print("TRAINING")
    print("="*80)
    
    for step in range(steps):
        # Anneal tau
        progress = min(step / max(1, int(steps * TAU_ANNEAL_STEPS)), 1.0)
        model.C.tau = TAU_INITIAL + (TAU_FINAL - TAU_INITIAL) * progress
        
        # Batch
        x = torch.randn(B, S, Di, device=device, dtype=torch.float32).contiguous()
        tgt = torch.randint(0, 2, (B, S), device=device)
        
        # Forward/backward
        with torch.cuda.amp.autocast(enabled=use_amp):
            y, _ = model(x)
            loss = loss_fn(y.reshape(-1, 2), tgt.reshape(-1))
        
        scaler.scale(loss).backward()
        if use_amp:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        
        # Log
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step:5d}/{steps:5d} | loss {loss.item():.4f} | tau {model.C.tau:.3f}")
    
    print("="*80)
    print("[OK] Training complete\n")
    return model


# ============================================================
# EXPORT & SAVE
# ============================================================

def export_and_save(model: FRNNTrainableModule, filename="frnn_aten_weights.npz"):
    """Export with comprehensive validation."""
    print("="*80)
    print("EXPORT & VALIDATION")
    print("="*80)
    
    # Round-trip validation
    test_x = torch.randn(2, 16, model.C.input_dim, device=next(model.parameters()).device)
    run_round_trip_validation(model, test_x)
    
    # Export
    weights = model.export_weights()
    config = model.get_config()
    
    print("EXPORTED WEIGHTS:")
    for name, arr in weights.items():
        c_contig = arr.flags['C_CONTIGUOUS']
        print(f"  {name:15s}: shape={arr.shape}, dtype={arr.dtype}, "
              f"C_contig={c_contig}, mean={arr.mean():+.6e}")
    
    print("\nEXPORTED CONFIG:")
    for key, val in config.items():
        print(f"  {key:20s}: {val}")
    
    # Save
    np.savez_compressed(filename, config=config, **weights)
    print(f"\n[OK] Saved to: {filename}")
    
    # Verify
    print("\nVERIFY SAVED FILE:")
    data = np.load(filename, allow_pickle=True)
    print(f"  Keys: {list(data.keys())}")
    loaded_config = data['config'].item()
    print(f"  use_gumbel: {loaded_config.get('use_gumbel', 'MISSING')}")
    
    print("="*80 + "\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n")
    print("#"*80)
    print("# FRNN ATen Trainable Workflow - PATH B HARDENED")
    print("#"*80)
    print("\n")
    
    model = run_training(B=DEFAULT_BATCH_SIZE, S=DEFAULT_SEQ_LENGTH, 
                        Di=DEFAULT_INPUT_DIM, steps=DEFAULT_NUM_STEPS, seed=0)
    
    if model is not None:
        export_and_save(model, "frnn_aten_weights.npz")
        
        print("\n")
        print("#"*80)
        print("# NEXT STEPS FOR PARITY TESTING")
        print("#"*80)
        print("1. Build CUDA fused bindings:")
        print("   cd frnn_fused_v4_bindings")
        print("   cmake -B build -S . -DBUILD_PYTHON=ON")
        print("   cmake --build build -j")
        print("")
        print("2. Run parity test:")
        print("   python parity_hardened.py")
        print("")
        print("Expected: [OK] SUCCESS: Parity test passed!")
        print("#"*80)
    else:
        print("\n[FAIL] Training failed. Check errors above.")
        sys.exit(1)
