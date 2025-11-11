# CRITICAL C++ FIX REQUIRED

## Location
`frnn_aten_trainable/frnn_aten/frnn_aten.cpp`

## The Problem
The deterministic path (when `use_gumbel=False`) uses incorrect argmax extraction that causes shape mismatches.

## Current Code (BROKEN)
```cpp
// In the deterministic branch:
auto max_result = logits.max(-1);
auto indices = std::get<1>(max_result);  // WRONG: Returns tuple element
m_t = at::one_hot(indices, C.K).to(logits.dtype());
```

**Issue**: `max(-1)` returns a tuple `(values, indices)`. Using `std::get<1>` extracts the indices, but they may have incorrect shape or type for `one_hot`.

## Fixed Code (CORRECT)
```cpp
// In the deterministic branch:
auto indices = logits.argmax(-1);  // [B] tensor of indices
m_t = at::one_hot(indices, C.K).to(logits.dtype());  // [B, K] one-hot
```

**Why it works**: `argmax(-1)` directly returns a `[B]` tensor of indices with correct shape for `one_hot`.

## How to Apply

1. **Locate the deterministic path** in `frnn_aten.cpp`:
   ```cpp
   if (!C.use_gumbel) {
       // Deterministic: argmax
       // FIND THIS SECTION
   }
   ```

2. **Replace the broken code** with the fixed version above

3. **Rebuild**:
   ```bash
   cd frnn_aten_trainable/frnn_aten
   python setup.py build_ext --inplace
   ```

4. **Verify**:
   ```bash
   python parity_hardened.py
   # Should show: [OK] Deterministic path verification PASSED
   ```

## Verification
The `parity_hardened.py` script includes `verify_deterministic_path()` which will **fail fast** if this fix is not applied:

```
[FAIL] Argmax fix NOT applied in C++ backend!
    Error: <shape mismatch error>

SOLUTION:
  In frnn_aten.cpp, replace deterministic path with:
    auto indices = logits.argmax(-1);  // [B] tensor
    m_t = at::one_hot(indices, C.K).to(logits.dtype());
```

## Impact
- **Without fix**: Shape errors, runtime crashes, or silent numerical errors
- **With fix**: Deterministic path works correctly, parity test passes

## Status Check
Run this to verify the fix is applied:
```bash
python -c "
import torch
import frnn_aten as fa

C = fa.Config()
C.use_gumbel = False
C.K = 8
C.input_dim = 4
C.output_dim = 2
C.Dm = 16
C.H = 16

# This will fail if fix not applied
try:
    model = fa.FRNNATen(C)
    print('[OK] C++ fix appears to be applied')
except:
    print('[FAIL] C++ fix NOT applied')
"
```

## Priority
**CRITICAL** - Must be applied before running any tests or training.
