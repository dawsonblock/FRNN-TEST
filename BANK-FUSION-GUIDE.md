# BANK-TO-READOUT FUSION: OPTIONAL ENHANCEMENT

## What It Does

Fuses the learned memory bank directly into the output computation, enabling **context-aware decision making**.

### Current Architecture
```
State → Mode Selection (m_t) → Memory (M @ m_t) → Readout
y_t = (M @ m_t) @ Wrd + brd
```

### Enhanced Architecture (Bank Fused)
```
State → Mode Selection (m_t) → Memory (M @ m_t) + Bank (v_long) → Readout
y_t = (M @ m_t + v_long) @ Wrd + brd
```

---

## Why Fuse the Bank?

### Current System
- Bank updates via EMA but is only used for future state transitions
- Historical information doesn't directly affect readout
- Context trapped in hidden state

### With Bank Fused
- Historical information directly shapes outputs
- Model can reason over long-term context in real-time
- Better for tasks needing explicit memory recall

---

## Mathematical Formulation

### Current (Path B without fusion)
```
m_t = one_hot(argmax(logits_t))        # Mode selection
M_t = M[m_t]                           # Memory vector (Dm,)
bank_update_t = update_ema(bank, v_t)  # Update memory
y_t = M_t @ Wrd + brd                  # Readout
```

### Enhanced (With bank fusion)
```
m_t = one_hot(argmax(logits_t))        # Mode selection
M_t = M[m_t]                           # Memory vector (Dm,)
bank_update_t = update_ema(bank, v_t)  # Update memory
v_long = bank.mean() or bank[m_t]      # Retrieve context
y_t = (M_t + α * v_long) @ Wrd + brd   # Fused readout
```

Where α is a learnable fusion weight (typically 0.1-0.5).

---

## Implementation Changes

### 1. ATen Layer (frnn_aten.cpp)

Add to readout computation:
```cpp
// Current:
auto y = at::matmul(m_t, M);
y = at::matmul(y, Wrd) + brd;

// Enhanced:
auto y = at::matmul(m_t, M);
auto bank_context = bank_keys.mean(0);  // [Dm]
auto fusion_weight = 0.2;               // Learnable parameter
y = y + fusion_weight * bank_context;   // Fuse bank
y = at::matmul(y, Wrd) + brd;
```

### 2. CUDA Kernel (frnn.cu)

Add bank retrieval to readout:
```cuda
// Current:
y_t = (m_t * M) @ Wrd + brd

// Enhanced:
v_long = bank_mean() or bank[argmax(m_t)]
y_t = ((m_t * M) + alpha * v_long) @ Wrd + brd
```

### 3. Training Loop (train_hardened.py)

Add fusion parameter:
```python
C.bank_fusion_weight = 0.2  # New hyperparameter
C.bank_fusion_mode = "mean"  # or "mode-specific"
```

Register as parameter if learnable:
```python
self.bank_fusion_weight = nn.Parameter(
    torch.tensor(0.2, dtype=torch.float32)
)
```

---

## Training Modifications

### Loss Function (Unchanged)
Standard cross-entropy on y_t remains the same.

### Gradient Flow (Enhanced)
Gradients now flow: loss → y_t → bank_context → bank_update

This means:
- Model learns to store useful information in bank
- Bank updates are guided by output loss
- Feedback loop improves memory quality over time

### Hyperparameter Tuning

```python
# New hyperparameters for fusion
BANK_FUSION_WEIGHT = 0.2        # How much to weight bank in readout
BANK_FUSION_LEARNABLE = True    # Learn the weight or fix it
BANK_FUSION_MODE = "mean"       # "mean", "mode-specific", or "weighted"
```

### Training Tips

1. **Start with fusion_weight = 0.0** (no fusion)
2. **Gradually increase** to 0.1, 0.2, 0.5
3. **Monitor loss** - should improve
4. **If loss increases** - bank is not capturing useful info
5. **Adjust EMA decay** - may need slower/faster bank updates

---

## Use Cases

### 1. Long-Context Reasoning
**Example:** Question answering over streaming text

```
Input stream: "Alice was born in 1990. Bob works in NYC. What year was Alice born?"
Without fusion: Model might forget Alice info by time question arrives
With fusion: Bank stores "Alice: 1990" and recalls it for answer
```

### 2. Stateful Agent Behavior
**Example:** Trading algorithm

```
State history: [bull_market, correction, rally, ...]
Decision: sell_when_volatility_high AND (past_month == bull_market)
Without fusion: History only in hidden state
With fusion: Can directly query "was market bullish recently?"
```

### 3. Pattern Continuation
**Example:** Dialogue consistency

```
Earlier: User mentioned "traveling to Japan"
Later: Respond with knowledge of that trip
Without fusion: Might lose context
With fusion: Bank stores "user_traveling_japan" for reference
```

### 4. Anomaly Detection
**Example:** System monitoring

```
Baseline: Normal behavior patterns stored in bank
Detection: Compare current input against bank vectors
Alert: When divergence exceeds threshold
```

---

## Performance Impact

### Computational Cost
- **Additional compute:** ~5% (one mean/query + fusion)
- **Memory:** +Dm values for context vector
- **Latency:** <0.1ms additional per forward

### Accuracy Impact
- **Expected improvement:** +2-5% on context-heavy tasks
- **No impact on:** Tasks not using historical context
- **Potential regression:** If bank fusion is too strong (tune fusion_weight down)

---

## Backward Compatibility

### Deployment Parity with Fusion
When enabled, parity test must validate with fusion ON:

```python
# parity_hardened.py modifications
C_aten.bank_fusion_weight = 0.2
C_dict["bank_fusion_weight"] = 0.2
# Rest of test unchanged
```

### Optional: Disabled by Default
```python
C.bank_fusion_weight = 0.0  # Disable fusion (original behavior)
```

This maintains parity with original build.

---

## Integration Guide

### Step 1: Modify frnn_aten.cpp Readout
```cpp
// In forward pass, readout section:

// Get bank context
auto bank_context = W.bank_keys.sum(0) / W.bank_keys.size(0);  // Mean

// Fuse into memory
auto fused_mem = at::matmul(m_t, W.M) + 0.2 * bank_context;

// Compute output
auto y = at::matmul(fused_mem, W.Wrd) + W.brd;
```

### Step 2: Update CUDA Kernel
Similar change in frnn.cu readout computation.

### Step 3: Update Train Script
Add fusion weight constant and update config export.

### Step 4: Re-validate Parity
```bash
python train_hardened.py  # Trains with fusion
python parity_hardened.py # Tests with fusion enabled
# Should still get MAE < 1e-3
```

---

## When NOT to Use Fusion

- ❌ Tasks that don't need historical context
- ❌ When model is already at ceiling performance
- ❌ When inference latency is critical (even 5% might matter)
- ❌ In highly constrained embedded environments

---

## Advanced Options

### 1. Mode-Specific Bank Retrieval
Retrieve bank vector closest to current mode:
```python
v_long = bank_keys[argmax(m_t)]  # Use bank for selected mode
```

### 2. Learned Fusion Gate
Make fusion weight dynamic per timestep:
```python
alpha_t = sigmoid(linear(state))  # Learned when to use bank
y_t = ((M @ m_t) + alpha_t * v_long) @ Wrd + brd
```

### 3. Multi-Head Bank Attention
Attend over multiple bank entries:
```python
bank_attention = softmax(query @ bank_keys.T)
v_long = bank_attention @ bank_vals
```

---

## Recommended Deployment

1. **Train without fusion first** - establish baseline
2. **Test fusion as add-on** - evaluate improvement
3. **If +2-5% gain** - keep it, update parity test
4. **If no improvement** - disable (set weight=0)
5. **Monitor in production** - ensure bank quality

---

## Summary

Bank-to-readout fusion:
- ✅ Enables context-aware decision making
- ✅ ~5% computational overhead
- ✅ Better for long-context tasks
- ✅ Optional enhancement (backward compatible)
- ✅ Production-ready when trained and validated

Use when your task benefits from explicit long-term memory in decisions.
