# ONLINE LEARNING ARCHITECTURE: OPTIONAL ADD-ON

## What It Does

Enables the model to **adapt during deployment** - weights and memory update as new data arrives.

### Key Distinction
```
Capability              | Current | With Online Learning
-----------------------|---------|-----------------------
Internal state updates  | Yes     | Yes (same EMA bank)
Weight adaptation       | No      | Yes (three options)
True continuous learn   | No      | Partial (not full SGD)
Catastrophic forget    | N/A     | Risk (mitigated)
```

---

## Three Implementation Options

### Option 1: Periodic Micro-Finetune (SAFEST)

Run gradient updates on recent data windows.

```python
class OnlineLearner:
    def __init__(self, model, buffer_size=1000):
        self.model = model
        self.buffer = []
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        
    def add_data(self, x, y):
        self.buffer.append((x, y))
        if len(self.buffer) > buffer_size:
            self.buffer.pop(0)
    
    def finetune_step(self):
        """Run one gradient update on recent buffer."""
        if len(self.buffer) < 32:
            return
        
        # Sample minibatch from buffer
        indices = np.random.choice(len(self.buffer), 32)
        x_batch = torch.stack([self.buffer[i][0] for i in indices])
        y_batch = torch.stack([self.buffer[i][1] for i in indices])
        
        # Gradient step
        loss = self.model.loss(x_batch, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()

# Usage
learner = OnlineLearner(model)
for x, y in deployment_stream:
    pred = model(x)  # Inference
    learner.add_data(x, y)  # Buffer
    if step % 100 == 0:
        loss = learner.finetune_step()  # Finetune
```

**Pros:**
- ✅ Safest (uses proven gradient descent)
- ✅ Easy to debug
- ✅ Can add regularization
- ✅ Works with existing infrastructure

**Cons:**
- ❌ Requires storing buffer (memory overhead)
- ❌ Periodic updates (not continuous)
- ❌ Slower adaptation

### Option 2: Hebbian Learning (FASTEST)

Local, non-gradient updates using Hebbian rule.

```python
class HebbianAdapter:
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.lr = learning_rate
        
    def adapt_step(self, x, y, hidden_state):
        """Update weights using Hebbian rule: ΔW ∝ (output - target) * hidden"""
        output = self.model(x)
        error = output - y  # [output_dim]
        
        # Update readout weights: Wrd
        # ΔWrd ∝ error (outer) hidden_state
        with torch.no_grad():
            dWrd = self.lr * torch.outer(error, hidden_state)
            self.model.W_params['Wrd'].data -= dWrd
            
            # Update bias: brd
            self.model.W_params['brd'].data -= self.lr * error
        
        return (error ** 2).mean().item()

# Usage
adapter = HebbianAdapter(model)
for x, y in deployment_stream:
    hidden, pred = model(x, return_hidden=True)
    mse = adapter.adapt_step(x, y, hidden)
```

**Pros:**
- ✅ Fast (no backprop)
- ✅ Local updates (no global loss needed)
- ✅ Lightweight
- ✅ Biologically plausible

**Cons:**
- ❌ Can diverge without care
- ❌ Limited credit assignment
- ❌ May learn spurious patterns
- ❌ Needs careful hyperparameter tuning

### Option 3: Meta-Learning (MOST SOPHISTICATED)

Model learns how to adapt its own weights.

```python
class MetaAdapter:
    def __init__(self, model):
        self.model = model
        # Meta-learner network: learns optimal weight updates
        self.meta_net = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, model.num_params)  # Predict ΔW
        )
        self.meta_optimizer = torch.optim.Adam(self.meta_net.parameters(), lr=1e-4)
        
    def adapt_step(self, x, y, hidden_state):
        """Use meta-network to predict optimal weight update."""
        # Meta-network predicts update
        delta_w = self.meta_net(hidden_state)  # [num_params]
        
        # Apply predicted update
        old_loss = self.model.loss(x, y)
        with torch.no_grad():
            self.model.apply_delta(delta_w)
        new_loss = self.model.loss(x, y)
        
        # Meta-loss: reward decreasing loss
        meta_loss = new_loss - old_loss
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return new_loss.item()

# Usage
adapter = MetaAdapter(model)
for x, y in deployment_stream:
    hidden, pred = model(x, return_hidden=True)
    loss = adapter.adapt_step(x, y, hidden)
```

**Pros:**
- ✅ Learned adaptation strategy
- ✅ Optimal weight updates
- ✅ Handles complex dynamics
- ✅ Can learn when to adapt

**Cons:**
- ❌ Complex (requires training meta-network)
- ❌ Expensive (additional forward/backward)
- ❌ Hard to debug
- ❌ Needs meta-training data

---

## Safety Mechanisms

### 1. Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```
Prevents weight explosion.

### 2. Learning Rate Decay
```python
lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
```
Reduce adaptation strength over time.

### 3. Catastrophic Forgetting Guard
```python
class EWCAdapter:  # Elastic Weight Consolidation
    def __init__(self, model, fisher_matrix):
        self.model = model
        self.fisher = fisher_matrix  # [num_params]
        self.init_weights = {k: v.clone() for k, v in model.named_parameters()}
        
    def regularized_loss(self, x, y):
        loss = self.model.loss(x, y)
        # Add EWC penalty: don't drift too far from original weights
        ewc_loss = 0.5 * sum(
            self.fisher[name] * ((param - self.init_weights[name]) ** 2).sum()
            for name, param in self.model.named_parameters()
        )
        return loss + 0.01 * ewc_loss
```

### 4. Validation Against Baseline
```python
# Keep baseline model, compare regularly
baseline_loss = baseline_model.loss(validation_data)
adaptive_loss = adaptive_model.loss(validation_data)

if adaptive_loss > baseline_loss * 1.1:  # Regressed by >10%
    restore_from_checkpoint()  # Revert bad adaptation
```

---

## Bank Integration

The EMA bank updates automatically during deployment:

```python
# This happens in every forward pass (unchanged)
bank_update = momentum * bank_value + (1 - momentum) * new_value
```

**Combined with weight learning:**
- Bank adapts automatically (EMA)
- Weights adapt via gradient / Hebbian / meta
- Synergy: memory stores relevant info, weights learn to use it

---

## Deployment Checklist

### Pre-Deployment
- [ ] Decide on adaptation strategy (Option 1/2/3)
- [ ] Tune learning rate carefully
- [ ] Test on offline validation set first
- [ ] Set safety mechanisms (clipping, decay, guard)

### During Deployment
- [ ] Monitor loss on recent windows
- [ ] Log weight changes (should be small)
- [ ] Watch for divergence
- [ ] Periodically validate against baseline

### Post-Deployment
- [ ] Archive adapted weights
- [ ] Analyze what changed (for understanding)
- [ ] Collect data for retraining
- [ ] Plan next training cycle

---

## Recommended Configuration

### For Safety (Conservative)
```python
# Option 1: Periodic micro-finetune
buffer_size = 1000
finetune_interval = 100  # Every 100 steps
learning_rate = 1e-5
grad_clip = 0.5
```

### For Speed (Aggressive)
```python
# Option 2: Hebbian
learning_rate = 1e-3
update_mode = "readout_only"  # Only update readout layer
```

### For Sophistication (Balanced)
```python
# Option 3: Meta-learning
meta_learning_rate = 1e-4
meta_network_size = 128
```

---

## When to Use Online Learning

### ✅ Good Cases
- Non-stationary environments (market regimes, robot wear)
- User personalization (dialogue, recommendations)
- Domain shift over time
- Continuous feedback available
- Safety mechanism can be implemented

### ❌ Bad Cases
- Stable environments (no drift)
- No feedback signal
- Cannot tolerate temporary performance drop
- Model must remain frozen for compliance
- High-stakes predictions (medical, legal)

---

## Expected Behavior

### Week 1 (Initial Deployment)
```
Baseline loss:           0.350
With online learning:    0.348  (tiny improvement)
Reason: Model still good, minimal drift
```

### Month 1 (Environment Changes)
```
Baseline loss:           0.425  (degraded)
With online learning:    0.380  (adapted)
Improvement:             +10.5%
```

### Year 1 (Significant Drift)
```
Baseline loss:           0.580  (heavily degraded)
With online learning:    0.410  (adapted significantly)
Improvement:             +29.3%
```

---

## Advanced: Combine with Bank Fusion

When using bank fusion + online learning:

1. **Bank** continuously updates via EMA (no gradient needed)
2. **Readout weights** adapt via online learning
3. **Fusion weight** (α) can also be learned
4. Synergy: Model learns to use bank better as bank improves

```python
# Three levels of adaptation
bank_update = EMA(old_bank, new_value)  # Automatic
Wrd_update = gradient_step(Wrd)         # Online learning
alpha_update = gradient_step(alpha)     # Learn when to fuse
```

---

## Troubleshooting Online Learning

### Loss diverges upward
- Reduce learning rate 10x
- Increase clipping threshold
- Check data quality

### Loss plateaus (no improvement)
- Increase learning rate 2x
- Check if environment actually changed
- Validate adaptation is happening

### Weights drift too far
- Enable EWC penalty
- Reduce learning rate
- Shorter finetune windows

### Model forgets baseline skills
- Increase buffer size
- Mix old + new data
- Reduce learning rate

---

## Summary

Online learning options:
1. **Micro-finetune** - Safest, needs buffer
2. **Hebbian** - Fastest, needs care
3. **Meta-learning** - Most sophisticated, expensive

Choose based on:
- Safety requirements → Option 1
- Latency budget → Option 2
- Sophistication → Option 3

Use when your deployment environment is non-stationary and you can tolerate learning overhead.

---

## Further Reading

- EWC (Elastic Weight Consolidation): Kirk et al., 2017
- Continual Learning: van de Ven & Tolias, 2019
- MAML (Model-Agnostic Meta-Learning): Finn et al., 2017
