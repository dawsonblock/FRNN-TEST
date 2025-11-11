#!/usr/bin/env python3
"""
Test script to verify the hardened system is ready.
This checks what's available without requiring the C++ extensions.
"""

import sys
import os

print("="*80)
print("FRNN PATH B HARDENED SYSTEM - VERIFICATION")
print("="*80)
print()

# Check Python files
print("1. Core Scripts:")
scripts = {
    "train_hardened.py": "Training with all blockers fixed",
    "parity_hardened.py": "Parity validation with deterministic checks"
}

for script, desc in scripts.items():
    if os.path.exists(script):
        size = os.path.getsize(script)
        print(f"   ✓ {script:25s} ({size:6d} bytes) - {desc}")
    else:
        print(f"   ✗ {script:25s} MISSING")

print()

# Check documentation
print("2. Documentation:")
docs = {
    "README.md": "Quick start guide",
    "CPP-FIX-REQUIRED.md": "Critical C++ fix",
    "HARDENED-SUMMARY.md": "Executive summary",
    "HARDENED-INDEX.md": "Navigation guide",
    "BANK-FUSION-GUIDE.md": "Optional enhancement"
}

for doc, desc in docs.items():
    if os.path.exists(doc):
        size = os.path.getsize(doc)
        print(f"   ✓ {doc:25s} ({size:6d} bytes) - {desc}")
    else:
        print(f"   ✗ {doc:25s} MISSING")

print()

# Check for C++ extensions
print("3. C++ Extensions (Required to Run):")
extensions = [
    ("frnn_aten_trainable/frnn_aten", "ATen backend"),
    ("frnn_fused_v4_bindings", "CUDA fused backend")
]

all_present = True
for ext_path, desc in extensions:
    if os.path.exists(ext_path):
        print(f"   ✓ {ext_path:40s} - {desc}")
    else:
        print(f"   ✗ {ext_path:40s} - {desc} (NOT FOUND)")
        all_present = False

print()

# Try importing
print("4. Import Test:")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"     - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     - CUDA version: {torch.version.cuda}")
except ImportError:
    print("   ✗ PyTorch not available")

try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__}")
except ImportError:
    print("   ✗ NumPy not available")

print()

# Try importing extensions
print("5. Extension Import Test:")
try:
    import frnn_aten
    print("   ✓ frnn_aten imported successfully")
except ImportError as e:
    print(f"   ✗ frnn_aten import failed: {e}")
    print("     Build with: cd frnn_aten_trainable/frnn_aten && python setup.py build_ext --inplace")

try:
    import frnn_fused_v4
    print("   ✓ frnn_fused_v4 imported successfully")
except ImportError as e:
    print(f"   ✗ frnn_fused_v4 import failed: {e}")
    print("     Build with: cd frnn_fused_v4_bindings && cmake -B build && cmake --build build -j")

print()
print("="*80)
print("SUMMARY")
print("="*80)

if all_present:
    print("✓ All required directories present")
    print()
    print("Next steps:")
    print("  1. Build extensions (see above)")
    print("  2. Run: python train_hardened.py")
    print("  3. Run: python parity_hardened.py")
else:
    print("✗ Missing required C++ extension directories")
    print()
    print("This hardened system provides:")
    print("  - Production-ready training scripts with all blockers fixed")
    print("  - Comprehensive parity validation")
    print("  - Complete documentation suite")
    print()
    print("To use it, you need:")
    print("  - frnn_aten_trainable/ directory with ATen backend")
    print("  - frnn_fused_v4_bindings/ directory with CUDA backend")
    print()
    print("These directories should contain your FRNN C++ implementations.")

print("="*80)
