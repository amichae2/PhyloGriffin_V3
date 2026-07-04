# Prompt: Add `torch.compile` Support to PhyloGriffin v3

## Context

You are modifying an existing PhyloGriffin v3 codebase. The model is implemented in a monolithic `model.py` (~1140 lines) with these key classes:

- `RGLRU` — Real-Gated Linear Recurrent Unit with both sequential (for-loop) and parallel (Blelloch scan) forward paths
- `GriffinBlock` — one Griffin layer (recurrent or local attention)
- `GriffinColumnEncoder` — full column processor, 8 Griffin layers stacked
- `DiffusionDenoiser` — bipartite GNN for denoising tree split matrices
- `PhyloGriffin` — top-level model wrapping all components

The RG-LRU's `forward_parallel` contains Python for-loops inside the Blelloch scan. The `forward_sequential` contains a Python for-loop over timesteps. Both are exactly the kind of code `torch.compile` accelerates — 3-8× speedup is typical for this pattern.

The codebase currently does NOT use `torch.compile` anywhere.

## What You Need To Do

### Step 1: Add a `compile_model` function to `model.py`

Add this function near the bottom of `model.py` (after the `PhyloGriffin` class, before any `if __name__ == "__main__"` block):

```python
def compile_model(
    model: "PhyloGriffin",
    mode: str = "reduce-overhead",
    dynamic: bool = True,
    compile_griffin: bool = True,
    compile_diffusion: bool = True,
    compile_supertree: bool = False,
) -> "PhyloGriffin":
    """
    Apply torch.compile to selected PhyloGriffin components.

    This function selectively compiles the components that benefit most
    from compilation while avoiding graph breaks in modules that use
    data-dependent control flow.

    Args:
        model: PhyloGriffin instance
        mode: torch.compile mode. Options:
              "default" — balanced compile time vs speedup
              "reduce-overhead" — best for loops, uses CUDA graphs
              "max-autotune" — best speedup, slowest compilation
        dynamic: If True, allow variable N (sequences) and L (columns)
                 without recompilation. REQUIRED for phylogenetics
                 since every MSA has different dimensions.
        compile_griffin: Compile the Griffin column encoder layers
        compile_diffusion: Compile the diffusion denoiser
        compile_supertree: Compile the supertree reconciler (may cause
                           graph breaks — off by default)

    Returns:
        The model with compiled components (modified in-place AND returned)

    Usage:
        model = PhyloGriffin(config)
        model = compile_model(model)

    Notes:
        - The first forward pass after compilation will be SLOWER due to
          compilation overhead. Subsequent passes will be 2-5x faster
          on compiled components.
        - This function is a no-op if torch.compile is not available
          (PyTorch < 2.0). It logs a warning instead of crashing.
        - The Griffin encoder's layers are compiled as a single unit
          (nn.Sequential wrapping the layer list) to maximize fusion.
    """
    import warnings

    if not hasattr(torch, "compile"):
        warnings.warn(
            "torch.compile not available (requires PyTorch >= 2.0). "
            "Skipping compilation. Performance will be significantly slower."
        )
        return model

    compile_opts = {
        "dynamic": dynamic,
    }
    if mode != "default":
        compile_opts["mode"] = mode

    # --- Compile Griffin column encoder layers ---
    if compile_griffin and hasattr(model, "column_encoder"):
        # Wrap the layer list in Sequential so torch.compile sees it as one graph
        encoder = model.column_encoder
        if hasattr(encoder, "layers") and len(encoder.layers) > 0:
            original_layers = encoder.layers
            # Create a Sequential wrapper for compilation
            seq = nn.Sequential(*list(original_layers))
            compiled_seq = torch.compile(seq, fullgraph=True, **compile_opts)
            # Replace the ModuleList with a simple wrapper that delegates to compiled_seq
            encoder.layers = compiled_seq
            encoder._compiled = True
            print(f"[compile_model] Griffin layers compiled (mode={mode}, dynamic={dynamic})")

    # --- Compile diffusion denoiser ---
    if compile_diffusion and hasattr(model, "diffusion_denoiser"):
        denoiser = model.diffusion_denoiser
        model.diffusion_denoiser = torch.compile(
            denoiser, fullgraph=True, **compile_opts
        )
        print(f"[compile_model] Diffusion denoiser compiled (mode={mode}, dynamic={dynamic})")

    # --- Optionally compile supertree reconciler ---
    if compile_supertree and hasattr(model, "supertree_reconciler"):
        # The supertree reconciler may have data-dependent attention masks
        # that cause graph breaks. Use fullgraph=False to tolerate them.
        reconciler = model.supertree_reconciler
        model.supertree_reconciler = torch.compile(
            reconciler, fullgraph=False, **compile_opts
        )
        print(f"[compile_model] Supertree reconciler compiled with fullgraph=False")

    return model
```

### Step 2: Refactor the `GriffinColumnEncoder` to support `nn.Sequential` wrapping

The current `GriffinColumnEncoder.forward()` iterates over `self.layers` (a ModuleList) in a Python for-loop and optionally wraps each call in `torch.utils.checkpoint`. This is incompatible with `torch.compile(fullgraph=True)` because Dynamo can't trace through the checkpoint wrapping inside the loop.

Modify `GriffinColumnEncoder` to expose the layer stack as a compilable `nn.Sequential`:

```python
class GriffinColumnEncoder(nn.Module):
    def __init__(self, config):
        # ... existing init code (unchanged) ...

        # Build the layer sequence for compilation
        self._build_layer_sequence()
    
    def _build_layer_sequence(self):
        """Create a Sequential layer stack that torch.compile can optimize."""
        layer_list = []
        for layer in self.layers:
            layer_list.append(layer)
        self._layer_sequence = nn.Sequential(*layer_list)
    
    def forward(self, msa, use_checkpoint=True):
        x = self._embed_sequences(msa)
        
        if hasattr(self, '_compiled') and self._compiled:
            # When compiled, use the Sequential directly (no checkpoint)
            x = self.layers(x)  # self.layers is now the compiled Sequential
        else:
            # Original path with checkpoint support
            for layer in self.layers:
                if use_checkpoint and self.training:
                    x = checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
        
        x = self.final_norm(x)
        return x
```

**Important**: The `compile_model` function replaces `encoder.layers` with a compiled `Sequential`. The `_build_layer_sequence` method stores the original uncompiled layers so they can be restored if needed. Add a method:

```python
def reset_layers(self):
    """Restore the original (uncompiled) layer list."""
    if hasattr(self, '_original_layers'):
        self.layers = self._original_layers
        self._compiled = False
```

And in `_build_layer_sequence`, save the reference:

```python
def _build_layer_sequence(self):
    self._original_layers = self.layers  # save for reset
    # ...
```

### Step 3: Add `torch.compile` to the RGLRU's parallel scan

The `RGLRU._parallel_scan` method contains Python for-loops over `stride` — this is the single highest-impact compilation target. Add a compiled wrapper:

```python
class RGLRU(nn.Module):
    def __init__(self, dim, c=8.0):
        # ... existing init ...
        self._scan_compiled = None  # lazy init
    
    def _get_compiled_scan(self, a, b):
        """Get or create a compiled version of the parallel scan."""
        if self._scan_compiled is None and hasattr(torch, 'compile'):
            # Compile a standalone scan function
            @torch.compile(fullgraph=True, dynamic=True, mode="reduce-overhead")
            def compiled_scan(a, b):
                return _parallel_scan_impl(a, b)
            self._scan_compiled = compiled_scan
        return self._scan_compiled
    
    def forward_parallel(self, x):
        B, T, D = x.shape
        i_t, r_t, a_t = self._compute_gates(x)
        b_t = torch.sqrt(1.0 - a_t * a_t + 1e-8) * i_t * x
        
        if hasattr(torch, 'compile'):
            scan_fn = self._get_compiled_scan(a_t, b_t)
            if scan_fn is not None:
                return scan_fn(a_t, b_t)
        
        # Fallback: uncompiled scan
        return self._parallel_scan(a_t, b_t)
```

**Refactor**: Extract the current `_parallel_scan` body into a standalone function `_parallel_scan_impl(a, b)` that takes the padded tensors and returns the result. The existing `_parallel_scan` method becomes a thin wrapper that calls `_parallel_scan_impl`. The compiled version also calls `_parallel_scan_impl`. This avoids code duplication.

### Step 4: Add a compile/uncompile toggle to the top-level model

Add to the `PhyloGriffin` class:

```python
class PhyloGriffin(nn.Module):
    def __init__(self, config):
        # ... existing init ...
        self._compiled = False
    
    def compile(self, mode="reduce-overhead", dynamic=True):
        """Apply torch.compile to performance-critical components."""
        compile_model(self, mode=mode, dynamic=dynamic)
        self._compiled = True
        return self
    
    def uncompile(self):
        """Restore uncompiled versions of all components."""
        if hasattr(torch, 'compile') and hasattr(torch.compile, 'reset'):
            torch.compiler.reset()
        if hasattr(self, 'column_encoder') and hasattr(self.column_encoder, 'reset_layers'):
            self.column_encoder.reset_layers()
        self._compiled = False
        return self
```

### Step 5: Update the Colab notebook

In `phylogriffin_v3_colab.ipynb`, after the cell that initializes the model (currently Cell 5), add a new cell:

```python
"""
COMPILE MODEL FOR PERFORMANCE
-----------------------------
Apply torch.compile to the Griffin layers and diffusion denoiser.
This provides 2-5x speedup on the column processor (where most
compute time is spent) and a ~1.5x end-to-end training speedup.

The first forward pass after this cell will be SLOW (compilation).
This is normal. Every subsequent pass will use the compiled graph.
"""

import torch

print(f"PyTorch version: {torch.__version__}")
if hasattr(torch, 'compile'):
    print("torch.compile is available. Compiling model components...")
    model = model.compile(mode="reduce-overhead", dynamic=True)
    
    # Warm up: run one forward pass to trigger compilation
    print("Running compilation warm-up pass...")
    dummy_msa = torch.randint(0, 20, (2, 50, 200), device=device)
    with torch.no_grad():
        _ = model.column_encoder(dummy_msa, use_checkpoint=False)
    print("Compilation complete! Subsequent forward passes will be 2-5x faster.")
else:
    print("torch.compile not available (requires PyTorch >= 2.0).")
    print("Install with: pip install torch>=2.0")
    print("Continuing without compilation — training will be slower.")
```

### Step 6: Add environment validation

Add a cell BEFORE the compilation cell that verifies the environment:

```python
"""
VERIFY ENVIRONMENT FOR torch.compile
------------------------------------
Check that we have a compatible PyTorch and CUDA setup.
"""

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    
    # torch.compile requires compute capability >= 7.0 for CUDA graphs
    major, minor = torch.cuda.get_device_capability(0)
    if major < 7:
        print("⚠️  GPU compute capability < 7.0. torch.compile may not work optimally.")
        print("    Consider using mode='default' instead of 'reduce-overhead'.")
        COMPILE_MODE = "default"
    else:
        COMPILE_MODE = "reduce-overhead"
else:
    COMPILE_MODE = "default"
    print("No GPU detected. torch.compile will use CPU backend.")

# Check torch.compile availability
if not hasattr(torch, 'compile'):
    print("⚠️  torch.compile NOT available. Upgrade to PyTorch >= 2.0.")
    print("    pip install torch>=2.0 --upgrade")
    USE_COMPILE = False
else:
    USE_COMPILE = True
    print("✅ torch.compile is available")
```

Then in the compilation cell, use `COMPILE_MODE` and `USE_COMPILE`:

```python
if USE_COMPILE:
    model = model.compile(mode=COMPILE_MODE, dynamic=True)
    # ... warm-up ...
```

---

## What NOT To Do

- **Do NOT compile the Titans memory module.** If the codebase has a Titans memory (or similar module with data-dependent if/else and in-place buffer mutation), leave it uncompiled. Compiling it will cause graph breaks on every forward pass, making things slower.

- **Do NOT use `fullgraph=True` on the entire model.** The top-level `PhyloGriffin` contains components that Dynamo cannot trace as a single graph (tree structure manipulation, Newick parsing, etc.). Always compile individual submodules.

- **Do NOT hardcode a static sequence length.** The `dynamic=True` flag is essential. Training uses batches with different N (sequences) and L (columns). Without `dynamic=True`, the compiler recompiles on every batch — slower than no compilation at all.

- **Do NOT compile during the first few training steps.** The compilation happens on the first forward pass (the warm-up in Step 5). Make sure this happens BEFORE the training loop, not during it, to avoid confusing timing and loss logging.

- **Do NOT ship `torch.compile` calls unconditionally.** The `compile_model` function must be a no-op on PyTorch < 2.0. The codebase should work without compilation (just slower).

---

## Validation Checklist

After implementing, verify:

- [ ] `compile_model()` runs without errors on PyTorch >= 2.0
- [ ] `compile_model()` prints a warning (not a crash) on PyTorch < 2.0
- [ ] A dummy forward pass succeeds after compilation
- [ ] The compiled model produces numerically identical results to the uncompiled model (within 1e-4 tolerance for float32)
- [ ] The second forward pass is measurably faster than the first (the first includes compilation time)
- [ ] Training runs for at least 100 steps without graph breaks or recompilation warnings
- [ ] Variable batch sizes (different N, different L) work without recompilation (check with `TORCH_LOGS=recompiles`)
- [ ] `model.uncompile()` restores the original uncompiled state

---

## Files You Need To Modify

| File | What changes |
|------|-------------|
| `model.py` | Add `compile_model()` function. Refactor `GriffinColumnEncoder` for compilable layer stack. Extract `_parallel_scan_impl()` from `RGLRU._parallel_scan()`. Add `compile()` and `uncompile()` to `PhyloGriffin`. Add compiled scan to `RGLRU`. |
| `phylogriffin_v3_colab.ipynb` | Add environment verification cell. Add compilation + warm-up cell. Add note about first-forward-pass slowness. |

No other files should change. The training scripts, data pipeline, configuration, and inference code are unaffected.

---

## Performance Expectations

| Component | Expected speedup | Notes |
|-----------|-----------------|-------|
| RGLRU parallel scan | 3-8× | Python for-loops → fused CUDA kernels |
| Griffin layer stack | 1.5-2× | Fused MLP + attention + norm operations |
| Diffusion denoiser | 2-4× | Message-passing loops get fused |
| End-to-end training | 1.5-2.5× | Dominated by column processor |
| First forward pass | 0.3-0.5× | SLOWER—compilation overhead. One-time cost. |

The `reduce-overhead` mode is recommended for Colab because:
- It enables CUDA graphs (reducing kernel launch overhead)
- It compiles faster than `max-autotune`
- The Griffin architecture's compute patterns are relatively simple — autotuning provides marginal additional benefit

If training on an A100, `max-autotune` may give an additional 10-20% but adds 2-5 minutes to compilation time.
