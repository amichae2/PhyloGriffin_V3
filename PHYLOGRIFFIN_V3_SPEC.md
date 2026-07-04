# PhyloGriffin v3 — Complete Build Specification

## Purpose of This Document

This is a machine-executable specification. You should be able to read this document and produce every file needed for a working implementation WITHOUT asking any follow-up questions. Every tensor shape, every loss function, every hyperparameter, every file path, every function signature, every Colab-specific constraint, and every implementation detail is specified.

## Project Structure

The project is split into TWO parts:

- **Part A — Local Development** (runs on your laptop): Python package with model definitions, utilities, and inference pipeline. You develop, lint, and test locally.
- **Part B — Training on Google Colab** (runs in browser): A Jupyter notebook that imports Part A, installs dependencies, generates/loads training data, trains all six stages, saves checkpoints to Google Drive, and runs an inference demo.

Part A files are `.py` files in a `phylogriffin/` package. Part B is a single `.ipynb` notebook at the project root. The other model must produce BOTH.

---

## FILE MANIFEST

You create these EXACT files:

```
phylogriffin/
├── __init__.py
├── config.py
├── data.py
├── tree_utils.py
├── simulation.py
├── inference.py
├── model/
│   ├── __init__.py
│   ├── column_processor.py
│   ├── graph_predictor.py
│   ├── decomposition.py
│   ├── diffusion.py
│   ├── supertree.py
│   └── refinement.py
├── train/
│   ├── __init__.py
│   ├── train_column_recon.py
│   ├── train_column_contrast.py
│   ├── train_graph.py
│   ├── train_diffusion.py
│   ├── train_supertree.py
│   └── train_refinement.py
phylogriffin_v3_colab.ipynb          <-- THE NOTEBOOK (Part B)
requirements.txt
setup.py
```

Total: 20 files (19 `.py` + 1 `.ipynb` + 1 `requirements.txt` + 1 `setup.py` = 22 files)

---

## SECTION A: COMPLETE ARCHITECTURAL SPECIFICATION

### A.1 Overview

PhyloGriffin-v3 produces a phylogenetic tree from a multiple sequence alignment (MSA) of shape `(N, L)` where N = number of sequences (taxa) and L = number of alignment columns (sites).

Unlike traditional pipelines (MSA → pairwise distances → tree), PhyloGriffin-v3 uses learned neural representations end-to-end, then generates the tree via hierarchical decomposition and diffusion-based tree generation.

### A.2 Pipeline Stages (Inference)

```
INPUT: MSA tensor (N, L) integers + sequence names list
  │
  ▼
STAGE A: Column Processor (Griffin SSM + Titans Memory)
  Produces: per-sequence embeddings (N, D_model)
  Produces: per-column memory signatures (L, D_mem)
  │
  ▼
STAGE B: Learned Phylogenetic Graph
  Produces: sparse edge list (E, 2), E ≈ k_neighbors × N
  │
  ▼
STAGE C: Hierarchical Decomposition
  Produces: K subproblems, each with N_k ≈ 500–1500 leaves
  │
  ▼
STAGE D: Per-Subproblem Diffusion Tree Generator
  Produces: K subtrees (Newick strings)
  │
  ▼
STAGE E: Learned Supertree Reconciler
  Produces: one global tree (Newick string)
  │
  ▼
STAGE F: Refinement Pass
  Produces: refined global tree (Newick string)
  │
  ▼
OUTPUT: Final phylogenetic tree (Newick string)
```

### A.3 Training Stages

| Stage | Component Trained | Objective | Data Required |
|-------|------------------|-----------|---------------|
| Train-1 | Column Processor | Masked column reconstruction (self-supervised) | Raw MSAs, no trees needed |
| Train-2 | Column Processor + Titans Memory | Contrastive phylogenetic embedding | Simulated MSAs with known trees |
| Train-3 | Graph Predictor | Binary edge classification (phylogenetic neighbor?) | Simulated MSAs with known trees |
| Train-4 | Diffusion Denoiser | Denoising score matching in tree space | Simulated subtrees (N ≤ 1500) |
| Train-5 | Supertree Reconciler | Reconstruct full tree from artificially split subtrees | Simulated decomposed trees |
| Train-6 | Refinement Pass | NNI correctness prediction | Simulated trees with injected errors |

---

## SECTION B: CONFIGURATION FILE — config.py

### B.1 Complete Specification

File: `phylogriffin/config.py`

```python
"""
PhyloGriffin v3 — Configuration dataclasses.
All hyperparameters, constants, and paths live here.
No other file defines magic numbers.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any


@dataclass
class GriffinConfig:
    d_model: int = 512
    d_rnn: int = 680
    n_layers: int = 12
    n_heads: int = 4
    head_dim: int = 128
    local_window: int = 1024
    mlp_expansion: int = 3
    dropout: float = 0.1
    pattern: Tuple[int, int] = (2, 1)


@dataclass
class TitansConfig:
    d_mem: int = 256
    n_memory_slots: int = 128
    memory_depth: int = 3
    surprise_threshold: float = 0.1
    momentum: float = 0.9


@dataclass
class GraphConfig:
    k_neighbors: int = 50
    k_candidates: int = 200
    predictor_hidden: List[int] = field(default_factory=lambda: [512, 256])
    edge_threshold: float = 0.5


@dataclass
class DecompositionConfig:
    max_subproblem_size: int = 1500
    min_subproblem_size: int = 500
    clustering_method: str = "spectral"


@dataclass
class DiffusionConfig:
    n_diffusion_steps: int = 1000
    denoiser_layers: int = 6
    denoiser_hidden: int = 256
    noise_schedule: str = "cosine"
    branch_length_min: float = 0.0
    branch_length_max: float = 5.0
    d_time: int = 128
    n_splits_max: int = 3000


@dataclass
class SupertreeConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_feedforward: int = 1024
    max_subtrees: int = 1024
    dropout: float = 0.1


@dataclass
class RefinementConfig:
    n_rounds: int = 4
    nni_radius: int = 5
    quartet_hidden: int = 256
    nni_margin: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 100000
    lr_schedule: str = "cosine"
    grad_clip: float = 1.0
    max_tokens_per_batch: int = 2_000_000


@dataclass
class PhyloGriffinConfig:
    alphabet_size: int = 21
    gap_idx: int = 20
    pad_idx: int = 21

    griffin: GriffinConfig = field(default_factory=GriffinConfig)
    titans: TitansConfig = field(default_factory=TitansConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    supertree: SupertreeConfig = field(default_factory=SupertreeConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict for checkpointing."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhyloGriffinConfig":
        return cls(**d)

    @classmethod
    def nucleotide_config(cls) -> "PhyloGriffinConfig":
        return cls(alphabet_size=5, gap_idx=4, pad_idx=5)
```

---

## SECTION C: DATA HANDLING — data.py

### C.1 Specification

File: `phylogriffin/data.py`

Must export:

```python
# Amino acid alphabet mapping (standard order matching FastTree-2)
AA_TO_IDX: Dict[str, int]   # A=0, R=1, N=2, D=3, C=4, Q=5, E=6, G=7, H=8, I=9,
                             # L=10, K=11, M=12, F=13, P=14, S=15, T=16, W=17, Y=18, V=19
IDX_TO_AA: Dict[int, str]

NT_TO_IDX: Dict[str, int]   # A=0, C=1, G=2, T=3
IDX_TO_NT: Dict[int, str]

def load_fasta(path: str, alphabet: str = "protein") -> Tuple[torch.Tensor, List[str]]:
    """
    Load a FASTA alignment file.

    Args:
        path: Path to FASTA file
        alphabet: "protein" or "nucleotide"

    Returns:
        msa: LongTensor of shape (N, L) with token indices, padded with pad_idx if needed
        names: List of N sequence names

    Rules:
    - Sequences MUST be same length (pre-aligned). If not, raise ValueError.
    - All characters beyond the standard alphabet (ambiguous codes like B, Z, X) map to gap_idx.
    - Uppercase only.
    - Gap characters: "-" or "." → gap_idx
    - Padding: if sequences have different lengths after loading, pad with pad_idx.
    """

def load_phylip(path: str, alphabet: str = "protein") -> Tuple[torch.Tensor, List[str]]:
    """Same as load_fasta but for interleaved phylip format."""

def load_msa(path: str, alphabet: str = "protein") -> Tuple[torch.Tensor, List[str]]:
    """Auto-detect format (FASTA or phylip) and load."""

class MSADataset(torch.utils.data.Dataset):
    """
    Dataset for self-supervised training (Train-1).

    __init__(self, msa_dir: str, alphabet: str = "protein", max_seq_len: int = 2048)

    __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        Returns dict with keys:
            "msa": LongTensor (N, L) — a random sub-alignment
            "mask": BoolTensor (N, L) — True where position is valid (not pad)

    __len__(self) -> int: number of MSAs in directory
    """

class ContrastiveDataset(torch.utils.data.Dataset):
    """
    Dataset for contrastive training (Train-2).

    __init__(self, msa_dir: str, tree_dir: str, alphabet: str = "protein")

    Each item is one MSA + its true tree.

    __getitem__(self, idx) -> Dict:
        Returns dict with:
            "msa": LongTensor (N, L)
            "mask": BoolTensor (N, L)
            "tree_newick": str
            "leaf_names": List[str]
            "pairwise_distances": FloatTensor (N, N) — patristic distances from true tree
    """

class GraphDataset(torch.utils.data.Dataset):
    """Dataset for graph predictor training (Train-3).
    Same structure as ContrastiveDataset but additionally provides
    sibling_pairs and non_sibling_pairs as precomputed lists."""

class SubproblemDataset(torch.utils.data.Dataset):
    """Dataset for diffusion training (Train-4).
    Each item is a small MSA + true subtree (N ≤ 1500)."""

class DecomposedTreeDataset(torch.utils.data.Dataset):
    """Dataset for supertree training (Train-5).
    Each item is a full MSA + true tree, pre-decomposed into subproblems."""

class ErrorTreeDataset(torch.utils.data.Dataset):
    """Dataset for refinement training (Train-6).
    Each item is a tree with injected NNIs + the correct tree."""
```

### C.2 Tokenization Rules

The `load_fasta` function must:
1. Read all sequences. Verify equal length.
2. For each character: map to index using the alphabet dict. Characters not in dict → gap_idx.
3. If a sequence contains a character that maps to gap_idx but is not "-" or ".", print a WARNING (matching FastTree-2 behavior) listing the unrecognized characters.
4. Return tensor of shape (N, L) with dtype=torch.long.

---

## SECTION D: TREE UTILITIES — tree_utils.py

### D.1 Specification

File: `phylogriffin/tree_utils.py`

Must export these functions. Use `dendropy` if available; otherwise implement a minimal Newick parser.

```python
def parse_newick(newick_str: str) -> object:
    """
    Parse a Newick string into an internal tree representation.

    The internal representation can be any object (custom class, dendropy Tree, etc.)
    as long as the other functions in this module can consume it.

    Supports:
    - Internal node labels (support values): ((A:0.1,B:0.2)0.95:0.3,C:0.4);
    - Branch lengths after colons
    - Quoted leaf names: 'name with spaces'
    - Multifurcations
    """

def tree_to_newick(tree: object) -> str:
    """Convert internal tree representation back to Newick string with branch lengths."""

def newick_to_splits(newick_str: str, n_leaves: int) -> List[Tuple[np.ndarray, float]]:
    """
    Extract bipartitions from a Newick tree.

    Args:
        newick_str: Newick tree string
        n_leaves: Total number of leaves

    Returns:
        List of (mask, branch_length) tuples where:
        - mask: numpy bool array of shape (n_leaves,) 
                True = leaf is on one side of the split
        - branch_length: float, length of the branch corresponding to this split

    Rules:
    - Trivial splits (one leaf vs rest) are EXCLUDED.
    - The root split (if the tree is rooted) is EXCLUDED.
    - Leaves are indexed 0..n_leaves-1 in order of appearance in the Newick string.
    """

def splits_to_newick(splits: List[Tuple[np.ndarray, float]], 
                     leaf_names: List[str]) -> str:
    """
    Convert a compatible set of splits back to a Newick tree.
    
    The splits must be pairwise compatible (form a valid tree).
    If not compatible, raise ValueError.
    """

def robinson_foulds(splits1, splits2) -> float:
    """
    Normalized Robinson-Foulds distance between two split sets.
    
    Args:
        splits1, splits2: lists of (mask, branch_length) tuples
        
    Returns:
        Float in [0, 1]: 0 = identical, 1 = completely different.
        
    Formula: RF = (|S1 \ S2| + |S2 \ S1|) / (|S1| + |S2|)
    where a split is identified by its bipartition mask.
    """

def patristic_distances(tree_newick: str, n_leaves: int) -> np.ndarray:
    """
    Compute all-pairs patristic (tree-path) distances.

    Returns:
        numpy array of shape (n_leaves, n_leaves), float32.
    """

def get_leaf_order(newick_str: str) -> List[str]:
    """Return leaf names in left-to-right order from the Newick string."""

def nni_alternatives(tree: object, internal_node) -> List[object]:
    """
    Given an internal node of a binary tree, return the 3 alternative 
    quartet topologies as tree objects.
    
    The three alternatives are:
    - Current topology: (A,B), (C,D)
    - Alternative 1:   (A,C), (B,D)  
    - Alternative 2:   (A,D), (B,C)
    
    Where A, B, C, D are the four subtrees surrounding the internal edge.
    """

def apply_nni(tree_newick: str, node_id: int, alternative: int) -> str:
    """Apply an NNI at the given node, return new Newick string."""

def is_binary(tree_newick: str) -> bool:
    """Check if all internal nodes have exactly 2 children."""

def collapse_low_support(tree_newick: str, threshold: float = 0.5) -> str:
    """Collapse internal branches with support below threshold into polytomies."""
```

---

## SECTION E: SIMULATION — simulation.py

### E.1 Specification

File: `phylogriffin/simulation.py`

This module generates synthetic MSAs with known trees for training.

```python
def simulate_yule_tree(n_leaves: int, birth_rate: float = 1.0, 
                       seed: int = None) -> str:
    """
    Simulate a pure-birth (Yule) tree.

    Algorithm:
    1. Start with 2 lineages.
    2. At each step, randomly pick an existing lineage to split.
    3. Continue until we have n_leaves.
    4. Branch lengths: each lineage accumulates length proportional to wait time.
    5. Return Newick string with branch lengths.
    """

def simulate_birth_death_tree(n_leaves: int, birth_rate: float = 1.0,
                               death_rate: float = 0.5, seed: int = None) -> str:
    """
    Simulate a birth-death tree with extinction.
    
    Algorithm:
    1. Start with 2 lineages.
    2. At each step: with prob proportional to birth_rate, split a random lineage.
       With prob proportional to death_rate, remove a random lineage.
    3. Continue until n_leaves extant lineages reached, then continue until all
       extant lineages coalesce.
    4. Return Newick string with branch lengths (only extant lineages).
    """

def jtt_rate_matrix() -> np.ndarray:
    """Return the JTT amino acid substitution rate matrix (20×20)."""

def wag_rate_matrix() -> np.ndarray:
    """Return the WAG amino acid substitution rate matrix (20×20)."""

def lg_rate_matrix() -> np.ndarray:
    """Return the LG amino acid substitution rate matrix (20×20)."""

def gtr_rate_matrix(base_freqs: np.ndarray, exchangeabilities: np.ndarray) -> np.ndarray:
    """Return a 4×4 GTR nucleotide substitution rate matrix."""

def evolve_sequences(tree_newick: str,
                     n_sites: int,
                     model: str = "JTT",
                     alpha: float = 1.0,
                     n_categories: int = 4,
                     include_indels: bool = False,
                     indel_rate: float = 0.01,
                     seed: int = None) -> Tuple[torch.Tensor, List[str]]:
    """
    Evolve sequences along a given tree.

    Args:
        tree_newick: Guide tree with branch lengths
        n_sites: Number of alignment columns to simulate
        model: "JTT", "WAG", "LG" (protein) or "JC", "GTR" (nucleotide)
        alpha: Gamma shape parameter for rate heterogeneity
        n_categories: Number of discrete rate categories for gamma approximation
        include_indels: Whether to simulate insertions/deletions
        indel_rate: Probability of an indel event per site per unit branch length
        seed: Random seed

    Returns:
        msa: LongTensor of shape (n_leaves, n_sites) with token indices
        seq_names: List of leaf names from the tree

    Algorithm:
    1. Parse the tree, get leaf order.
    2. Generate a random ancestral sequence of length n_sites at the root.
       - Amino acid frequencies from the model's stationary distribution.
    3. For each site, sample a rate multiplier from a discretized gamma distribution
       with shape=alpha, scale=1/alpha (mean=1).
    4. Traverse the tree in preorder. For each branch of length b:
       a. Effective branch length = b * rate_multiplier[site].
       b. Compute transition probability matrix: P = expm(Q * b_effective).
       c. For each site, sample the descendant state from P[ancestral_state, :].
    5. The sequences at the leaves form the MSA.
    6. Return tokenized MSA and leaf names.

    If include_indels:
    - After substitutions, simulate insertions and deletions using a simple model.
    - Deletions: randomly remove some columns from a sequence.
    - Insertions: randomly insert columns (drawn from stationary distribution).
    - The final MSA has columns that exist in at least one sequence.
    - Gaps from indels are represented as gap_idx.
    """

def generate_training_batch(n_examples: int,
                            n_leaves_range: Tuple[int, int],
                            n_sites_range: Tuple[int, int],
                            model: str = "JTT",
                            include_indels: bool = False,
                            output_dir: str = None,
                            seed: int = None) -> List[Dict]:
    """
    Generate a batch of training examples and optionally save to disk.

    Args:
        n_examples: Number of (MSA, tree) pairs to generate
        n_leaves_range: (min, max) for number of leaves per tree
        n_sites_range: (min, max) for number of sites per MSA
        model: Substitution model
        include_indels: Whether to include indels
        output_dir: If provided, save each example as msa_{i}.fa and tree_{i}.nwk
        seed: Random seed

    Returns:
        List of dicts with keys: "msa", "tree_newick", "seq_names", "n_leaves", "n_sites"
    """
```

**IMPORTANT**: The simulation module MUST work without external bioinformatics libraries (no `pyvolve`, no `dendropy` for simulation). Use only `numpy` + `torch`. The `expm` (matrix exponential) function can use `scipy.linalg.expm` if available, or implement a simple Pade approximation. This is critical for Colab compatibility.

---

## SECTION F: MODEL COMPONENTS

### F.1 Column Processor — model/column_processor.py

#### F.1.1 Module Exports

```python
class ColumnProcessor(nn.Module):
    """
    Stage A: Griffin SSM + Titans Co-evolution Memory.

    __init__(self, config: PhyloGriffinConfig)
    
    forward(self, msa: LongTensor, mask: BoolTensor = None) -> Tuple[Tensor, Tensor]:
        Args:
            msa: (N, L) integer token indices
            mask: (N, L) bool, True = valid position. If None, computed from msa != pad_idx.
        Returns:
            seq_embeddings: (N, d_model) float tensor
            col_memory: (n_memory_slots, d_mem) float tensor
    """

class TokenEmbedding(nn.Module):
    """Embedding layer. Maps token index → d_model vector."""

class RG_LRU(nn.Module):
    """
    Real-Gated Linear Recurrent Unit.

    Implements one step of:
        r_t = sigmoid(W_r * x_t)
        i_t = sigmoid(W_i * x_t)
        a_t = sigmoid(Lambda) ^ (c * r_t)
        h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * x_t)
        y_t = h_t

    __init__(self, d_model: int, d_rnn: int)
    forward(self, x: Tensor, state: Tensor = None) -> Tuple[Tensor, Tensor]:
        Args:
            x: (N, d_model) — input for one column
            state: (N, d_rnn) — previous hidden state, or None for zeros
        Returns:
            y: (N, d_model) — output
            new_state: (N, d_rnn) — new hidden state
    """

class ParallelRG_LRU(nn.Module):
    """
    Parallel implementation using associative scan.
    Computes the entire sequence at once rather than step-by-step.

    forward(self, x: Tensor) -> Tensor:
        Args:
            x: (N, L, d_model) — full sequence
        Returns:
            y: (N, L, d_model)
    """

class GatedMLP(nn.Module):
    """GeGeLU gated MLP block."""

class LocalMQA(nn.Module):
    """Local sliding-window Multi-Query Attention with RoPE."""

class GriffinLayer(nn.Module):
    """
    One Griffin layer: temporal mixing + gated MLP with residual connections.
    temporal_mixing is either RG_LRU or LocalMQA, determined by layer index.
    """

class TitansMemory(nn.Module):
    """
    Co-evolution memory module.
    
    __init__(self, d_model: int, d_mem: int, n_slots: int, depth: int, 
             surprise_threshold: float, momentum: float)
    
    forward(self, col_repr: Tensor, mask: Tensor) -> Tensor:
        Args:
            col_repr: (N, d_model) — processed column representation
            mask: (N,) bool — which sequences are valid at this column
        Returns:
            enriched: (N, d_model) — representation enriched with memory context
    
    reset_state(self):
        """Reset memory for a new MSA."""
```

#### F.1.2 RG-LRU Implementation Details

The RG-LRU is the core recurrent unit. It must be implemented correctly.

**Step-by-step for ParallelRG_LRU.forward(x)** where x has shape `(N, L, d_model)`:

```
1. Project to RNN dimension:
   x_rnn = Linear(d_model, d_rnn)(x)        # (N, L, d_rnn)
   x_gate = Linear(d_model, d_rnn)(x)       # (N, L, d_rnn)

2. Apply depthwise Conv1D along L dimension:
   # Use Conv1d with groups=d_rnn, kernel_size=4, padding=3
   # This is applied per-sequence independently.
   # Reshape to (N*d_rnn, 1, L) for grouped conv, then back to (N, L, d_rnn)

3. Compute gates (parallel across all t):
   r = sigmoid(Linear(d_rnn, d_rnn)(x_rnn_conv))   # (N, L, d_rnn)
   i = sigmoid(Linear(d_rnn, d_rnn)(x_rnn_conv))   # (N, L, d_rnn)

4. Learnable Lambda parameter: shape (d_rnn,), initialized to 
   torch.randn(d_rnn) * 0.01 (small values near zero)

5. Compute a_t for all t:
   # Lambda is broadcast: (1, 1, d_rnn)
   # r is: (N, L, d_rnn)
   # c = 8 (constant)
   log_a = c * r * F.logsigmoid(Lambda)     # log-space for stability
   a = exp(log_a)                            # (N, L, d_rnn)
   # Clamp a to [0, 1] after exponentiation

6. Associative scan to compute h_t:
   # We need: h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * x_rnn_conv_t)
   # This is a first-order linear recurrence with diagonal a_t.
   # Use parallel prefix sum / associative scan.
   
   # Compute the input term:
   input_term = sqrt(1 - a_t^2) * (i_t * x_rnn_conv_t)   # (N, L, d_rnn)
   
   # The scan computes: h_t = sum_{j=0}^{t} [prod_{k=j+1}^{t} a_k] * input_term_j
   # This can be done with a parallel scan (Blelloch scan).
   # Implementation: use torch.cumsum in log-space for the cumulative product,
   # then multiply and cumsum again.
   
   log_a_cumsum = torch.cumsum(F.logsigmoid(Lambda) * c * r, dim=1)  # (N, L, d_rnn)
   # Shift: log_a_cumsum_shifted[t] = log_a_cumsum[t-1]
   # Then h_t = sum over j of exp(log_a_cumsum[t] - log_a_cumsum[j]) * input_term_j
   
   # Efficient implementation using the associative scan property:
   # Let pair = (a, b). Define binary operator: (a1,b1) • (a2,b2) = (a1*a2, b1*a2 + b2)
   # Then scan over these pairs gives cumulative (product, output).
   # Implement with a simple Python loop if torch doesn't have associative_scan,
   # OR use the heuristic_scan fallback.
   
   # FALLBACK for Colab (if no triton/custom scan):
   # For L ≤ 10K, a simple sequential scan in PyTorch is acceptable:
   h = torch.zeros(N, d_rnn, device=x.device)
   outputs = []
   for t in range(L):
       h = a[:, t] * h + input_term[:, t]
       outputs.append(h)
   h_seq = torch.stack(outputs, dim=1)   # (N, L, d_rnn)

7. Output gating:
   y = h_seq * F.gelu(x_gate)              # (N, L, d_rnn)
   y = Linear(d_rnn, d_model)(y)           # (N, L, d_model)

8. Return y
```

**CRITICAL**: The sequential scan fallback (loop over L) is ACCEPTABLE for Colab because L (alignment length) is typically ≤ 10K for protein alignments and ≤ 50K for nucleotide. At L=10K and N=100 (a typical batch), the loop is fast. If performance is needed, use `torch.compile` on the loop, or implement the true associative scan.

#### F.1.3 Titans Memory Implementation Details

```
reset_state():
    self.keys = nn.Parameter(torch.randn(n_slots, d_mem) * 0.02)   # learnable init
    self.values = torch.zeros(n_slots, d_mem)
    self.usage = torch.zeros(n_slots)

forward(col_repr, mask):
    # col_repr: (N, d_model)
    # mask: (N,) bool
    
    # 1. Aggregate column signature:
    valid = col_repr[mask]  # (N_valid, d_model)
    c_col = valid.mean(dim=0)  # (d_model,)
    c_col = Linear(d_model, d_mem)(c_col)  # (d_mem,)
    
    # 2. Query memory:
    query = Linear(d_mem, d_mem)(c_col)    # (d_mem,)
    scores = F.softmax(self.keys @ query / sqrt(d_mem), dim=0)  # (n_slots,)
    predicted = scores @ self.values        # (d_mem,)
    
    # 3. Surprise gating:
    surprise = F.mse_loss(c_col, predicted)
    if surprise > self.surprise_threshold:
        idx = scores.argmax()
        self.keys.data[idx] = self.momentum * self.keys[idx] + (1-self.momentum) * c_col.detach()
        # Apply memory MLP:
        mem_val = c_col
        for _ in range(self.memory_depth):
            mem_val = F.silu(Linear(d_mem, d_mem)(mem_val))
        self.values[idx] = self.momentum * self.values[idx] + (1-self.momentum) * mem_val.detach()
        self.usage[idx] += 1
    
    # 4. Read and enrich:
    memory_context = scores @ self.values   # (d_mem,)
    enriched = col_repr + Linear(d_mem, d_model)(memory_context).unsqueeze(0)  # (N, d_model)
    
    return enriched
```

#### F.1.4 ColumnProcessor.forward() Pseudocode

```
forward(msa, mask=None):
    N, L = msa.shape
    
    if mask is None:
        mask = (msa != pad_idx)
    
    # Token embedding
    x = self.token_embed(msa)  # (N, L, d_model)
    
    # Reset Titans memory
    self.titans.reset_state()
    
    # Process through Griffin layers
    for layer_idx, layer in enumerate(self.griffin_layers):
        # Temporal mixing
        if layer.is_recurrent:
            x_temporal = self.rg_lru[layer.rg_lru_idx](x)
        else:
            x_temporal = self.local_attn[layer.attn_idx](x, mask)
        
        # Residual + RMSNorm
        x = RMSNorm(x + x_temporal)
        
        # Titans memory read/write (after temporal mixing)
        for t in range(L):
            col = x[:, t, :]       # (N, d_model)
            col_mask = mask[:, t]  # (N,)
            x[:, t, :] = self.titans(col, col_mask)
        
        # Gated MLP
        x = RMSNorm(x + self.mlp[layer_idx](x))
    
    # Final RMSNorm
    x = RMSNorm(x)
    
    # Mean pool to get per-sequence embeddings
    # Only pool over valid positions (not masked)
    seq_emb = (x * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1).float()
    # (N, d_model)
    
    return seq_emb, self.titans.values  # also return final memory state
```

---

### F.2 Graph Predictor — model/graph_predictor.py

```python
class GraphPredictor(nn.Module):
    """
    Stage B: Predicts whether two sequences are phylogenetically adjacent.

    __init__(self, d_model: int, hidden_dims: List[int])

    forward_single(self, emb_i: Tensor, emb_j: Tensor) -> Tensor:
        Predict for one pair.
        Args:
            emb_i, emb_j: both (d_model,)
        Returns:
            prob: scalar tensor in [0, 1]

    forward_batch(self, emb_i: Tensor, emb_j: Tensor) -> Tensor:
        Predict for many pairs.
        Args:
            emb_i, emb_j: both (B, d_model)
        Returns:
            prob: (B,) in [0, 1]

    build_graph(self, embeddings: Tensor) -> Tuple[Tensor, Tensor]:
        Build the full sparse phylogenetic graph.
        Args:
            embeddings: (N, d_model)
        Returns:
            edge_index: (2, E) LongTensor
            edge_weights: (E,) FloatTensor
        
        Algorithm:
        1. Compute L2 distance matrix between all pairs of embeddings.
           If N > 10000, use chunked computation on GPU.
        2. For each node i, take the top k_candidates (200) nearest neighbors.
        3. For each candidate edge, compute predictor probability.
        4. Keep edges where prob > edge_threshold.
        5. Enforce symmetry and remove self-loops.
    """
```

The `forward_single` method implements:
```
concat = torch.cat([
    emb_i, emb_j, 
    emb_i * emb_j,           # interaction
    torch.abs(emb_i - emb_j) # absolute difference
])  # (4 * d_model,)

x = concat
for hidden_dim in self.hidden_dims:
    x = F.leaky_relu(self.layers[i](x))
    x = F.dropout(x, p=0.1, training=self.training)
x = self.output_layer(x)  # Linear(last_hidden, 1)
return torch.sigmoid(x).squeeze(-1)
```

---

### F.3 Decomposition — model/decomposition.py

```python
class HierarchicalDecomposition(nn.Module):
    """
    Stage C: Partition leaves into manageable subproblems.

    __init__(self, config: DecompositionConfig)

    def forward(self, msa: Tensor, embeddings: Tensor, 
                edge_index: Tensor, edge_weights: Tensor) -> Tuple[List[Dict], str]:
        """
        Args:
            msa: (N, L) integer tensor
            embeddings: (N, d_model) float tensor
            edge_index: (2, E) long tensor
            edge_weights: (E,) float tensor

        Returns:
            subproblems: List of dicts, each with:
                "indices": LongTensor of leaf indices in this subproblem
                "sub_msa": LongTensor (N_k, L)
                "sub_embeddings": FloatTensor (N_k, d_model)
                "sub_edge_index": LongTensor (2, E_k) — edges within subproblem
            guide_tree_newick: str — Newick tree over subproblems (built from mean embeddings)
        """

    def _spectral_clustering(self, edge_index, edge_weights, n_nodes, 
                             max_size, min_size) -> List[Tensor]:
        """Spectral clustering on the learned graph. Returns list of node index tensors."""

    def _build_guide_tree(self, subproblems, embeddings) -> str:
        """
        Build a UPGMA tree over subproblems.
        1. Mean-pool embeddings within each subproblem → (K, d_model).
        2. Compute pairwise L2 distances between subproblem means → (K, K).
        3. Run UPGMA clustering → Newick tree.
        """
```

The spectral clustering algorithm:
```
1. Build sparse adjacency matrix A from edge_index, edge_weights. Shape (N, N).
   Use torch.sparse_coo_tensor for memory efficiency.
2. Compute normalized graph Laplacian:
   D = diag(sum(A, dim=1))
   L = I - D^{-1/2} @ A @ D^{-1/2}
3. Compute the k eigenvectors of L corresponding to the k smallest eigenvalues.
   Use torch.lobpcg or scipy.sparse.linalg.eigsh.
   k = ceil(N / max_subproblem_size).
4. Run k-means on the eigenvector matrix (N, k) to get cluster assignments.
5. If any cluster > max_subproblem_size, recursively apply spectral clustering to it.
6. If any cluster < min_subproblem_size, merge with nearest neighbor cluster.
```

If spectral clustering fails (e.g., scipy not available), fallback:
```
Fallback: Simple greedy partitioning.
1. Sort nodes by degree (from the learned graph).
2. For each unassigned node, grow a cluster by BFS on the learned graph
   until cluster reaches max_subproblem_size.
3. Assign orphan nodes to nearest cluster by embedding distance.
```

---

### F.4 Diffusion Tree Generator — model/diffusion.py

#### F.4.1 Module Exports

```python
class DiffusionTreeGenerator(nn.Module):
    """
    Stage D: Generates a tree via denoising diffusion.
    
    __init__(self, config: PhyloGriffinConfig)
    
    def forward(self, sub_msa: Tensor, sub_embeddings: Tensor, 
                true_splits: List = None, true_branch_lengths: Tensor = None,
                true_pendant_lengths: Tensor = None, t: Tensor = None) -> Dict:
        """Training forward pass."""
        
    @torch.no_grad()
    def generate(self, sub_msa: Tensor, sub_embeddings: Tensor) -> str:
        """Inference: generate a tree from noise."""
        
    def _add_noise(self, splits, branch_lengths, pendant_lengths, t) -> Tuple:
        """Forward diffusion step."""
        
    def _denoise(self, noisy_splits, noisy_branch_lengths, noisy_pendant_lengths, 
                 t, sub_embeddings) -> Tuple:
        """Single denoising step."""
        
    def _discretize(self, splits_continuous, branch_lengths, pendant_lengths) -> str:
        """Convert continuous split matrix to discrete Newick tree."""

class DenoiserGNN(nn.Module):
    """
    Bipartite GNN denoiser operating on the (leaf × split) bipartite graph.
    
    __init__(self, config: PhyloGriffinConfig, d_time: int)
    
    forward(self, splits: Tensor, branch_lengths: Tensor, pendant_lengths: Tensor,
            t_emb: Tensor, seq_embeddings: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            splits: (M, n_splits_max) float — continuous split matrix
            branch_lengths: (n_splits_max,) float
            pendant_lengths: (M,) float
            t_emb: (d_time,) float — sinusoidal timestep embedding
            seq_embeddings: (M, d_model) float
            
        Returns:
            eps_splits: (M, n_splits_max) — predicted noise on splits
            eps_branch: (n_splits_max,) — predicted noise on branch lengths
            eps_pendant: (M,) — predicted noise on pendant lengths
        """

def cosine_beta_schedule(n_steps: int, s: float = 0.008) -> Tensor:
    """Cosine noise schedule. Returns beta_t for t=1..T."""

def linear_beta_schedule(n_steps: int, beta_start: float = 1e-4, 
                          beta_end: float = 0.02) -> Tensor:
    """Linear noise schedule."""
```

#### F.4.2 DenoiserGNN Architecture

```
Input:
  - splits: (M, n_splits_max) float
  - branch_lengths: (n_splits_max,) float
  - pendant_lengths: (M,) float
  - t_emb: (d_time,) float
  - seq_embeddings: (M, d_model) float

Where M = number of leaves, n_splits_max = 2 * M (fixed for the subproblem size).

Step 1: Leaf node initial features
  For each leaf i:
    leaf_feat_i = concat[seq_embeddings[i], pendant_lengths[i], t_emb_expanded]
    shape: (M, d_model + 1 + d_time)
    Project: Linear → (M, d_hidden)

Step 2: Split node initial features
  For each split j:
    # Compute participation of leaves in this split
    weights_j = softmax(|splits[:, j]|)     # (M,) — how much each leaf participates
    leaf_aggregate_j = weights_j @ leaf_feat_proj   # (d_hidden,)
    split_feat_j = concat[branch_lengths[j], leaf_aggregate_j, t_emb_expanded]
    # (n_splits_max, 1 + d_hidden + d_time)
    Project: Linear → (n_splits_max, d_hidden)

Step 3: Bipartite message passing (repeat denoiser_layers times)
  For each layer:
    # Leaves → Splits (with edge features from split matrix)
    For each split j:
      # Gather messages from all leaves
      edge_weight_ij = splits[:, j]  # (M,)
      msg_ij = Linear(2*d_hidden + 1, d_hidden)(
          concat[leaf_feat_i, split_feat_j, edge_weight_ij]
      )
      # Aggregate: weighted sum by |edge_weight|
      split_msg_j = sum_i(|edge_weight_ij| * msg_ij) / sum_i(|edge_weight_ij|)
    split_feat = RMSNorm(split_feat + split_msg)
    split_feat = RMSNorm(split_feat + MLP(split_feat))

    # Splits → Leaves
    For each leaf i:
      edge_weight_ij = splits[i, :]  # (n_splits_max,)
      msg_ji = Linear(2*d_hidden + 1, d_hidden)(
          concat[leaf_feat_i, split_feat_j, edge_weight_ij]
      )
      leaf_msg_i = sum_j(|edge_weight_ij| * msg_ji) / sum_j(|edge_weight_ij|)
    leaf_feat = RMSNorm(leaf_feat + leaf_msg)
    leaf_feat = RMSNorm(leaf_feat + MLP(leaf_feat))

Step 4: Output heads
  For each leaf i, split j:
    pair_feat = concat[leaf_feat_i, split_feat_j, splits[i,j]]
    eps_splits[i,j] = MLP(pair_feat)  # scalar
    
  For each split j:
    eps_branch[j] = MLP(split_feat_j)  # scalar
    
  For each leaf i:
    eps_pendant[i] = MLP(leaf_feat_i)  # scalar

Return: eps_splits, eps_branch, eps_pendant
```

#### F.4.3 Generation (Denoising) Loop

```python
@torch.no_grad()
def generate(self, sub_msa, sub_embeddings):
    M = sub_embeddings.shape[0]
    n_splits_max = self.config.diffusion.n_splits_max
    
    # Initialize from pure noise
    splits = torch.randn(M, n_splits_max, device=device)
    branch_lengths = torch.randn(n_splits_max, device=device)
    pendant_lengths = torch.randn(M, device=device)
    
    for t in reversed(range(1, self.n_steps + 1)):
        t_tensor = torch.tensor([t], device=device)
        t_emb = self._time_embedding(t_tensor)
        
        # Predict noise
        eps_s, eps_b, eps_p = self.denoiser(
            splits, branch_lengths, pendant_lengths, t_emb, sub_embeddings
        )
        
        # DDPM reverse step:
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        if t > 1:
            noise_s = torch.randn_like(splits)
            noise_b = torch.randn_like(branch_lengths)
            noise_p = torch.randn_like(pendant_lengths)
        else:
            noise_s, noise_b, noise_p = 0, 0, 0
        
        splits = (1 / sqrt(alpha_t)) * (
            splits - (beta_t / sqrt(1 - alpha_bar_t)) * eps_s
        ) + sqrt(beta_t) * noise_s
        
        branch_lengths = (1 / sqrt(alpha_t)) * (
            branch_lengths - (beta_t / sqrt(1 - alpha_bar_t)) * eps_b
        ) + sqrt(beta_t) * noise_b
        
        pendant_lengths = (1 / sqrt(alpha_t)) * (
            pendant_lengths - (beta_t / sqrt(1 - alpha_bar_t)) * eps_p
        ) + sqrt(beta_t) * noise_p
    
    # Discretize final splits
    return self._discretize(splits, branch_lengths, pendant_lengths)
```

#### F.4.4 Discretization Algorithm

```python
def _discretize(self, splits_continuous, branch_lengths, pendant_lengths):
    """
    Convert continuous split representation to a valid Newick tree.
    
    Steps:
    1. Threshold splits at 0: sign(splits[i,j]) → +1 or -1 for each leaf.
       For leaves where |splits[i,j]| < 0.05, assign 0 (ambiguous).
       
    2. Extract bipartitions from the discretized split matrix.
       Each column j defines a bipartition:
         left_set  = {i: sign(splits[i,j]) > 0}
         right_set = {i: sign(splits[i,j]) < 0}
       Ignore columns where either set has < 2 leaves (trivial or invalid).
       
    3. Sort bipartitions by |branch_lengths[j]| (descending).
    
    4. Greedy compatibility filter:
       Start with empty set of accepted splits.
       For each candidate split in sorted order:
         If candidate is compatible with all accepted splits:
           Accept it.
       Stop when we have M-3 accepted splits (fully resolved unrooted tree).
       
    5. Build Newick tree from accepted splits + pendant lengths.
       Use the standard algorithm: start with each leaf as its own cluster,
       then progressively join clusters based on accepted splits.
       Branch lengths for internal splits come from branch_lengths[j].
       Pendant (terminal) branch lengths come from pendant_lengths[i].
       
    6. Return Newick string.
    
    Fallback: If we get fewer than M-3 compatible splits, fill in
    remaining splits using NJ on the leaf embedding distances.
    """
```

---

### F.5 Supertree Reconciler — model/supertree.py

```python
class SupertreeReconciler(nn.Module):
    """
    Stage E: Combines K subtrees into one global tree.
    
    __init__(self, config: PhyloGriffinConfig)
    
    def forward(self, subtrees: List[Tuple[Tensor, str]], 
                guide_tree_newick: str,
                global_embeddings: Tensor) -> str:
        """
        Args:
            subtrees: List of (leaf_indices: LongTensor, subtree_newick: str) tuples
            guide_tree_newick: Newick tree over K subproblems
            global_embeddings: (N, d_model)
            
        Returns:
            global_tree_newick: str
        """
    
    def _encode_subtree(self, subtree_newick: str, n_leaves: int) -> Tensor:
        """
        Encode a subtree into a fixed-size vector.
        Uses recursive TreeLSTM.
        Returns: (d_tree,) tensor.
        """
    
    def _encode_tree_structure(self, subtree_newick: str) -> Tensor:
        """
        Alternative: encode tree topology using split-based encoding.
        For each internal split, compute a feature vector.
        Mean-pool across splits for a fixed-size embedding.
        Returns: (d_tree,) tensor.
        """

class TreeLSTM(nn.Module):
    """Recursive LSTM for encoding tree structures."""
```

#### Reconciler Architecture

```
Input processing:
  For each subproblem k (k = 1..K):
    # Subproblem token:
    sub_emb_k = mean(global_embeddings[leaf_indices_k])   # (d_model,)
    tree_enc_k = self._encode_tree_structure(subtree_newick_k)  # (d_tree,)
    quality_k = self._estimate_quality(subtree_newick_k, sub_emb_k)  # scalar
    size_norm_k = len(leaf_indices_k) / max_subproblem_size  # scalar
    
    token_k = Linear(d_model + d_tree + 2, d_supertree)(
        concat[Linear(d_model, d_supertree//2)(sub_emb_k),
               Linear(d_tree, d_supertree//2)(tree_enc_k),
               quality_k.unsqueeze(0),
               size_norm_k.unsqueeze(0)]
    )  # (d_supertree,)

  # Stack tokens: (K, d_supertree)
  
  # Positional encoding: depth in guide tree
  guide_tree_depths = compute_depth_from_root(guide_tree_newick, K)
  pos_enc = sinusoidal_embedding(guide_tree_depths, d_supertree)  # (K, d_supertree)
  
  x = token + pos_enc  # (K, d_supertree)

Transformer:
  For each layer:
    x = x + MultiHeadAttention(x, x, x, mask=guide_tree_adjacency_mask)
    x = x + FeedForward(x)
    x = LayerNorm(x)

Output heads:
  # Per-subproblem branch length scaling
  branch_scales = exp(MLP(x))  # (K,) positive scaling factors
  
  # Cross-subproblem leaf affinities (for detecting misplaced leaves)
  affinity = MLP_interaction(x)  # (K, K, d_affinity)
  
  # Per-leaf stay probabilities
  leaf_features = self._compute_leaf_features(subtrees, global_embeddings)
  # For each leaf l in subproblem k:
  stay_prob[l] = sigmoid(MLP(concat[x[k], leaf_features[l]]))

Reconciliation algorithm (non-learned, uses learned outputs):
  1. Scale all branch lengths in subtree k by branch_scales[k].
  2. For each leaf l with stay_prob[l] < 0.5:
     - Compute best destination subproblem from affinity scores.
     - Place leaf in new subproblem using ML phylogenetic placement 
       (find branch that minimally increases tree length based on embedding distance).
  3. Connect subproblems according to guide tree:
     - Treat each modified subtree as a clade.
     - Internal branches in guide tree → internal branches in global tree.
     - Branch lengths: average of the scaled pendant lengths of the two connected clades.
  4. Return global Newick tree.
```

---

### F.6 Refinement Pass — model/refinement.py

```python
class RefinementPass(nn.Module):
    """
    Stage F: Local NNI adjustments on the full tree.
    
    __init__(self, config: PhyloGriffinConfig)
    
    def forward(self, tree_newick: str, seq_embeddings: Tensor) -> str
    
    def _score_quartet(self, emb_a: Tensor, emb_b: Tensor, 
                        emb_c: Tensor, emb_d: Tensor) -> Tensor:
        """
        Score the three possible quartet topologies.
        
        For quartet (A, B, C, D):
        - Topology T1 = ((A,B),(C,D))
        - Topology T2 = ((A,C),(B,D))
        - Topology T3 = ((A,D),(B,C))
        
        Returns: (3,) tensor of quality scores (higher = better).
        """

class QuartetScorer(nn.Module):
    """
    MLP that takes embeddings from 4 subtrees and scores topologies.
    
    Input: concatenated mean embeddings of the 4 subtrees → (4*d_model,)
    Hidden: [512, 256]
    Output: (3,) — scores for the three topologies
    """

class BranchLengthPredictor(nn.Module):
    """
    MLP that predicts branch length from two subtree embeddings.
    
    Input: concat[emb_subtree_a, emb_subtree_b] → (2*d_model,)
    Output: scalar (positive, via softplus)
    """
```

#### Refinement Algorithm

```
refine(tree_newick, seq_embeddings):
    tree = parse_newick(tree_newick)
    n_leaves = len(seq_embeddings)
    
    for round in range(n_rounds):
        # Collect all internal nodes
        internal_nodes = get_internal_nodes(tree)
        
        for node in internal_nodes:
            # Get 4 subtrees around this internal edge
            A, B, C, D = get_quartet_subtrees(tree, node)
            
            # Mean-pool embeddings for each subtree
            emb_a = seq_embeddings[leaves_in(A)].mean(dim=0)
            emb_b = seq_embeddings[leaves_in(B)].mean(dim=0)
            emb_c = seq_embeddings[leaves_in(C)].mean(dim=0)
            emb_d = seq_embeddings[leaves_in(D)].mean(dim=0)
            
            # Score the 3 topologies
            scores = quartet_scorer(emb_a, emb_b, emb_c, emb_d)  # (3,)
            
            # Current topology = index 0
            best_idx = scores.argmax()
            
            if best_idx != 0 and scores[best_idx] > scores[0] + nni_margin:
                # Apply the NNI
                tree = apply_nni(tree, node, best_idx)
                
                # Update branch lengths for the 5 branches in the new quartet
                update_quartet_branch_lengths(tree, node, branch_length_predictor, 
                                               seq_embeddings)
    
    return tree_to_newick(tree)
```

---

## SECTION G: TRAINING MODULES

### G.1 Train-1: Masked Column Reconstruction — train/train_column_recon.py

```python
def train_column_reconstruction(
    model: ColumnProcessor,
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> ColumnProcessor:
    """
    Train the column processor by masking random columns and predicting them.

    Training loop:
    For each batch (msa: (N, L), mask: (N, L)):
      1. Select 15% of columns at random for masking.
         - 80% of masked columns: replace token with a special [MASK] token
           (use alphabet_size as MASK token index)
         - 10%: replace with random token
         - 10%: keep unchanged
      2. Forward pass through ColumnProcessor (Titans memory DISABLED during this stage).
         Get x: (N, L, d_model).
      3. For each masked column t:
           logits = Linear(d_model, alphabet_size)(x[:, t, :])  # (N, alphabet_size)
      4. Cross-entropy loss ONLY on masked positions.
      5. AdamW optimizer, cosine LR schedule.

    Return trained model.

    Hyperparameters from config.training:
      - lr = 1e-3
      - weight_decay = 1e-4
      - warmup_steps = 1000
      - max_steps = 100000
      - grad_clip = 1.0

    The ColumnProcessor should only include Griffin layers here.
    The Titans memory is not trained in this stage.
    Create the ColumnProcessor with titans_config=None or with a flag 
    to disable Titans during this training stage.
    """
```

### G.2 Train-2: Contrastive Phylogenetic — train/train_column_contrast.py

```python
def train_contrastive(
    model: ColumnProcessor,
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> ColumnProcessor:
    """
    Fine-tune the column processor + Titans memory for phylogenetics.

    For each batch (msa, mask, tree_newick):
      1. Forward pass through FULL ColumnProcessor (WITH Titans memory active).
         Get seq_embeddings: (N, d_model).
      2. Compute pairwise patristic distances from the true tree.
      3. Define positive pairs: pairs with patristic distance < 25th percentile.
      4. Define negative pairs: pairs with patristic distance > 75th percentile.
      5. NT-Xent contrastive loss:
         For each positive pair (i, j):
           sim_pos = cosine_sim(emb_i, emb_j) / temperature (τ=0.1)
           sim_all = [cosine_sim(emb_i, emb_k)/τ for all k]
           loss_i = -log(exp(sim_pos) / sum(exp(sim_all)))
         Average over all positive pairs.
      6. Additional triplet loss (margin=0.5):
         For random triplets (anchor i, positive j, negative k):
           loss_triplet = max(0, 0.5 + ||emb_i - emb_j||_2 - ||emb_i - emb_k||_2)
      7. Total loss = contrastive_loss + 0.5 * triplet_loss.
      8. Backprop through ALL model parameters.

    Freezing policy:
      - TokenEmbedding: frozen (no grad)
      - First 6 Griffin layers: frozen (no grad)
      - Last 6 Griffin layers: trainable
      - Titans memory: trainable

    Hyperparameters:
      - lr = 1e-4 (lower than stage 1)
      - temperature = 0.1
      - triplet_margin = 0.5
      - max_steps = 50000

    Return trained model.
    """
```

### G.3 Train-3: Graph Predictor — train/train_graph.py

```python
def train_graph_predictor(
    column_processor: ColumnProcessor,
    graph_predictor: GraphPredictor,
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> GraphPredictor:
    """
    Train the graph predictor for phylogenetic adjacency.

    For each batch (msa, mask, tree_newick):
      1. Freeze column_processor. Forward pass → embeddings (N, d_model).
      2. From true tree, extract sibling pairs and parent-child pairs → positive labels.
      3. Sample negative pairs (random distant leaves) → negative labels.
      4. For each pair, forward through graph_predictor → prob.
      5. Weighted binary cross-entropy:
         pos_weight = (N-1)/2 (to balance classes)
      6. Backprop through graph_predictor only.

    Sampling ratio per MSA: 100 positive pairs + 1000 negative pairs.

    Hyperparameters:
      - lr = 1e-3
      - max_steps = 20000
      - pos_weight as computed above

    Return trained graph_predictor.
    """
```

### G.4 Train-4: Diffusion Denoiser — train/train_diffusion.py

```python
def train_diffusion(
    column_processor: ColumnProcessor,
    diffusion: DiffusionTreeGenerator,
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> DiffusionTreeGenerator:
    """
    Train the diffusion denoiser.

    For each batch item (small MSA of M ≤ 1500 leaves, true tree):
      1. Freeze column_processor. Forward pass → embeddings (M, d_model).
      2. Extract true split matrix S_0 (M, n_splits_max), branch lengths b_0,
         and pendant lengths p_0 from the true tree.
      3. Sample random t ~ Uniform(1, T).
      4. Add noise via forward process to get S_t, b_t, p_t.
         Also store the actual noise eps_S, eps_b, eps_p that was added.
      5. Denoiser predicts: hat_eps_S, hat_eps_b, hat_eps_p.
      6. Loss:
         L_S = MSE(hat_eps_S, eps_S)
         L_b = MSE(hat_eps_b, eps_b)
         L_p = MSE(hat_eps_p, eps_p)

         # Soft Robinson-Foulds regularization:
         # Denoise one step back and compute split consistency
         S_{t-1} from the DDPM equation using predicted noise.
         For each true split, find best matching predicted split.
         L_rf = (1 - cosine_similarity).mean()

         L_total = L_S + L_b + L_p + 0.1 * L_rf

      7. Backprop through denoiser only (column_processor frozen).

    Hyperparameters:
      - lr = 2e-4
      - max_steps = 50000
      - diffusion_steps during training: randomly sample t
      - rf_loss_weight = 0.1

    Return trained diffusion generator.
    """
```

### G.5 Train-5: Supertree Reconciler — train/train_supertree.py

```python
def train_supertree(
    column_processor: ColumnProcessor,
    diffusion: DiffusionTreeGenerator,
    supertree: SupertreeReconciler,
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> SupertreeReconciler:
    """
    Train the supertree reconciler.

    For each batch item (large MSA + true tree, N = 5000-50000):
      1. Freeze column_processor and diffusion.
      2. Forward through column_processor → embeddings (N, d_model).
      3. Artificially decompose the true tree into K subproblems
         (by cutting the tree at K-1 branches). This simulates what
         Stage C will produce, but with known ground truth.
         Inject noise: randomly reassign 5-10% of leaves to wrong subproblems.
      4. For each subproblem, run diffusion.generate() → subtrees.
      5. Forward through supertree → reconciled tree.
      6. Loss:
         - Robinson-Foulds between reconciled tree and true tree (soft version)
         - Branch length MSE between reconciled tree and true tree
      7. Backprop through supertree only.

    Hyperparameters:
      - lr = 1e-4
      - max_steps = 30000
      - noise_fraction = 0.075 (fraction of leaves randomly reassigned)
      - rf_loss_weight = 1.0
      - branch_length_weight = 0.5

    Return trained supertree reconciler.
    """
```

### G.6 Train-6: Refinement Pass — train/train_refinement.py

```python
def train_refinement(
    refinement: RefinementPass,
    dataloader: DataLoader,
    config: PhyloGriffinConfig,
    device: str = "cuda",
) -> RefinementPass:
    """
    Train the refinement module.

    For each batch item (tree + embeddings):
      1. Introduce random NNIs to corrupt the tree (5-10% of internal nodes).
      2. Forward through refinement module, which attempts to detect and fix errors.
      3. Loss:
         - NNI classification: for each internal node, which of 3 topologies is correct?
           Cross-entropy loss.
         - Branch length regression: MSE between predicted and true branch lengths
           for corrected branches.
      4. Backprop through QuartetScorer and BranchLengthPredictor.

    Hyperparameters:
      - lr = 1e-3
      - max_steps = 10000
      - nni_fraction = 0.10 (fraction of internal nodes to corrupt)
      - classification_weight = 1.0
      - branch_length_weight = 0.5

    Return trained refinement module.
    """
```

---

## SECTION H: INFERENCE PIPELINE — inference.py

```python
class PhyloGriffinV3(nn.Module):
    """
    Complete model wrapping all stages.
    
    __init__(self, config: PhyloGriffinConfig)
    
    Attributes:
        column_processor: ColumnProcessor
        graph_predictor: GraphPredictor
        decomposition: HierarchicalDecomposition
        diffusion: DiffusionTreeGenerator
        supertree: SupertreeReconciler
        refinement: RefinementPass
    """

def infer_tree(
    msa: torch.Tensor,
    seq_names: List[str],
    config: PhyloGriffinConfig,
    model: PhyloGriffinV3,
    device: str = "cuda",
    chunk_size: int = 5000,
) -> str:
    """
    Full inference pipeline. Supports million-taxon scale via chunking.

    Args:
        msa: (N, L) LongTensor
        seq_names: N strings
        config: configuration
        model: loaded PhyloGriffinV3
        device: "cuda" or "cpu"
        chunk_size: process N in chunks of this size for Stage A

    Returns:
        Newick string of the inferred tree.

    Algorithm:
    Stage A (chunked):
      embeddings_list = []
      memory_state = None  # could accumulate Titans memory across chunks
      for chunk_start in range(0, N, chunk_size):
          chunk = msa[chunk_start:chunk_start+chunk_size]
          chunk_mask = (chunk != pad_idx)
          emb, mem = model.column_processor(chunk, chunk_mask)
          embeddings_list.append(emb)
      seq_embeddings = torch.cat(embeddings_list, dim=0)  # (N, d_model)

    Stage B:
      edge_index, edge_weights = model.graph_predictor.build_graph(seq_embeddings)

    Stage C:
      subproblems, guide_tree = model.decomposition(
          msa, seq_embeddings, edge_index, edge_weights
      )

    Stage D (parallel or sequential):
      subtrees = []
      for idxs, sub_msa, sub_emb in subproblems:
          subtree = model.diffusion.generate(sub_msa, sub_emb)
          subtrees.append((idxs, subtree))

    Stage E:
      full_tree = model.supertree(subtrees, guide_tree, seq_embeddings)

    Stage F:
      full_tree = model.refinement(full_tree, seq_embeddings)

    Return full_tree
    """
```

---

## SECTION I: GLUE CODE

### I.1 __init__.py files

`phylogriffin/__init__.py`:
```python
from .config import PhyloGriffinConfig
from .inference import PhyloGriffinV3, infer_tree
```

`phylogriffin/model/__init__.py`:
```python
from .column_processor import ColumnProcessor
from .graph_predictor import GraphPredictor
from .decomposition import HierarchicalDecomposition
from .diffusion import DiffusionTreeGenerator
from .supertree import SupertreeReconciler
from .refinement import RefinementPass
```

`phylogriffin/train/__init__.py`:
```python
from .train_column_recon import train_column_reconstruction
from .train_column_contrast import train_contrastive
from .train_graph import train_graph_predictor
from .train_diffusion import train_diffusion
from .train_supertree import train_supertree
from .train_refinement import train_refinement
```

### I.2 requirements.txt

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
dendropy>=4.6.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

### I.3 setup.py

```python
from setuptools import setup, find_packages

setup(
    name="phylogriffin",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "dendropy>=4.6.0",
        "tqdm>=4.65.0",
    ],
)
```

---

## SECTION J: THE COLAB NOTEBOOK — phylogriffin_v3_colab.ipynb

### J.1 Notebook Structure (Cell-by-Cell)

The notebook must contain these cells in this EXACT order. Each cell is described below with its content.

#### CELL 0: Title and Instructions (Markdown)

```markdown
# PhyloGriffin v3 — Training on Google Colab

This notebook trains the complete PhyloGriffin-v3 model for phylogenetic tree inference.

## Prerequisites
- Google Colab with GPU runtime (T4 or A100 recommended)
- Google Drive mounted for checkpoint storage
- PhyloGriffin package uploaded to your Drive or cloned from GitHub

## Training Stages
1. Masked column reconstruction (self-supervised, ~2-4 hours)
2. Contrastive phylogenetic embedding (~3-6 hours)
3. Graph predictor (~1-2 hours)
4. Diffusion denoiser (~4-8 hours)
5. Supertree reconciler (~2-4 hours)
6. Refinement pass (~1 hour)

Total estimated time: 13-25 hours. Save checkpoints frequently.
```

#### CELL 1: Install Dependencies and Mount Drive (Code)

```python
# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy scipy dendropy tqdm matplotlib

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Check GPU
!nvidia-smi

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

#### CELL 2: Clone or Copy the PhyloGriffin Package (Code)

```python
import os
import sys

# Option A: If the package is in your Google Drive
PACKAGE_PATH = "/content/drive/MyDrive/phylogriffin"
if os.path.exists(PACKAGE_PATH):
    !cp -r {PACKAGE_PATH} /content/phylogriffin
    print("Copied package from Drive")
else:
    # Option B: Clone from GitHub (if uploaded)
    !git clone https://github.com/YOUR_USERNAME/phylogriffin.git /content/phylogriffin
    print("Cloned from GitHub")

sys.path.insert(0, '/content')
os.chdir('/content/phylogriffin')
!pip install -e .

# Verify imports
from phylogriffin.config import PhyloGriffinConfig
from phylogriffin.model.column_processor import ColumnProcessor
from phylogriffin.model.graph_predictor import GraphPredictor
from phylogriffin.model.decomposition import HierarchicalDecomposition
from phylogriffin.model.diffusion import DiffusionTreeGenerator
from phylogriffin.model.supertree import SupertreeReconciler
from phylogriffin.model.refinement import RefinementPass
from phylogriffin.inference import PhyloGriffinV3, infer_tree
from phylogriffin.simulation import simulate_yule_tree, evolve_sequences
from phylogriffin.data import load_fasta, load_msa
from phylogriffin.tree_utils import newick_to_splits, robinson_foulds

print("All imports successful!")
```

#### CELL 3: Configuration (Code)

```python
# Initialize configuration
config = PhyloGriffinConfig(
    alphabet_size=21,   # protein
    gap_idx=20,
    pad_idx=21,
)

# Override for smaller model (Colab-friendly)
config.griffin.d_model = 256          # reduced from 512
config.griffin.d_rnn = 340            # ~4/3 of d_model
config.griffin.n_layers = 8           # reduced from 12
config.griffin.local_window = 512     # reduced from 1024
config.titans.d_mem = 128             # reduced from 256
config.titans.n_memory_slots = 64     # reduced from 128
config.diffusion.n_diffusion_steps = 500  # reduced for faster training
config.diffusion.denoiser_layers = 4      # reduced from 6
config.diffusion.n_splits_max = 3000
config.decomposition.max_subproblem_size = 1000  # reduced from 1500
config.training.batch_size = 4         # reduced from 8
config.training.learning_rate = 1e-3
config.training.max_steps = 50000      # reduced for Colab runtime limits

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Config: d_model={config.griffin.d_model}, n_layers={config.griffin.n_layers}")

# Create checkpoint directory in Drive
CHECKPOINT_DIR = "/content/drive/MyDrive/phylogriffin_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
```

#### CELL 4: Generate or Load Training Data (Code)

```python
"""
Generate synthetic training data.

For a real run, you would pre-generate data on a more powerful machine
and upload to Drive. For Colab, we generate small datasets on-the-fly.
"""

import numpy as np
from torch.utils.data import DataLoader
from phylogriffin.simulation import simulate_yule_tree, evolve_sequences, generate_training_batch
from phylogriffin.data import MSADataset, ContrastiveDataset

# Parameters for Colab-friendly data generation
N_TRAIN_MSAS = 200       # number of MSAs for stage 1
N_TRAIN_TREE = 100       # number of MSAs with trees for stage 2+
N_LEAVES_MIN = 50
N_LEAVES_MAX = 300
N_SITES_MIN = 200
N_SITES_MAX = 1000

# Generate data directory on Drive
DATA_DIR = "/content/drive/MyDrive/phylogriffin_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Check if data already exists
existing_files = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
if len(existing_files) < N_TRAIN_MSAS * 2:
    print("Generating training data...")
    # Generate MSAs only (no trees needed for stage 1)
    for i in range(N_TRAIN_MSAS):
        n_leaves = np.random.randint(N_LEAVES_MIN, N_LEAVES_MAX)
        n_sites = np.random.randint(N_SITES_MIN, N_SITES_MAX)
        tree = simulate_yule_tree(n_leaves, seed=i)
        msa, names = evolve_sequences(tree, n_sites, model="JTT", alpha=1.0, seed=i*100)
        
        # Save as FASTA
        fasta_path = os.path.join(DATA_DIR, f"msa_{i:05d}.fa")
        with open(fasta_path, 'w') as f:
            for j, name in enumerate(names):
                aa_str = ''.join([_idx_to_aa(idx) for idx in msa[j].tolist()])
                f.write(f">{name}\n{aa_str}\n")
        
        # Save tree
        tree_path = os.path.join(DATA_DIR, f"tree_{i:05d}.nwk")
        with open(tree_path, 'w') as f:
            f.write(tree)
        
        if (i+1) % 20 == 0:
            print(f"Generated {i+1}/{N_TRAIN_MSAS} examples")
    print("Data generation complete!")
else:
    print(f"Found {len(existing_files)//2} existing examples in {DATA_DIR}")

# Helper function for amino acid indexing
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
def _idx_to_aa(idx):
    if idx < 20:
        return AA_ORDER[idx]
    return '-'
```

#### CELL 5: Initialize Model (Code)

```python
# Create the column processor
column_processor = ColumnProcessor(config).to(device)

# Count parameters
n_params = sum(p.numel() for p in column_processor.parameters())
print(f"Column processor parameters: {n_params:,}")

# Initialize other components (they'll be trained later)
graph_predictor = GraphPredictor(
    d_model=config.griffin.d_model,
    hidden_dims=config.graph.predictor_hidden,
).to(device)

diffusion = DiffusionTreeGenerator(config).to(device)

supertree = SupertreeReconciler(config).to(device)

refinement = RefinementPass(config).to(device)

# Count total parameters
total_params = (
    sum(p.numel() for p in column_processor.parameters()) +
    sum(p.numel() for p in graph_predictor.parameters()) +
    sum(p.numel() for p in diffusion.parameters()) +
    sum(p.numel() for p in supertree.parameters()) +
    sum(p.numel() for p in refinement.parameters())
)
print(f"Total parameters (all components): {total_params:,}")
```

#### CELL 6: Train Stage 1 — Masked Column Reconstruction (Code)

```python
"""
TRAINING STAGE 1: Masked Column Reconstruction
-----------------------------------------------
Self-supervised pre-training of the Griffin column processor.
No trees required. No Titans memory used.
"""

from phylogriffin.train.train_column_recon import train_column_reconstruction
from phylogriffin.data import MSADataset

# Create dataloader
dataset = MSADataset(DATA_DIR, alphabet="protein", max_seq_len=1024)
dataloader = DataLoader(
    dataset, 
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=_collate_msa_batch,
    pin_memory=True,
)

print(f"Dataset size: {len(dataset)} MSAs")

# Train
STAGE1_PATH = os.path.join(CHECKPOINT_DIR, "stage1_column_recon.pt")

if os.path.exists(STAGE1_PATH):
    print("Loading existing Stage 1 checkpoint...")
    column_processor.load_state_dict(torch.load(STAGE1_PATH, map_location=device))
else:
    print("Starting Stage 1 training...")
    column_processor = train_column_reconstruction(
        column_processor, dataloader, config, device
    )
    torch.save(column_processor.state_dict(), STAGE1_PATH)
    print(f"Stage 1 complete! Saved to {STAGE1_PATH}")

# Helper collate function
def _collate_msa_batch(batch):
    """Pad MSAs in a batch to the same length."""
    # batch is list of dicts with keys "msa" and "mask"
    max_len = max(item["msa"].shape[1] for item in batch)
    max_n = max(item["msa"].shape[0] for item in batch)
    
    padded_msa = []
    padded_mask = []
    for item in batch:
        n, l = item["msa"].shape
        msa_pad = torch.full((max_n, max_len), config.pad_idx, dtype=torch.long)
        mask_pad = torch.zeros(max_n, max_len, dtype=torch.bool)
        msa_pad[:n, :l] = item["msa"]
        mask_pad[:n, :l] = item["mask"]
        padded_msa.append(msa_pad)
        padded_mask.append(mask_pad)
    
    return {
        "msa": torch.stack(padded_msa),
        "mask": torch.stack(padded_mask),
    }
```

#### CELL 7: Train Stage 2 — Contrastive Phylogenetic Embedding (Code)

```python
"""
TRAINING STAGE 2: Contrastive Phylogenetic Embedding
----------------------------------------------------
Fine-tune the column processor + Titans memory for phylogenetic distance.
Requires MSAs with known trees.
"""

from phylogriffin.train.train_column_contrast import train_contrastive
from phylogriffin.data import ContrastiveDataset

# Create dataloader
contrast_dataset = ContrastiveDataset(DATA_DIR, DATA_DIR, alphabet="protein")
contrast_dataloader = DataLoader(
    contrast_dataset,
    batch_size=2,  # small batch: each item is a full MSA
    shuffle=True,
    num_workers=2,
    collate_fn=_collate_contrast_batch,
)

print(f"Contrastive dataset size: {len(contrast_dataset)}")

# Freeze early layers
for i in range(6):
    for p in column_processor.griffin_layers[i].parameters():
        p.requires_grad = False

# Unfreeze Titans memory
for p in column_processor.titans.parameters():
    p.requires_grad = True

STAGE2_PATH = os.path.join(CHECKPOINT_DIR, "stage2_contrastive.pt")

if os.path.exists(STAGE2_PATH):
    print("Loading existing Stage 2 checkpoint...")
    column_processor.load_state_dict(torch.load(STAGE2_PATH, map_location=device))
else:
    print("Starting Stage 2 training...")
    column_processor = train_contrastive(
        column_processor, contrast_dataloader, config, device
    )
    torch.save(column_processor.state_dict(), STAGE2_PATH)
    print(f"Stage 2 complete! Saved to {STAGE2_PATH}")

def _collate_contrast_batch(batch):
    """Batch of (MSA, tree) pairs."""
    # Each item is already a full MSA + tree, just stack along batch dim
    msa_list = [item["msa"] for item in batch]
    mask_list = [item["mask"] for item in batch]
    tree_list = [item["tree_newick"] for item in batch]
    dist_list = [item["pairwise_distances"] for item in batch]
    
    # Pad MSAs
    max_n = max(m.shape[0] for m in msa_list)
    max_l = max(m.shape[1] for m in msa_list)
    
    msa_batch = torch.zeros(len(batch), max_n, max_l, dtype=torch.long) + config.pad_idx
    mask_batch = torch.zeros(len(batch), max_n, max_l, dtype=torch.bool)
    dist_batch = torch.zeros(len(batch), max_n, max_n)
    
    for b, (msa, mask, dist) in enumerate(zip(msa_list, mask_list, dist_list)):
        n, l = msa.shape
        msa_batch[b, :n, :l] = msa
        mask_batch[b, :n, :l] = mask
        dist_batch[b, :n, :n] = torch.from_numpy(dist)
    
    return {
        "msa": msa_batch,
        "mask": mask_batch,
        "tree_newick": tree_list,
        "pairwise_distances": dist_batch,
    }
```

#### CELL 8: Train Stage 3 — Graph Predictor (Code)

```python
"""
TRAINING STAGE 3: Graph Predictor
---------------------------------
Train the edge predictor for phylogenetic adjacency.
"""

from phylogriffin.train.train_graph import train_graph_predictor
from phylogriffin.data import GraphDataset

graph_dataset = GraphDataset(DATA_DIR, DATA_DIR, alphabet="protein")
graph_dataloader = DataLoader(
    graph_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    collate_fn=_collate_contrast_batch,  # same format
)

STAGE3_PATH = os.path.join(CHECKPOINT_DIR, "stage3_graph.pt")

if os.path.exists(STAGE3_PATH):
    print("Loading existing Stage 3 checkpoint...")
    graph_predictor.load_state_dict(torch.load(STAGE3_PATH, map_location=device))
else:
    print("Starting Stage 3 training...")
    # Freeze column processor
    for p in column_processor.parameters():
        p.requires_grad = False
    
    graph_predictor = train_graph_predictor(
        column_processor, graph_predictor, graph_dataloader, config, device
    )
    torch.save(graph_predictor.state_dict(), STAGE3_PATH)
    print(f"Stage 3 complete! Saved to {STAGE3_PATH}")
```

#### CELL 9: Train Stage 4 — Diffusion Denoiser (Code)

```python
"""
TRAINING STAGE 4: Diffusion Denoiser
------------------------------------
Train the denoising diffusion model for tree generation.
Each training item is a SMALL subtree (N ≤ 1000 leaves).
"""

from phylogriffin.train.train_diffusion import train_diffusion
from phylogriffin.data import SubproblemDataset

# Generate or load subproblem data
# For Colab, extract subproblems from the existing tree-tagged MSAs
subproblem_dataset = SubproblemDataset(DATA_DIR, DATA_DIR, 
                                        max_leaves=config.decomposition.max_subproblem_size,
                                        alphabet="protein")
subproblem_dataloader = DataLoader(
    subproblem_dataset,
    batch_size=1,  # one subproblem per batch (each is a small tree)
    shuffle=True,
    num_workers=2,
    collate_fn=_collate_subproblem_batch,
)

STAGE4_PATH = os.path.join(CHECKPOINT_DIR, "stage4_diffusion.pt")

if os.path.exists(STAGE4_PATH):
    print("Loading existing Stage 4 checkpoint...")
    diffusion.load_state_dict(torch.load(STAGE4_PATH, map_location=device))
else:
    print("Starting Stage 4 training...")
    # Column processor is frozen
    for p in column_processor.parameters():
        p.requires_grad = False
    
    diffusion = train_diffusion(
        column_processor, diffusion, subproblem_dataloader, config, device
    )
    torch.save(diffusion.state_dict(), STAGE4_PATH)
    print(f"Stage 4 complete! Saved to {STAGE4_PATH}")

def _collate_subproblem_batch(batch):
    """Subproblem batch items are small MSAs with subtrees."""
    return {
        "sub_msa": batch[0]["sub_msa"],
        "sub_mask": batch[0]["sub_mask"],
        "true_tree": batch[0]["true_tree"],
        "leaf_indices": batch[0]["leaf_indices"],
    }
```

#### CELL 10: Train Stage 5 — Supertree Reconciler (Code)

```python
"""
TRAINING STAGE 5: Supertree Reconciler
--------------------------------------
Train the transformer that stitches subtrees together.
"""

from phylogriffin.train.train_supertree import train_supertree
from phylogriffin.data import DecomposedTreeDataset

decomp_dataset = DecomposedTreeDataset(DATA_DIR, DATA_DIR, alphabet="protein")
decomp_dataloader = DataLoader(
    decomp_dataset,
    batch_size=1,  # one full tree per batch
    shuffle=True,
    num_workers=2,
    collate_fn=_collate_decomposed_batch,
)

STAGE5_PATH = os.path.join(CHECKPOINT_DIR, "stage5_supertree.pt")

if os.path.exists(STAGE5_PATH):
    print("Loading existing Stage 5 checkpoint...")
    supertree.load_state_dict(torch.load(STAGE5_PATH, map_location=device))
else:
    print("Starting Stage 5 training...")
    # Freeze column processor and diffusion
    for p in column_processor.parameters():
        p.requires_grad = False
    for p in diffusion.parameters():
        p.requires_grad = False
    
    supertree = train_supertree(
        column_processor, diffusion, supertree, decomp_dataloader, config, device
    )
    torch.save(supertree.state_dict(), STAGE5_PATH)
    print(f"Stage 5 complete! Saved to {STAGE5_PATH}")

def _collate_decomposed_batch(batch):
    """Each item is an artificially decomposed large tree."""
    item = batch[0]
    return {
        "msa": item["msa"],
        "mask": item["mask"],
        "true_tree": item["true_tree"],
        "subproblems": item["subproblems"],  # list of (indices, sub_msa, sub_emb)
        "guide_tree": item["guide_tree"],
    }
```

#### CELL 11: Train Stage 6 — Refinement Pass (Code)

```python
"""
TRAINING STAGE 6: Refinement Pass
---------------------------------
Train the NNI-based tree refinement module.
"""

from phylogriffin.train.train_refinement import train_refinement
from phylogriffin.data import ErrorTreeDataset

error_dataset = ErrorTreeDataset(DATA_DIR, DATA_DIR, alphabet="protein")
error_dataloader = DataLoader(
    error_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=_collate_error_batch,
)

STAGE6_PATH = os.path.join(CHECKPOINT_DIR, "stage6_refinement.pt")

if os.path.exists(STAGE6_PATH):
    print("Loading existing Stage 6 checkpoint...")
    refinement.load_state_dict(torch.load(STAGE6_PATH, map_location=device))
else:
    print("Starting Stage 6 training...")
    refinement = train_refinement(
        refinement, error_dataloader, config, device
    )
    torch.save(refinement.state_dict(), STAGE6_PATH)
    print(f"Stage 6 complete! Saved to {STAGE6_PATH}")

def _collate_error_batch(batch):
    return {
        "corrupted_tree": [item["corrupted_tree"] for item in batch],
        "true_tree": [item["true_tree"] for item in batch],
        "msa": torch.stack([item["msa"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "embeddings": torch.stack([item["embeddings"] for item in batch]),
    }
```

#### CELL 12: Assemble Full Model and Save (Code)

```python
"""
Assemble the full PhyloGriffinV3 model with all trained components.
"""

model = PhyloGriffinV3(config)
model.column_processor = column_processor
model.graph_predictor = graph_predictor
model.decomposition = HierarchicalDecomposition(config)
model.diffusion = diffusion
model.supertree = supertree
model.refinement = refinement

model = model.to(device)
model.eval()

FULL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "phylogriffin_v3_full.pt")
torch.save({
    "config": config.to_dict(),
    "model_state_dict": model.state_dict(),
}, FULL_MODEL_PATH)

print(f"Full model saved to {FULL_MODEL_PATH}")
```

#### CELL 13: Inference Demo (Code)

```python
"""
Run inference on a small test example to verify the pipeline works.
"""

# Generate a small test MSA
print("Generating test MSA...")
test_tree = simulate_yule_tree(50, seed=999)
test_msa, test_names = evolve_sequences(test_tree, n_sites=500, model="JTT", alpha=1.0, seed=999)

print(f"Test MSA shape: {test_msa.shape}")
print(f"True tree (first 200 chars): {test_tree[:200]}...")

# Infer tree
print("\nRunning inference...")
inferred_tree = infer_tree(
    test_msa, test_names, config, model, device=device
)

print(f"\nInferred tree (first 200 chars): {inferred_tree[:200]}...")

# Compute RF distance
true_splits = newick_to_splits(test_tree, 50)
inf_splits = newick_to_splits(inferred_tree, 50)
rf = robinson_foulds(true_splits, inf_splits)
print(f"\nRobinson-Foulds distance: {rf:.4f} (0=identical, 1=completely different)")
```

#### CELL 14: Save Inference Script for Later Use (Code)

```python
"""
Save the model and a minimal inference script to Google Drive
so you can run inference on your laptop later without re-training.
"""

INFERENCE_SCRIPT = """
#!/usr/bin/env python3
\"\"\"Standalone inference script for PhyloGriffin v3.\"\"\"
import torch
import sys
from phylogriffin.config import PhyloGriffinConfig
from phylogriffin.inference import PhyloGriffinV3, infer_tree
from phylogriffin.data import load_msa

def main():
    if len(sys.argv) < 3:
        print("Usage: python infer.py <alignment.fa> <model_checkpoint.pt>")
        sys.exit(1)
    
    msa_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    
    # Load MSA
    msa, names = load_msa(msa_path)
    print(f"Loaded MSA: {msa.shape[0]} sequences, {msa.shape[1]} columns")
    
    # Load config and model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = PhyloGriffinConfig.from_dict(checkpoint['config'])
    model = PhyloGriffinV3(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Infer tree
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    tree = infer_tree(msa, names, config, model, device=device)
    
    # Output
    print(tree)

if __name__ == "__main__":
    main()
"""

with open(os.path.join(CHECKPOINT_DIR, "infer.py"), 'w') as f:
    f.write(INFERENCE_SCRIPT)

print("Inference script saved to Drive!")
print(f"To use on your laptop: python {CHECKPOINT_DIR}/infer.py alignment.fa {FULL_MODEL_PATH}")
```

---

## SECTION K: CRITICAL IMPLEMENTATION NOTES

### K.1 Things the Other Model MUST Get Right

1. **RG-LRU parallelization**: The sequential loop over L is ACCEPTABLE. Do NOT waste time implementing a fancy associative scan in CUDA/Triton. A simple for-loop over columns in PyTorch will handle L ≤ 10K efficiently. If needed, use `torch.compile` on the loop.

2. **Titans memory reset**: The memory MUST be reset between MSAs (every forward pass, call `reset_state()`). The memory accumulates column signatures DURING one MSA's processing, but starts fresh for the next MSA.

3. **Split matrix dimensionality**: `n_splits_max = 2 * max_leaves` for a subproblem. This is the fixed number of columns in the split matrix. Inactive splits have all-zero entries. The denoiser outputs noise predictions for all `n_splits_max` columns, but the loss only applies to columns corresponding to actual splits in the true tree.

4. **Diffusion discretization**: After the denoising loop produces continuous splits, the discretization step MUST enforce tree compatibility (no conflicting bipartitions). Use the greedy algorithm described in Section F.4.4.

5. **Chunked inference at scale**: For N > 10K, the column processor MUST process sequences in chunks. Each chunk is independent (the RG-LRU operates per-sequence). The chunk size is limited by GPU VRAM.

6. **No O(N²) anywhere during inference**: Not in the column processor (per-sequence), not in the graph construction (sparse, k-NN), not in the tree generation (per-subproblem, subproblems are small).

7. **Checkpoint compatibility**: Checkpoints must include the config dict so they're portable across machines. The `save` method should serialize `config.to_dict()` alongside `model.state_dict()`.

8. **Error handling**: If the discretization produces an incompatible set of splits, fall back to NJ on embedding distances. This ensures the pipeline NEVER crashes — it always returns a valid tree.

9. **Numerical stability in RG-LRU**: The computation `a_t = sigmoid(Λ) ^ (c * r_t)` must be done in log-space to avoid underflow. Specifically: `log_a = c * r_t * F.logsigmoid(Λ)`, then `a_t = exp(log_a)`. Clamp `a_t` to [0, 1] after exponentiation.

10. **Gradient flow through Titans memory**: The memory write is gated by a surprise threshold. During training, use a straight-through estimator for the gate (always allow gradients to flow through the write path, even when surprise < threshold, but still use the gated value for the forward pass).

---

## SECTION L: TESTING CHECKLIST

The other model should verify:

- [ ] `config.py` imports without errors
- [ ] `data.py` can load a FASTA file and return correct shapes
- [ ] `tree_utils.py` can parse and generate Newick trees, extract splits, compute RF distance
- [ ] `simulation.py` can generate a tree and evolve sequences
- [ ] `ColumnProcessor` forward pass runs without error on a random MSA
- [ ] RG-LRU produces non-NaN, non-inf values after 1000 steps of recurrence
- [ ] Titans memory doesn't crash with large L
- [ ] `GraphPredictor` produces probabilities in [0, 1]
- [ ] `HierarchicalDecomposition` partitions a graph into valid subproblems
- [ ] `DiffusionTreeGenerator` can denoise from pure noise to a valid tree
- [ ] `SupertreeReconciler` produces a Newick string
- [ ] `RefinementPass` modifies the tree without breaking it
- [ ] `infer_tree` runs end-to-end on a small test case
- [ ] All training functions run without NaN losses
- [ ] Checkpoints save and load correctly with config
- [ ] The notebook runs all cells in order without manual intervention

---

## END OF SPECIFICATION

This document contains every detail needed to implement PhyloGriffin v3.
No additional information, clarification, or design decisions should be required.
If something is ambiguous, default to the simplest interpretation that maintains correctness.

The total implementation is estimated at ~3000-4000 lines of Python across 22 files.
