# Refactoring Prompt: Structurally-Realistic Training Data for PhyloGriffin v3

## What This Document Is

You (the model reading this) have already implemented the PhyloGriffin v3 codebase from the original spec (`PHYLOGRIFFIN_V3_SPEC.md`). This document tells you how to **replace the naive training data simulation** with a realistic pipeline that generates structurally-constrained, evolutionarily-grounded protein MSAs. This is necessary because the original `simulation.py` uses site-independent substitution models (JTT/WAG/LG) that produce sequences with no secondary structure, no co-evolution, and no tertiary contact signal — meaning the Titans co-evolution memory, the contrastive embedding, the diffusion denoiser, and the graph predictor all train on data that lacks the very signal they were designed to detect.

## Files You Will Modify

| File | Action |
|------|--------|
| `phylogriffin/simulation.py` | **Major rewrite.** Replace the naive `evolve_sequences()` with the new pipeline. Add ProteinMPNN wrapper. Add structural template handling. Keep `simulate_yule_tree()`, `simulate_birth_death_tree()`, and the rate matrix functions — they're still needed. |
| `phylogriffin/data.py` | **Minor addition.** Add a `PreGeneratedDataset` class that loads from the pre-generated FASTA cache. |
| `phylogriffin/config.py` | **Minor addition.** Add `SimulationConfig` dataclass with paths and settings for the new pipeline. |
| `requirements.txt` | **Add** the ProteinMPNN dependency and the PDB download utility. |
| `phylogriffin_v3_colab.ipynb` | **Modify Cells 4, 5, 6, 7, 8, 9, 10.** Cells 4 gets a new "pre-generation" section. Training cells are updated to use the new data pipeline. |

No model architecture files need to change. No training logic files need to change beyond the data loading.

---

## The Core Insight

The original `evolve_sequences()` treats every alignment column as statistically independent. Real proteins have **position-position dependencies** from three-dimensional structure. The correct way to generate training data that exercises every component of PhyloGriffin v3 is:

1. **Start from real protein 3D structures** (PDB files) — these provide the ground-truth structural constraints.
2. **Use ProteinMPNN** to generate diverse amino acid sequences that all fold to each structure — these sequences naturally contain secondary structure preferences, solvent exposure patterns, and co-evolving residue pairs.
3. **Use Pyvolve** to simulate realistic evolution along known phylogenetic trees using these sequences as leaves — this adds substitution noise along branches while preserving the structural signal.
4. **Pre-generate the ProteinMPNN sequence pools once**, cache them, and use Pyvolve on-the-fly during training — this decouples the slow structural generation from the fast evolutionary simulation.

---

## Step-by-Step Implementation

### Step 1: Add `SimulationConfig` to `config.py`

Add this dataclass inside `config.py` (alongside the existing config classes):

```python
@dataclass
class SimulationConfig:
    """Configuration for structurally-realistic training data generation."""
    # PDB backbone source
    pdb_cache_dir: str = "/content/drive/MyDrive/phylogriffin_data/pdb_cache"
    n_backbones: int = 300               # Number of PDB structures to use
    pdb_resolution_max: float = 2.5      # Maximum resolution in Angstroms
    pdb_length_min: int = 50             # Minimum chain length
    pdb_length_max: int = 500            # Maximum chain length
    
    # ProteinMPNN settings
    mpnn_sequences_per_temp: int = 100   # Sequences generated per temperature
    mpnn_temperatures: List[float] = field(default_factory=lambda: [0.2, 0.5])
    mpnn_model_name: str = "v_48_020"    # ProteinMPNN model variant
    mpnn_batch_size: int = 1
    
    # Pre-generated cache
    pregen_dir: str = "/content/drive/MyDrive/phylogriffin_data/pregen_fasta"
    
    # Pyvolve settings
    pyvolve_model: str = "JTT"           # "JTT", "WAG", "LG" for protein
    pyvolve_alpha: float = 1.0           # Gamma shape for rate heterogeneity
    pyvolve_n_categories: int = 4        # Discrete rate categories
    
    # Data split
    train_frac: float = 0.8              # Fraction of backbones for training
    val_frac: float = 0.1                # Fraction for validation
    test_frac: float = 0.1               # Fraction for testing (held out backbones)
```

Add `simulation: SimulationConfig = field(default_factory=SimulationConfig)` to `PhyloGriffinConfig`.

### Step 2: Download PDB Backbones

Create a new function in `simulation.py`:

```python
def download_representative_pdbs(
    output_dir: str,
    n_structures: int = 300,
    resolution_max: float = 2.5,
    length_min: int = 50,
    length_max: int = 500,
) -> List[str]:
    """
    Download a curated, non-redundant set of PDB structures.

    Strategy (in priority order):
    
    1. Try the RCSB PDB REST API with a search query:
       https://search.rcsb.org/rcsbsearch/v2/query
       
       Query JSON:
       {
         "query": {
           "type": "group",
           "logical_operator": "and",
           "nodes": [
             {"type": "terminal", "service": "text", "parameters": {
               "attribute": "rcsb_entry_info.resolution_combined",
               "operator": "less_or_equal",
               "value": 2.5
             }},
             {"type": "terminal", "service": "text", "parameters": {
               "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
               "operator": "equals",
               "value": 1
             }},
             {"type": "terminal", "service": "text", "parameters": {
               "attribute": "rcsb_entry_info.polymer_entity_count_protein",
               "operator": "greater_or_equal",
               "value": 1
             }}
           ]
         },
         "return_type": "entry",
         "request_options": {
           "results_content_type": ["experimental"],
           "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
           "results_max": 1000
         }
       }
       
       This gives ~1000 PDB IDs sorted by resolution. Filter client-side
       by chain length (50-500 residues). Take the first n_structures.

    2. FALLBACK: If the REST API is unreachable, use a hardcoded list of
       50 well-known, diverse PDB structures covering all major fold classes.
       
       These 50 are:
       1A3N, 1A6M, 1A8E, 1ABA, 1AIM, 1AKZ, 1AMM, 1APM, 1AQZ, 1ARB,
       1ATN, 1AUO, 1B7F, 1BDO, 1BFG, 1BGF, 1BMF, 1BOB, 1BPI, 1BTN,
       1C52, 1C75, 1CCR, 1CEW, 1CHD, 1CKA, 1CQY, 1CRN, 1CSE, 1CTF,
       1CUK, 1D4T, 1D7P, 1DDT, 1DFN, 1DHN, 1DIN, 1DKZ, 1DOZ, 1DVR,
       1E0L, 1E2A, 1E4M, 1E6V, 1EAJ, 1ECP, 1EDM, 1EGW, 1EJG, 1EM8
       
       These span TIM barrels, immunoglobulin folds, Rossmann folds, 
       globins, beta-propellers, jelly rolls, all-alpha bundles, etc.

    3. For each PDB ID, download the biological assembly CIF or PDB file:
       https://files.rcsb.org/download/{pdb_id}.pdb
       
       Or use the mmCIF format:
       https://files.rcsb.org/download/{pdb_id}.cif
    
    Implementation notes:
    - Use `requests` for HTTP calls. Add retry logic (3 attempts, exponential backoff).
    - Save each structure as `{output_dir}/{pdb_id}.pdb`.
    - Return the list of successfully downloaded PDB IDs.
    - If fewer than n_structures download successfully, use whatever is available.
    - Print progress every 10 structures.
    """
```

**IMPORTANT**: The hardcoded fallback list of 50 PDB IDs must be embedded in the function. These 50 structures span all major SCOP/CATH fold classes and guarantee the pipeline works even without internet access to the search API.

### Step 3: ProteinMPNN Sequence Generation

Create a new function in `simulation.py`:

```python
def generate_sequences_with_proteinmpnn(
    pdb_path: str,
    output_fasta_path: str,
    temperatures: List[float] = [0.2, 0.5],
    num_seq_per_temp: int = 100,
    model_name: str = "v_48_020",
    seed: int = 0,
) -> int:
    """
    Generate diverse amino acid sequences for a protein backbone using ProteinMPNN.

    This function shells out to ProteinMPNN's protein_mpnn_run.py.
    It does NOT import ProteinMPNN as a library — it runs it as a subprocess,
    which is more robust to path/environment issues in Colab.

    Args:
        pdb_path: Path to input PDB file
        output_fasta_path: Path to write output FASTA
        temperatures: List of sampling temperatures (higher = more diversity)
        num_seq_per_temp: Number of sequences per temperature
        model_name: ProteinMPNN model variant
        seed: Random seed

    Returns:
        Total number of sequences generated

    What it does internally:
    
    1. Locate the ProteinMPNN installation.
       Check in order:
         a. /content/ProteinMPNN/protein_mpnn_run.py  (Colab default)
         b. The directory specified by PROTEINMPNN_PATH env var
         c. Search the Python path for protein_mpnn_run
    
    2. Parse the PDB to JSONL format using ProteinMPNN's helper:
       python helper_scripts/parse_multiple_chains.py \
           --input_path={pdb_path} \
           --output_path={temp_jsonl}
    
    3. Run ProteinMPNN for each temperature:
       python protein_mpnn_run.py \
           --jsonl_path={temp_jsonl} \
           --chain_id_jsonl="" \
           --fixed_positions_jsonl="" \
           --sampling_temp="{T}" \
           --num_seq_per_target={num_seq_per_temp} \
           --batch_size=1 \
           --model_name={model_name} \
           --out_folder={temp_output_dir} \
           --seed={seed}
    
    4. Collect all generated .fa files from the output directory,
       concatenate them into a single FASTA file at output_fasta_path.
    
    5. The output FASTA format:
       >backbone={pdb_id}_T={temp}_sample={i}_score={score}
       MSEQUENCE...
       
       Each header encodes the source backbone, sampling temperature,
       sample index, and ProteinMPNN confidence score.
    
    6. Return the total count of generated sequences.

    Error handling:
    - If ProteinMPNN is not installed, raise RuntimeError with instructions.
    - If a PDB has multiple chains, design all of them.
    - If generation fails for one temperature, continue with others.
    - If no sequences are generated at all, return 0.

    Colab note: ProteinMPNN downloads its pre-trained weights (~250 MB) on 
    first use from the GitHub releases page. This only happens once.
    """
```

### Step 4: Pre-Generation (Run Once, Cache Forever)

Create a new function in `simulation.py`:

```python
def pregenerate_all_backbones(
    pdb_dir: str,
    output_dir: str,
    config: SimulationConfig,
) -> Dict[str, int]:
    """
    Generate ProteinMPNN sequences for all downloaded PDB backbones.
    
    This is intended to run ONCE before training begins. Results are saved
    as FASTA files in output_dir and loaded during training.

    Args:
        pdb_dir: Directory containing downloaded PDB files
        output_dir: Directory to save generated FASTA files
        config: SimulationConfig with ProteinMPNN settings
    
    Returns:
        Dict mapping pdb_id -> number of sequences generated
    
    Algorithm:
    1. List all .pdb files in pdb_dir.
    2. For each PDB file:
       a. Check if output FASTA already exists in output_dir. If yes, skip.
       b. Call generate_sequences_with_proteinmpnn().
       c. Move/copy the output to output_dir/{pdb_id}.fa.
       d. Print progress: "[i/N] {pdb_id}: {n_seqs} sequences generated"
    3. Write a metadata file: output_dir/manifest.json with:
       {
         "generated_at": "<timestamp>",
         "n_backbones": N,
         "total_sequences": M,
         "temperatures": [0.2, 0.5],
         "sequences_per_temp": 100,
         "per_backbone": {pdb_id: n_seqs, ...}
       }
    4. Return the counts dict.
    
    Estimated runtime on Colab T4: ~30-60 seconds per backbone.
    For 300 backbones: ~2.5-5 hours. Run as a separate Colab session
    if needed, or overnight.
    """
```

### Step 5: Pyvolve-Based Tree Evolution

Create a new function in `simulation.py`:

```python
def evolve_sequences_from_pool(
    fasta_pool_path: str,
    n_leaves: int,
    n_sites: int,
    tree_type: str = "yule",
    birth_rate: float = 1.0,
    death_rate: float = 0.5,
    model: str = "JTT",
    alpha: float = 1.0,
    n_categories: int = 4,
    seed: int = None,
) -> Tuple[torch.Tensor, List[str], str]:
    """
    Generate one training example: MSA + true tree, with structural realism.

    This is the main function called during training. It combines a
    pre-generated ProteinMPNN sequence pool with Pyvolve-based
    evolutionary simulation.

    Algorithm:
    
    1. Load the pre-generated FASTA file from fasta_pool_path.
       The FASTA contains ~200 sequences that all fold to the same backbone.
       These sequences naturally contain structural constraints.
    
    2. Randomly select n_leaves sequences from the pool.
       If the pool has fewer than n_leaves sequences, sample with replacement
       (this is rare — typical pools have ~200 sequences).
       Store the selected leaf sequences.
    
    3. Determine the ancestral sequence:
       - Compute the position-wise consensus (most frequent amino acid at
         each column) across the selected leaf sequences.
       - For ties, sample from the tied amino acids proportional to their
         frequency in the pool.
       - This consensus approximates the maximum-likelihood ancestral state.
    
    4. Simulate a random phylogenetic tree:
       If tree_type == "yule":
           tree_newick = simulate_yule_tree(n_leaves, birth_rate, seed)
       If tree_type == "birth_death":
           tree_newick = simulate_birth_death_tree(n_leaves, birth_rate, death_rate, seed)
       
       This returns a Newick tree with the leaf sequences as tip labels.
       The tree branch lengths are in expected substitutions per site.
    
    5. Create a Pyvolve Partition object:
       from pyvolve import Partition, Model
       
       # Select the substitution model
       if model == "JTT":
           matrix = jtt_rate_matrix()
       elif model == "WAG":
           matrix = wag_rate_matrix()
       elif model == "LG":
           matrix = lg_rate_matrix()
       
       # Compute stationary frequencies from the matrix
       # (eigenvector corresponding to eigenvalue 0)
       
       pyvolve_model = Model("custom", {"matrix": matrix, 
                                         "freqs": equilibrium_frequencies,
                                         "alpha": alpha,
                                         "num_categories": n_categories})
       
       partition = Partition(models=pyvolve_model, size=n_sites)
    
    6. Evolve from the ancestral sequence:
       - Pyvolve requires an "ancestral" or "root" sequence.
       - Use the consensus as the root sequence.
       - Pyvolve will simulate substitutions along each branch according
         to the tree and the substitution model.
       - Rate heterogeneity (alpha, n_categories) controls site-to-site
         rate variation.
       
       result = partition.evolve(
           tree=tree_newick,
           root_sequence=ancestral_sequence,
           seqfile=None,  # Don't write to disk
           return_sequences=True,
       )
       
       # result contains the evolved leaf sequences
    
    7. Align and tokenize:
       - Extract the evolved leaf sequences from the Pyvolve result.
       - They should already be aligned (same length).
       - Tokenize using the standard amino acid alphabet:
         A=0, R=1, N=2, D=3, C=4, Q=5, E=6, G=7, H=8, I=9,
         L=10, K=11, M=12, F=13, P=14, S=15, T=16, W=17, Y=18, V=19
         Gap = 20
    
    8. Return:
       - msa: LongTensor of shape (n_leaves, n_sites)
       - seq_names: List of leaf names from the tree
       - tree_newick: The true tree with branch lengths
    
    Important: This function is called DURING training. It must be fast
    (~0.5-2 seconds per example). The expensive ProteinMPNN step already
    happened during pre-generation.
    
    The structural signal flows through as follows:
      PDB backbone → ProteinMPNN → leaf sequence pool
                                         ↓
      consensus → ancestral → Pyvolve evolution → MSA
                                         ↓
      Positional amino acid preferences, co-evolving pairs, and secondary
      structure propensities are PRESERVED in the leaf pool and partially
      reflected in the consensus → partially degraded by evolution.
      
      The model must learn to reconstruct the tree from the imperfect
      signal in the MSA — exactly as in real phylogenetics.
    """
```

**Pyvolve installation note**: Pyvolve is pip-installable (`pip install pyvolve`). It uses numpy only, no GPU needed. It runs in ~0.1-0.5 seconds for typical protein simulations (500 sites, 100 leaves).

### Step 6: Training Data Pipeline Functions

Add these convenience functions to `simulation.py`:

```python
def build_pregen_index(pregen_dir: str) -> Dict:
    """
    Build an index of all pre-generated FASTA files for fast random access.

    Reads manifest.json from pregen_dir. Returns a dict:
    {
        "files": ["path/to/backbone_1abc.fa", ...],
        "n_sequences": [200, 195, ...],    # parallel arrays
        "pdb_ids": ["1abc", "2def", ...],
    }

    This index is loaded once at the start of training.
    """

def random_training_example(
    pregen_index: Dict,
    n_leaves_range: Tuple[int, int] = (50, 500),
    n_sites_range: Tuple[int, int] = (200, 1500),
    model: str = "JTT",
    alpha: float = 1.0,
    seed: int = None,
) -> Tuple[torch.Tensor, List[str], str, str]:
    """
    Generate one random training example for any training stage.

    Args:
        pregen_index: Index from build_pregen_index()
        n_leaves_range: (min, max) number of leaves to sample
        n_sites_range: (min, max) number of sites
        model: Substitution model for Pyvolve
        alpha: Gamma shape parameter
        seed: Random seed

    Returns:
        msa: (N, L) tensor
        seq_names: list of N strings
        tree_newick: ground truth tree
        backbone_id: which PDB backbone was used (for tracking)

    Usage during training:
        # Train-1 (masked reconstruction): just need MSAs
        msa, _, _, _ = random_training_example(...)
        
        # Train-2 (contrastive): need MSAs + true trees
        msa, names, tree, _ = random_training_example(...)
        
        # Train-3 (graph predictor): same as Train-2
        
        # Train-4 (diffusion): smaller subproblems
        msa, _, tree, _ = random_training_example(
            n_leaves_range=(100, 1000), ...
        )
    """

def training_batch_generator(
    pregen_index: Dict,
    batch_size: int,
    n_leaves_range: Tuple[int, int],
    n_sites_range: Tuple[int, int],
    model: str = "JTT",
    alpha: float = 1.0,
    infinite: bool = True,
) -> Iterator[Dict]:
    """
    Infinite (or finite) generator of training batches.
    
    Each batch is a dict of collated examples, ready for DataLoader-style iteration.
    Automatically pads MSAs to the same dimensions within a batch.

    Yields:
        {
            "msa": LongTensor (batch, max_N, max_L),
            "mask": BoolTensor (batch, max_N, max_L),
            "tree_newick": [str] * batch,
            "backbone_ids": [str] * batch,
        }
    """
```

### Step 7: Modify `data.py`

Add the `PreGeneratedDataset` class:

```python
class PreGeneratedDataset(torch.utils.data.Dataset):
    """
    Dataset that generates structurally-realistic MSAs on-the-fly.

    Each __getitem__ call generates a fresh MSA by:
    1. Randomly selecting a pre-generated ProteinMPNN sequence pool
    2. Sampling N leaves from the pool
    3. Running Pyvolve evolution along a random tree

    This means the dataset is INFINITE — no two calls produce the same MSA.
    Set __len__ to a fixed number for one epoch, or use an infinite generator.

    __init__(
        self,
        pregen_dir: str,           # Directory with pre-generated FASTA files
        n_examples_per_epoch: int = 5000,
        n_leaves_range: Tuple[int, int] = (50, 500),
        n_sites_range: Tuple[int, int] = (200, 1500),
        model: str = "JTT",
        alpha: float = 1.0,
        include_trees: bool = True,  # False for Train-1 (self-supervised)
        split: str = "train",        # "train", "val", "test"
        train_val_test: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    )

    __getitem__(self, idx) -> Dict:
        # idx is ignored — each call generates a random example
        # (but seeded deterministically for reproducibility)
        # The split parameter controls which subset of backbones to use.
        
    __len__(self) -> int:
        return self.n_examples_per_epoch
    """
```

### Step 8: Modify the Colab Notebook

#### Cell 4 Replacement: "Setup Training Data"

The old Cell 4 used naive `evolve_sequences()`. Replace with:

```python
"""
SETUP: Pre-generate or load ProteinMPNN sequence pools.

This cell runs ONCE. It either:
- Downloads PDB structures and runs ProteinMPNN (first time, ~3-5 hours)
- Loads existing pre-generated data from Drive (subsequent runs, ~1 minute)
"""

import os
from phylogriffin.simulation import (
    download_representative_pdbs,
    generate_sequences_with_proteinmpnn,
    pregenerate_all_backbones,
    build_pregen_index,
)
from phylogriffin.config import SimulationConfig

sim_config = config.simulation

# --- Step 1: Clone ProteinMPNN if needed ---
PROTEINMPNN_PATH = "/content/ProteinMPNN"
if not os.path.exists(PROTEINMPNN_PATH):
    print("Cloning ProteinMPNN...")
    !git clone https://github.com/dauparas/ProteinMPNN.git {PROTEINMPNN_PATH}
    # Install ProteinMPNN dependencies (PyTorch already installed)
    !pip install pyvolve  # for the evolutionary simulation

# --- Step 2: Download PDB backbones if needed ---
os.makedirs(sim_config.pdb_cache_dir, exist_ok=True)
existing_pdbs = [f for f in os.listdir(sim_config.pdb_cache_dir) if f.endswith('.pdb')]

if len(existing_pdbs) < sim_config.n_backbones:
    print(f"Downloading {sim_config.n_backbones} PDB structures...")
    pdb_ids = download_representative_pdbs(
        sim_config.pdb_cache_dir,
        n_structures=sim_config.n_backbones,
        resolution_max=sim_config.pdb_resolution_max,
        length_min=sim_config.pdb_length_min,
        length_max=sim_config.pdb_length_max,
    )
    print(f"Downloaded {len(pdb_ids)} PDB structures")
else:
    print(f"Found {len(existing_pdbs)} existing PDB structures")

# --- Step 3: Pre-generate ProteinMPNN sequences if needed ---
os.makedirs(sim_config.pregen_dir, exist_ok=True)

manifest_path = os.path.join(sim_config.pregen_dir, "manifest.json")
if not os.path.exists(manifest_path):
    print(f"Pre-generating ProteinMPNN sequences for all backbones...")
    print(f"This will take approximately {sim_config.n_backbones * 0.5:.0f}-{sim_config.n_backbones * 1:.0f} minutes.")
    counts = pregenerate_all_backbones(
        sim_config.pdb_cache_dir,
        sim_config.pregen_dir,
        sim_config,
    )
    n_total = sum(counts.values())
    print(f"Pre-generation complete! {n_total} total sequences across {len(counts)} backbones.")
else:
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Loaded existing pre-generated data: {manifest['n_backbones']} backbones, "
          f"{manifest['total_sequences']} total sequences")

# --- Step 4: Build the index for training ---
pregen_index = build_pregen_index(sim_config.pregen_dir)
print(f"Training data index built: {len(pregen_index['files'])} FASTA pools available")
```

#### Cells 6-11 (Training Cells): Update Data Loading

Replace each training cell's data loading section. Instead of the old `MSADataset`/`ContrastiveDataset`, use:

```python
# For Train-1 (masked column reconstruction — only needs MSAs, not trees):
from phylogriffin.data import PreGeneratedDataset

dataset = PreGeneratedDataset(
    pregen_dir=sim_config.pregen_dir,
    n_examples_per_epoch=5000,
    n_leaves_range=(50, 500),
    n_sites_range=(200, 1500),
    model=sim_config.pyvolve_model,
    alpha=sim_config.pyvolve_alpha,
    include_trees=False,   # No trees needed for self-supervised training
    split="train",
)

dataloader = DataLoader(dataset, batch_size=config.training.batch_size, ...)
```

```python
# For Train-2 through Train-6 (need trees):
dataset = PreGeneratedDataset(
    pregen_dir=sim_config.pregen_dir,
    n_examples_per_epoch=5000,
    n_leaves_range=(50, 500),    # Smaller for diffusion: (100, 1000)
    n_sites_range=(200, 1500),
    model=sim_config.pyvolve_model,
    alpha=sim_config.pyvolve_alpha,
    include_trees=True,
    split="train",
)
```

For Train-4 (diffusion), use `n_leaves_range=(100, 1000)` to keep subproblem sizes manageable.

For Train-5 (supertree), use `n_leaves_range=(500, 5000)` with a note that very large examples may need gradient accumulation.

### Step 9: Update `requirements.txt`

Add:
```
pyvolve>=1.1.0
requests>=2.28.0
```

ProteinMPNN is NOT added to requirements.txt because it's cloned via git in the notebook.

---

## What Gets Removed

The following functions in `simulation.py` become **dead code** and should be removed or marked deprecated:

- `evolve_sequences()` (the ORIGINAL site-independent version) — replaced by `evolve_sequences_from_pool()`
- `generate_training_batch()` (the original) — replaced by `training_batch_generator()`
- Any hand-crafted structural template generation code from the original spec's "Level 2" simulation — replaced by ProteinMPNN

**Keep these functions** — they're still needed:
- `simulate_yule_tree()`
- `simulate_birth_death_tree()`
- `jtt_rate_matrix()`, `wag_rate_matrix()`, `lg_rate_matrix()`, `gtr_rate_matrix()`
- The amino acid alphabet constants

---

## How This Changes Training Behavior

| Original | New |
|----------|-----|
| Each training example uses a random tree + independent JTT/WAG/LG evolution | Each training example uses a random tree + Pyvolve evolution **from ProteinMPNN-generated leaf sequences** |
| No structural signal in the data | Position-specific amino acid preferences, secondary structure propensities, and co-evolving pairs are all present |
| Titans memory has nothing to learn | Titans memory can detect genuine co-evolving positions from the ProteinMPNN structural constraints |
| Contrastive training distinguishes random noise from tree signal | Contrastive training must distinguish genuine phylogenetic similarity from structural similarity (both are present — a harder, more realistic task) |
| Model can cheat by memorizing site-independent patterns | Model must learn generalizable phylogenetic principles that work across hundreds of structurally distinct protein families |

---

## Validation Checklist

After implementing, verify:

- [ ] `download_representative_pdbs()` successfully downloads at least 50 PDB files (or uses the hardcoded list)
- [ ] ProteinMPNN generates ≥150 sequences per backbone (some may fail at extreme temperatures)
- [ ] `manifest.json` is created with correct counts
- [ ] `evolve_sequences_from_pool()` produces valid tokenized MSAs (no out-of-range indices)
- [ ] Pyvolve evolution correctly respects the tree topology (RF distance between output and input tree is 0 when run with seed)
- [ ] Generated MSAs have realistic amino acid distributions (not all alanines, not uniform)
- [ ] The PreGeneratedDataset produces different MSAs on each `__getitem__` call (even with the same idx)
- [ ] Training runs without NaN losses on the new data
- [ ] The Titans memory module shows non-zero surprise values during training (indicating it's detecting real co-variation)
- [ ] The full pipeline runs in Colab without OOM errors

---

## Estimated Impact

With this refactoring, the training data now contains genuine structural constraints that exercise every component of the PhyloGriffin v3 architecture:

- The **Titans co-evolution memory** has real contacting residue pairs to detect — salt bridges, hydrophobic packing, disulfide bonds — rather than being dead weight.
- The **contrastive embedding** must learn to separate phylogenetic distance from structural similarity, just as real phylogenetic methods must separate homology from convergence.
- The **diffusion denoiser** sees trees where branch lengths reflect both evolutionary time AND structural constraint (slow-evolving buried core vs. fast-evolving surface loops).
- The **graph predictor** cannot simply connect sequences by structural class — it must learn genuine phylogenetic adjacency.

Without this refactoring, the Titans memory and learned graph predictor would train on data that lacks the signal they were designed to detect. With it, every architectural investment pays off.

---

## If Something Goes Wrong

| Problem | Solution |
|---------|----------|
| RCSB API is down | The hardcoded 50-PDB fallback list is used automatically |
| ProteinMPNN OOM on Colab | Reduce `mpnn_batch_size` to 1, use `--max_length` flag, or skip the longest backbones |
| ProteinMPNN not installed | The notebook cell clones it from GitHub; verify internet access |
| Pyvolve produces garbled sequences | Check that the root sequence uses valid amino acid characters only (no gaps at root) |
| Pre-generation takes too long | Run Cell 4 in a separate Colab session overnight, save to Drive, skip on subsequent runs |
| Training loss doesn't decrease | The data may be too hard. Start with small N (20-50 leaves) and short sequences (200 sites), then scale up |
