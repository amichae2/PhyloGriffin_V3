"""
PhyloGriffin v3 -- Configuration dataclasses.
All hyperparameters, constants, and paths live here.
No other file defines magic numbers.
"""

import os
from dataclasses import dataclass, field
from typing import Any


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
    pattern: tuple[int, int] = (2, 1)


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
    predictor_hidden: list[int] = field(default_factory=lambda: [512, 256])
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
class SimulationConfig:
    """Configuration for structurally-realistic training data generation."""

    pdb_cache_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.expanduser("~"), ".cache", "phylogriffin", "pdb_cache"
        )
    )
    n_backbones: int = 300
    pdb_resolution_max: float = 2.5
    pdb_length_min: int = 50
    pdb_length_max: int = 500

    mpnn_sequences_per_temp: int = 100
    mpnn_temperatures: list[float] = field(default_factory=lambda: [0.2, 0.5])
    mpnn_model_name: str = "v_48_020"
    mpnn_batch_size: int = 1

    pregen_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.expanduser("~"), ".cache", "phylogriffin", "pregen_fasta"
        )
    )

    pyvolve_model: str = "JTT"
    pyvolve_alpha: float = 1.0
    pyvolve_n_categories: int = 4

    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1


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
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def to_dict(self) -> dict[str, Any]:
        import dataclasses

        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PhyloGriffinConfig":
        return cls(**d)

    @classmethod
    def nucleotide_config(cls) -> "PhyloGriffinConfig":
        return cls(alphabet_size=5, gap_idx=4, pad_idx=5)
