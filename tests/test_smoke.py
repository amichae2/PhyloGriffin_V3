"""Smoke tests: verify that key modules import and basic operations work."""

import sys
import torch
import numpy as np
from phylogriffin.config import PhyloGriffinConfig
from phylogriffin.model.column_processor import ColumnProcessor
from phylogriffin.model.graph_predictor import GraphPredictor
from phylogriffin.model.diffusion import DiffusionTreeGenerator
from phylogriffin.model.decomposition import HierarchicalDecomposition
from phylogriffin.model.supertree import SupertreeReconciler
from phylogriffin.model.refinement import RefinementPass
from phylogriffin.inference import PhyloGriffinV3
from phylogriffin.simulation import simulate_yule_tree, evolve_sequences
from phylogriffin.tree_utils import (
    parse_newick, tree_to_newick, newick_to_splits,
    robinson_foulds, corrupt_tree,
)


def test_config():
    config = PhyloGriffinConfig()
    assert config.alphabet_size == 21
    assert config.gap_idx == 20
    assert config.pad_idx == 21
    assert config.simulation.n_backbones == 300


def test_nucleotide_config():
    config = PhyloGriffinConfig.nucleotide_config()
    assert config.alphabet_size == 5


def test_newick_round_trip():
    newick = "(A:0.1,(B:0.2,C:0.3):0.4);"
    tree = parse_newick(newick)
    result = tree_to_newick(tree)
    assert "A" in result and "B" in result and "C" in result


def test_rf_distance():
    tree1 = "(A:0.1,(B:0.2,C:0.3):0.4);"
    tree2 = "(A:0.1,(B:0.2,C:0.3):0.4);"
    splits1 = newick_to_splits(tree1, 3)
    splits2 = newick_to_splits(tree2, 3)
    rf = robinson_foulds(splits1, splits2)
    assert rf == 0.0


def test_column_processor_forward():
    config = PhyloGriffinConfig()
    config.griffin.d_model = 64
    config.griffin.d_rnn = 85
    config.griffin.n_layers = 2
    config.griffin.local_window = 32
    config.titans.n_memory_slots = 16
    config.titans.d_mem = 32
    config.diffusion.n_splits_max = 100
    model = ColumnProcessor(config)
    msa = torch.randint(0, 20, (4, 30))
    mask = (msa != config.pad_idx)
    seq_emb, mem = model(msa, mask)
    assert seq_emb.shape == (4, 64)
    hidden = model.forward_hidden(msa, mask)
    assert hidden.shape == (4, 30, 64)


def test_column_processor_batched():
    config = PhyloGriffinConfig()
    config.griffin.d_model = 64
    config.griffin.d_rnn = 85
    config.griffin.n_layers = 2
    config.griffin.local_window = 32
    config.titans.n_memory_slots = 16
    config.titans.d_mem = 32
    config.diffusion.n_splits_max = 100
    model = ColumnProcessor(config)
    msa = torch.randint(0, 20, (2, 4, 30))
    seq_emb, mem = model(msa)
    assert seq_emb.shape == (2, 4, 64)


def test_graph_predictor():
    d_model = 64
    gp = GraphPredictor(d_model=d_model, hidden_dims=[128, 64])
    emb_i = torch.randn(3, d_model)
    emb_j = torch.randn(3, d_model)
    probs = gp.forward_batch(emb_i, emb_j)
    assert probs.shape == (3,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_simulation():
    tree = simulate_yule_tree(10, seed=42)
    msa, names = evolve_sequences(tree, n_sites=50, model="JTT", seed=42)
    assert msa.shape[0] == 10


def test_corrupt_tree():
    tree = "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);"
    corrupted = corrupt_tree(tree, n_swaps=1, seed=42)
    assert corrupted != tree


def test_supertree_forward():
    config = PhyloGriffinConfig()
    config.supertree.d_model = 64
    config.supertree.n_layers = 1
    config.supertree.n_heads = 2
    config.supertree.d_feedforward = 128
    config.griffin.d_model = 64
    model = SupertreeReconciler(config)
    embeddings = torch.randn(20, 64)
    subtrees = [
        (torch.arange(10), "(leaf_0:0.1,leaf_1:0.2);"),
        (torch.arange(10, 20), "(leaf_10:0.1,leaf_11:0.2);"),
    ]
    guide_tree = "(sub0:0.5,sub1:0.5);"
    tree, intermediates = model(subtrees, guide_tree, embeddings)
    assert isinstance(tree, str)
    assert "branch_scales" in intermediates
    loss = model.compute_loss(intermediates, "", 20, "cpu")
    assert isinstance(loss, torch.Tensor)


def test_refinement_forward():
    config = PhyloGriffinConfig()
    config.refinement.quartet_hidden = 64
    config.griffin.d_model = 64
    model = RefinementPass(config)
    tree = "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);"
    emb = torch.randn(4, 64)
    refined, intermediates = model(tree, emb)
    assert isinstance(refined, str)
    assert "quartet_scores" in intermediates
    assert "quartet_metadata" in intermediates
    loss = model.compute_loss(intermediates, tree, emb, "cpu")
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_refinement_loss_gradient():
    config = PhyloGriffinConfig()
    config.refinement.quartet_hidden = 64
    config.griffin.d_model = 64
    model = RefinementPass(config)
    tree = "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);"
    emb = torch.randn(4, 64)
    refined, intermediates = model(tree, emb)
    true_tree = "((A:0.1,C:0.4):0.3,(B:0.2,D:0.5):0.6);"
    loss = model.compute_loss(intermediates, true_tree, emb, "cpu")
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad, "RefinementPass.compute_loss should produce non-zero gradients"

    quartet_metadata = intermediates.get("quartet_metadata", [])
    leaf_to_idx = intermediates.get("leaf_to_idx", {})
    from phylogriffin.model.refinement import _determine_quartet_topology
    from phylogriffin.tree_utils import parse_newick
    true_tree_parsed = parse_newick(true_tree)
    if quartet_metadata:
        a_set, b_set, c_set, d_set = quartet_metadata[0]
        topology = _determine_quartet_topology(
            true_tree_parsed, a_set, b_set, c_set, d_set, leaf_to_idx
        )
        assert topology == 1, (
            f"Expected topology 1 for quartet ((A,B),(C,D)) with true tree "
            f"((A,C),(B,D)), got {topology}"
        )


def test_inference_wiring():
    config = PhyloGriffinConfig()
    config.griffin.d_model = 64
    config.griffin.d_rnn = 85
    config.griffin.n_layers = 2
    config.titans.n_memory_slots = 16
    config.titans.d_mem = 32
    config.diffusion.n_splits_max = 100
    config.supertree.d_model = 64
    config.supertree.n_layers = 1
    config.supertree.d_feedforward = 128
    m = PhyloGriffinV3(config)
    assert hasattr(m, "column_processor")
    assert hasattr(m, "supertree")
    assert hasattr(m, "refinement")


if __name__ == "__main__":
    tests = [
        test_config, test_nucleotide_config, test_newick_round_trip,
        test_rf_distance, test_column_processor_forward,
        test_column_processor_batched, test_graph_predictor,
        test_simulation, test_corrupt_tree, test_supertree_forward,
        test_refinement_forward, test_refinement_loss_gradient,
        test_inference_wiring,
    ]
    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"PASS: {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_fn.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
