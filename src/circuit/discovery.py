"""Circuit discovery using Sparse Feature Circuits (ICLR 2025).

Reference: Marks, Rager et al. (ICLR 2025) — Sparse Feature Circuits:
           Discovering and Editing Interpretable Causal Graphs in LLMs.

Pipeline:
    1. Load SAE features for target model
    2. Identify subgraph of features causally implicated in a behavior
    3. Build and verify the circuit
    4. Edit circuit to modify model behavior
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import networkx as nx


@dataclass
class CircuitEdge:
    """A causal edge between two SAE features."""
    source_layer: int
    source_feature: int
    target_layer: int
    target_feature: int
    attribution_score: float = 0.0


@dataclass
class SparseCircuit:
    """A sparse feature circuit — subnetwork of SAE features."""
    edges: List[CircuitEdge] = field(default_factory=list)
    input_features: List[int] = field(default_factory=list)
    output_features: List[int] = field(default_factory=list)
    behavior_description: str = ""

    @property
    def graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for e in self.edges:
            G.add_edge(
                f"L{e.source_layer}F{e.source_feature}",
                f"L{e.target_layer}F{e.target_feature}",
                weight=e.attribution_score,
            )
        return G


class ActivationPatching:
    """Activation patching utilities for circuit discovery."""

    def __init__(self, model, sae_dict: Dict[int, "SAE"]):
        """
        Args:
            model: TransformerLens HookedTransformer
            sae_dict: {layer: SAE} — SAE for each layer
        """
        self.model = model
        self.sae_dict = sae_dict

    @torch.no_grad()
    def compute_feature_attribution(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        target_layer: int,
        target_logit_idx: int,
    ) -> Dict[Tuple[int, int], float]:
        """Compute attribution scores for all features at target_layer.

        Uses activation patching: replace clean activation with corrupted,
        measure change in target logit.

        Returns:
            {(layer, feature_idx): attribution_score}
        """
        clean_tokens = self.model.to_tokens(clean_prompt)
        corrupted_tokens = self.model.to_tokens(corrupted_prompt)

        # Run model with cache
        _, clean_cache = self.model.run_with_cache(clean_tokens)
        _, corrupted_cache = self.model.run_with_cache(corrupted_tokens)

        # Get clean and corrupted activations at target layer
        hook_name = f"blocks.{target_layer}.hook_resid_pre"
        clean_act = clean_cache[hook_name]  # [batch, pos, d_model]
        corrupted_act = corrupted_cache[hook_name]

        sae = self.sae_dict[target_layer]

        # Encode to SAE features
        clean_features = sae.encode(clean_act)  # [batch, pos, d_sae]
        corrupted_features = sae.encode(corrupted_act)

        # Measure logit difference for each feature
        # Patch one feature at a time: clean_feature → corrupted_feature
        def patch_hook(activation, hook, feature_idx: int, clean_val, corrupted_val):
            # Replace specific feature with corrupted version
            features = sae.encode(activation)
            features[:, :, feature_idx] = corrupted_val[:, :, feature_idx]
            return sae.decode(features)

        attributions = {}

        for feature_idx in range(sae.d_sae):
            # Create hook that patches this feature
            def make_hook(f_idx, c_val, co_val):
                def hook(act, hook):
                    return patch_hook(act, hook, f_idx, c_val, co_val)
                return hook

            # Run with patched feature
            with self.model.hooks(
                fwd_hooks=[(hook_name, make_hook(
                    feature_idx, clean_features, corrupted_features
                ))]
            ):
                patched_logits = self.model(clean_tokens)
                patched_logit = patched_logits[0, -1, target_logit_idx]

                clean_logit = self.model(clean_tokens)[0, -1, target_logit_idx].item()
                attribution = (clean_logit - patched_logit.item()) ** 2

            if attribution > 1e-6:
                attributions[(target_layer, feature_idx)] = attribution

        return attributions

    def discover_circuit(
        self,
        behavior_prompts: List[Tuple[str, str]],
        target_token_idx: int,
        threshold: float = 0.01,
        max_layers: int = 4,
    ) -> SparseCircuit:
        """Discover sparse feature circuit for a behavior.

        Uses layer-by-layer backward attribution:
        1. Start from logit output
        2. Find SAE features at final layer that affect the logit
        3. Trace back through earlier layers
        4. Build the causal graph

        Args:
            behavior_prompts: [(clean, corrupted)] pairs
            target_token_idx: token index of interest
            threshold: minimum attribution to include feature

        Returns:
            SparseCircuit object
        """
        edges = []

        for layer in range(self.model.cfg.n_layers - 1, -1, -1):
            if layer not in self.sae_dict:
                continue

            for clean_prompt, corrupted_prompt in behavior_prompts:
                attributions = self.compute_feature_attribution(
                    clean_prompt=clean_prompt,
                    corrupted_prompt=corrupted_prompt,
                    target_layer=layer,
                    target_logit_idx=target_token_idx,
                )

                for (l, f_idx), score in attributions.items():
                    if score > threshold:
                        edges.append(CircuitEdge(
                            source_layer=l,
                            source_feature=f_idx,
                            target_layer=l + 1,
                            target_feature=0,  # simplified
                            attribution_score=score,
                        ))

        # Build circuit from discovered edges
        circuit = SparseCircuit(edges=edges)
        circuit.behavior_description = (
            f"Circuit for token {target_token_idx} "
            f"({len(edges)} edges, {len(behavior_prompts)} prompts)"
        )
        return circuit


class CircuitIntervention:
    """Apply circuit-level interventions to modify model behavior."""

    def __init__(self, model, sae_dict, circuit: SparseCircuit):
        self.model = model
        self.sae_dict = sae_dict
        self.circuit = circuit

    def ablate_feature(self, feature_idx: int, layer: int, value: float = 0.0):
        """Zero-ablate a specific feature in the circuit."""
        sae = self.sae_dict[layer]
        hook_name = f"blocks.{layer}.hook_resid_pre"

        def ablation_hook(activation, hook):
            features = sae.encode(activation)
            features[:, :, feature_idx] = value
            return sae.decode(features)

        return hook_name, ablation_hook

    def steer_via_circuit(
        self,
        prompt: str,
        steer_direction: Dict[Tuple[int, int], float],
        max_new_tokens: int = 50,
    ) -> str:
        """Steer model generation by modifying circuit features."""
        tokens = self.model.to_tokens(prompt)

        hooks = []
        for (layer, feature_idx), strength in steer_direction.items():
            sae = self.sae_dict[layer]
            hook_name = f"blocks.{layer}.hook_resid_pre"

            def make_hook(f_idx, str_val):
                def hook(activation, _):
                    features = sae.encode(activation)
                    # Add feature direction scaled by strength
                    features[:, :, f_idx] += str_val * activation.norm(dim=-1, keepdim=True)
                    return sae.decode(features)
                return hook

            hooks.append((hook_name, make_hook(feature_idx, strength)))

        generated = self.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            fwd_hooks=hooks,
        )
        return self.model.to_string(generated[0])
