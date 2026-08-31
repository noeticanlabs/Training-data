#!/usr/bin/env python3
from __future__ import annotations
import torch

TARGET_KEYS={"executable_mask","exact_operator_mask","operator_multihot"}

def load_torch_math_graph(path, *, map_location="cpu"):
    """Load the source tensor dataset without arbitrary pickle execution."""
    return torch.load(path,weights_only=True,map_location=map_location)

def split_inputs_targets(data):
    """Separate model/GI inputs from execution/operator supervision targets."""
    inputs={
      "schema":data.get("schema"),
      "node_ids":data["node_ids"],
      "node_types":data["node_types"],
      "node_type_to_idx":data["node_type_to_idx"],
      "edge_types":data["edge_types"],
      "edge_type_to_idx":data["edge_type_to_idx"],
      "edge_index":data["edge_index"],
      "edge_type_index":data["edge_type_index"],
      "node_type_index":data["node_type_index"],
      "basic_features":data["basic_features"],
      "formula_like_mask":data["formula_like_mask"],
    }
    targets={k:data[k] for k in sorted(TARGET_KEYS)}
    return inputs,targets
