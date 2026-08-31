from __future__ import annotations
import tempfile, unittest
import torch
import ingest_torch_math_graph_v004 as ingest
import safe_torch_math_graph_loader as safe

class TorchGraphIngestionTests(unittest.TestCase):
    def test_content_blob_detection(self):
        self.assertTrue(ingest.content_blob_like('"content": "#!/usr/bin/env python3\\ndef x(): pass"'))
        self.assertFalse(ingest.content_blob_like(r"\|u\|_{L^2}^2"))

    def test_typed_edge_aggregation_preserves_multiplicity(self):
        data={
          "node_ids":["a","b"],
          "edge_types":["rel"],
          "edge_index":torch.tensor([[0,0],[1,1]],dtype=torch.long),
          "edge_type_index":torch.tensor([0,0],dtype=torch.long),
        }
        rows=ingest.aggregate_typed_edges(data)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["multiplicity"],2)
        self.assertFalse(rows[0]["canonical_relation"])

    def test_safe_tensor_loader_uses_weights_only_compatible_data(self):
        data={"schema":"x","node_ids":["n"],"node_types":["formula"],"node_type_to_idx":{"formula":0},
              "edge_types":[],"edge_type_to_idx":{},"operator_ids":["op"],"operator_to_idx":{"op":0},
              "edge_index":torch.empty((2,0),dtype=torch.long),"edge_type_index":torch.empty((0,),dtype=torch.long),
              "node_type_index":torch.tensor([0]),"formula_like_mask":torch.tensor([True]),
              "executable_mask":torch.tensor([True]),"exact_operator_mask":torch.tensor([False]),
              "operator_multihot":torch.tensor([[True]]),"basic_features":torch.zeros((1,6))}
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(data,f.name)
            loaded=safe.load_torch_math_graph(f.name)
            inputs,targets=safe.split_inputs_targets(loaded)
        self.assertNotIn("operator_multihot",inputs)
        self.assertNotIn("executable_mask",inputs)
        self.assertIn("operator_multihot",targets)
        self.assertIn("executable_mask",targets)

if __name__=="__main__": unittest.main()
