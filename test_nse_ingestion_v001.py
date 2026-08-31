#!/usr/bin/env python3
import json, tempfile, unittest
from pathlib import Path
import ingest_nse_formula_dataset_v001 as ingest

class NSEIngestionTests(unittest.TestCase):
    def test_dangling_latex_relation_with_trailing_slash_is_flagged(self):
        self.assertIn('dangling_relation', ingest.quality_flags(r'|T_j| \\;\\le\\'))
        self.assertIn('dangling_relation', ingest.quality_flags(r'T_j \\;=\\'))
        self.assertEqual(ingest.quality_flags(r'|T_j| \\le C'), [])

    def test_source_weight_formula(self):
        records=[{'record_id':'NSF00000','split':'train'},{'record_id':'NSF00001','split':'train'}]
        rec={x['record_id']:x for x in records}
        edge={'edge_id':'NSE000000','source_id':'NSF00000','target_id':'NSF00001','shared_source_count':1,'shared_families':['a','b'],'weight':2.0}
        row=next(ingest.correlation_records([edge],rec))
        self.assertEqual(row['heuristic']['reconstructed_weight'],2.0)
        self.assertTrue(row['heuristic']['weight_formula_verified'])
        self.assertFalse(row['canonical_dependency'])

    def test_provenance_components(self):
        records=[
            {'record_id':'A','sources':[{'sha256':'s1'}]},
            {'record_id':'B','sources':[{'sha256':'s1'}]},
            {'record_id':'C','sources':[{'sha256':'s2'}]},
        ]
        self.assertEqual([len(x) for x in ingest.provenance_components(records)],[2,1])

    def test_relation_negatives_do_not_use_existing_edges(self):
        records=[
            {'record_id':'A','split':'train','formula_normalized':'A'},
            {'record_id':'B','split':'train','formula_normalized':'B'},
            {'record_id':'C','split':'train','formula_normalized':'C'},
            {'record_id':'D','split':'train','formula_normalized':'D'},
        ]
        edges=[{'edge_id':'E1','source_id':'A','target_id':'B','weight':1.5}]
        rows=list(ingest.relation_candidates(records,edges,1.5))
        self.assertEqual(len(rows),2)
        pos=next(r for r in rows if r['target']['structured']['heuristic_correlated'])
        neg=next(r for r in rows if not r['target']['structured']['heuristic_correlated'])
        self.assertEqual(set(pos['source_record_ids']),{ingest.research_id('A'),ingest.research_id('B')})
        self.assertNotEqual(set(neg['source_record_ids']),set(pos['source_record_ids']))
        self.assertEqual(neg['split'],'train')

if __name__=='__main__': unittest.main()
