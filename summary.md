# Evaluation summary

Weeping scenarios detected: 27 of 27 (recall 100%), episode precision 89%, median onset delay 0.08 h.
oof: test AP 0.9993, AUC 1.0, F1 0.9864 at threshold 0.95
forward: test AP 0.9984, AUC 0.9999, F1 0.9648 at threshold 0.95
loco: test AP 0.8636, AUC 0.9948, F1 0.5108 at threshold 0.95
physics_rule: test AP 0.9828, AUC 0.9987, F1 0.8955 at threshold 0.15
isolation_forest: test AP 0.694, AUC 0.9863, F1 0.7587 at threshold 0.55
Root cause top1 100% (signals only 96%, records only 96%)
Impact: total estimated GBP 4,486,953 vs true GBP 5,503,610 (-18.5 percent)
Retrieval bm25: capped recall@10 0.042, MRR 0.134, nDCG@10 0.048
Retrieval dense: capped recall@10 0.087, MRR 0.204, nDCG@10 0.093
Retrieval hybrid: capped recall@10 0.063, MRR 0.162, nDCG@10 0.068
Retrieval hybrid_graph: capped recall@10 0.178, MRR 0.603, nDCG@10 0.255
Retrieval full: capped recall@10 0.834, MRR 0.969, nDCG@10 0.851
Recurrence pairwise F1 1.0
Verifier coverage 100.0%; injected error detection number_changed 98%, fake_evidence_id 100%, out_of_window_evidence 100%, uncited_claim 100%
Entity linking correct column 100.0%
