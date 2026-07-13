test-block-A:
	pytest tests/unit/test_cost_model.py tests/unit/test_ev_metrics.py tests/unit/test_forensic_checks.py tests/unit/test_migration_idempotency.py tests/unit/test_verdict_gate.py tests/integration/test_verdict_end_to_end.py -q
