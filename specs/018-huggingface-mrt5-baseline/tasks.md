# Tasks: Hugging Face MrT5 Baseline

- [X] T001 [US1] Configure MrT5 to resolve to `stanfordnlp/mrt5-small`.
- [X] T002 [US1] Load the custom model through `AutoModelForSeq2SeqLM` with `trust_remote_code=True`.
- [X] T003 [US2] Capture and disable the embedded Stanford gate in `src/models/delete_gate.py`.
- [X] T004 [US2] Map Stanford gate weights into the local `DeleteGate`.
- [X] T005 [US2] Align RMS normalization, scaled-sigmoid direction, Gumbel training noise, and vectorized hard deletion.
- [X] T006 [US3] Raise `RuntimeError` when pretrained gate loading fails.
- [X] T007 Add exact loader, gate-transfer, and one-gate tests in `tests/test_mrt5_model_loading.py`.
- [X] T008 Verify focused model/gate tests and the complete Python suite.

## Open convergence tasks

- [ ] T009 Correct conditional decoder-output scaling before each manual
  `lm_head` call and add native-parity regression tests.
- [ ] T010 Persist and validate both soft training rate and hard inference rate;
  do not claim target-rate attainment without hard-rate evidence.
- [ ] T011 Apply shared checkpoint source/stage validation in standalone
  evaluation and benchmarking.
- [ ] T012 Narrow and log expected pretrained-gate loading failures while
  preserving fail-closed construction.
