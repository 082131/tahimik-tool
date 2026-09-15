# Tasks: MrT5-Backed TAHIMIK

- [X] T001 [US1] Configure TAHIMIK to inherit `stanfordnlp/mrt5-small`.
- [X] T002 [US1] Load Stanford custom model code and pretrained gate in `src/models/noise_adaptive.py`.
- [X] T003 [US1] Disable embedded gate execution and fail closed on transfer failure.
- [X] T004 [US2] Preserve the 256-unit estimator and pre-gate masked pooling.
- [X] T005 [US2] Apply `d_max * (1 - n)` and `cn * (n - navg)`.
- [X] T006 [US3] Preserve detach boundaries so `L_NE` exclusively trains the estimator.
- [X] T007 Verify common MrT5 initialization, adaptive direction, diagnostics, loss, and gradient tests.
- [X] T008 Verify the complete Python suite.

## Open convergence tasks

- [ ] T009 Apply the same decoder-output scaling parity fix and tests required
  by the compressed MrT5 path.
- [ ] T010 Persist soft and hard deletion diagnostics by noise band and verify
  hard-rate behavior against adaptive targets.
- [ ] T011 Enforce shared checkpoint source/stage validation in standalone
  evaluation and benchmarking.
- [ ] T012 Narrow and log expected pretrained-gate loading failures while
  preserving fail-closed construction.
