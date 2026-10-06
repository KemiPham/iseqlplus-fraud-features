# Pre-registered protocol: composite grouping key (written before any uid statistic or model was computed)

Motivation (from the revision-4 paper): the reversal of direction grows with card1 group size, so card1 pools cardholders.

Key (single variant, no alternatives will be tried):
  uid = (card1, addr1, start) with start = day - D1, where day = elapsed 24-h bucket and D1 the raw
  Vesta timedelta (no imputation). A missing addr1 or D1 is its own level of that component.
  Provenance: engineering choice (E), following public analyses of the IEEE-CIS competition.

Statistics: the same eight specifications, conventions and defaults as revision 4, with g = uid instead of card1.
  P2 uses the frozen revision-4 device map; P4 uses the strict same-bucket earlier count.

Models: D_uid = model C's 436 standard features + the eight uid statistics (same column names). LightGBM config
  identical to revision 4 (31 leaves, lr 0.05, is_unbalance, seed 42, <=500 rounds, early stopping on validation AUC, patience 50).
  Deletion probe: four refits on data4 drop manifest (same rows as revision 4).

Primary comparison: D_uid vs C, AUC, test month, paired DeLong + card1-group bootstrap (1,000; seed 42).
Secondary (reported regardless): PR-AUC; D_uid vs D (card1 statistics); signs across the 5 probe fits.
Claim rule: a gain is claimed only if test dAUC > 0, its card1-group 95% CI excludes 0, AND D_uid > C in all 5 probe
  fits in both periods. Otherwise the result is reported as a null/sensitivity result. The test month is scored once.
Checks: reference evaluator on uid groups (all groups with tied timestamps + 300 random groups, seed 7).
