# Judge showcase pack — Datathon 2026, Team Super intelligence

Prepared 2026-10-16 (during judging). Grounded in `final_solution.py` in this folder.

## The walkthrough script (for the demo slot)

"Hi — we're team Super intelligence. The problem: predict battery health — how many
cycles until a battery degrades to 80% health, and its current health percentage.

Our approach starts with physics, not ML. The key insight: if a battery is at 90%
health and degrading 0.1% per cycle, it has roughly (90 − 80) / 0.1 = 100 cycles
left. That simple ratio became our strongest feature.

We engineered 7 features around this idea — charge efficiency, thermal swings,
total stress across six scores, depth of discharge, heat-plus-fast-charging
interaction, and an age-times-wear interaction.

For validation we held out 30 entire batteries — not random rows, because readings
from the same battery look alike and random rows would leak.

We trained LightGBM and XGBoost with MAE loss for remaining life, ensembled them,
and LightGBM for health. Then retrained everything on all data and wrote the exact
Kaggle submission format. We scored 0.0097 on the public board."

## Questions judges are likely to ask (and the honest answers)

- **Why hold out whole batteries?** Readings from the same battery resemble each
  other; random rows would let the model memorize batteries instead of learning
  degradation. Whole-battery holdout keeps validation honest.
- **Why an ensemble?** LightGBM and XGBoost make different mistakes on different
  rows — averaging them cut the variance of the remaining-life prediction.
- **Why predict SOH first?** The test set has no SOH column, but the physics ratio
  needs it — so SOH is predicted first, then fed back as the `rul_ratio` feature
  for the remaining-life model. (SOH's own model excludes `rul_ratio` — using it
  there would be circular.)
- **Libraries?** pandas, numpy, LightGBM, XGBoost, scikit-learn (only
  `GroupShuffleSplit`), os.
- **How did you avoid overfitting?** Battery-grouped holdout, early stopping
  (50 rounds) on the held-out batteries, and the public-board score ≈ validation
  score as a final sanity check.
- **What would you do next?** Tune the ensemble blend weights, add battery-age
  trajectory features, and try the CatBoost experiments from earlier today.
- **If asked about Team Diamond's 0.0039:** "They found a stronger approach — we
  don't know their method. Our validation was honest: battery-grouped holdout,
  no peeking at test labels."
- **If asked your rank:** say "0.0097 on the public board (30% of the test data);
  final standings are computed on the private 70%." Do not claim a rank or a
  first place — the deck's old "1st place" slide must stay out (flagged earlier
  today).

## Fair-play note

Everything used is standard libraries and docs, per the rulebook. No test labels
were ever available — the test set was features-only. Own the story: the physics
ratio was Vinit's line of thinking, and the grouped validation is the part judges
respect most.
