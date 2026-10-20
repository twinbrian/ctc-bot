# Critical evaluation of `ctc-bot 2.zip` — v1

## Bottom line

**This is a credible weather-informed research prototype, but its claimed prospective PnL is not established, and its live order/control layer has defects I would fix before relying on it.** More sophisticated features do not resolve the historical-price-versus-executable-market gap that undermined our old strategy.

The new upload's trading code, research code and all four saved model files are **byte-for-byte identical** to the previous upload. All 46 compared code/documentation/parameter/model/cache files match; the archives themselves have different hashes. This review verified the newly extracted source and reran the small synthetic reproductions against it. It did not assume fixes or changes from the new filename.

Our trading and model-research work remain stopped. No live API calls, model fitting or historical PnL trials were run for this review. Supplied pickles were hashed, not executed. All findings concern the supplied version; they do not establish the behavior of another deployed revision. Synthetic examples demonstrate behavior, not realized trading losses.

## Highest-priority findings

### 1. HIGH — A symbol-level pause does not reliably stop acquisition

**Confirmed by reproduction.** The manual pause handler changes `strat.paused`, but does not remove the symbol's existing broker goal. The next model decision skips the paused stock and submits no replacement goal. The broker retains the old target and continues working toward it.

Test: start with an unfilled target of +10 PORT, pause PORT, run another decision, then advance the fake broker. **Ten shares are acquired after the pause.** Rejecting an implausible prediction also leaves its previous acquisition goal alive.

Evidence: `bot/live.py:197–198`, `bot/strategy.py:464–468`, `bot/broker.py:53–57,245–289`; `CONTROL-CHECKS-v2.json`.

The separate whole-bot halt does submit hold goals. This is a symbol-pause/stale-goal defect, not a claim that every halt mechanism fails. Repair requires explicit goal invalidation plus reconciliation/cancellation of outstanding orders, not just skipping subsequent model decisions.

### 2. HIGH — It can buy after its own forecasted upside is already gone

**Confirmed by reproduction.** Features/predictions are anchored to a stored midpoint, but execution normally occurs at a later observation. Although it computes the correct endpoint, it applies the old price change to the newer midpoint when setting direction and price cap.

| Synthetic input/output | Value |
|---|---:|
| Feature-time midpoint | USD 50.00 |
| Predicted change | USD +4.00 |
| Correct anchored endpoint | USD 54.00 |
| New midpoint | USD 55.00 |
| Bot's buy cap | **USD 57.25** |
| Fake-account acquisition | **200 PORT** |

The intended endpoint is below the current market, yet it acquires a full long position. A higher-quality weather model cannot repair this execution formula.

Evidence: `bot/strategy.py:158–167,453–484`. Remaining edge must be recomputed from the fixed endpoint and current executable prices at order time.

### 3. HIGH — Failed order checks/cancels can create unintended duplicate exposure

**Confirmed with two fake-transport reproductions.** A failed open-order read is treated as an empty result. A failed cancel response is ignored. The bot then forgets the original working order and can submit again.

For a desired total of ten shares, both failure cases leave **20 resting buy shares** in the fake exchange while tracking only the new order. Both orders fit within the exchange's PORT limit, so exchange limit enforcement does not fix the intended-target error.

Evidence: `bot/broker.py:254–289`; `EXECUTION-CHECKS-v1.json`. Unknown order outcomes must remain reserved until reconciled; retrying is not proof the original order failed.

### 4. HIGH — “200 unseen seasons” overstates the validation

There are two distinct sources of optimism:

- **Supervised preprocessing includes evaluation outcomes.** `analysis/build_all.py:29–57` fits the damage model on all seasons before constructing features. Future-storm priors also use the full library (`analysis/nowcast.py:65–80`; `bot/train.py:60–64`). Excluding seasons only from the final HGB fit does not exclude them from these fitted inputs.
- **Evaluation blocks were used to select settings.** Thresholds, flipping, start age, holding policy and fraction of predicted movement paid were compared on those blocks. See `bot/cost_variants.py:13–39` and `bot/README.md:26–71`. After this selection, the reported mean is a development result, not an untouched final test.

I found no direct future-price/season-result lookup in the trading inference path. The amount of optimism from preprocessing and selection is unmeasured; this is not proof that all of its profits are fictitious. But it prevents treating the selected mean as a sound competition forecast.

### 5. HIGH — Historical house-price prediction is not the live trade objective

The regression target is **historical house price at second 542 minus house price now** (`bot/train.py:35–57`). Live features instead use actual top-of-book midpoints, and orders trade the live book (`strategy.py:108–122,453–484`).

Other teams, house inventory skew, asynchronous quotes and receipt delays can change the live price path. A physical-minus-price gap may therefore be a genuine convergence opportunity or a live-price distortion outside the training distribution. The replay always centers liquidity on historical house prices, so it cannot distinguish them. Increasing a spread assumption does not reproduce a different market path.

There is also a holding-horizon mismatch: the same regression target is used for round 3, but those positions are held to **final settlement**, not sold at second 542 (`strategy.py:170–176`). A predictor of one endpoint is not automatically optimal for the other.

**This is the most important inherited weakness from our old model.** Better storm physics can help estimate fundamentals, but does not by itself demonstrate positive executable net edge. Our old historical PnL is not independent evidence in favor of this bot.

### 6. HIGH — The replay understates important execution constraints

The finite-depth mode is real, so “all orders fill without limit” is too broad. The actual problems are:

- Default no-profile mode fills arbitrary size at the touch.
- Finite-profile mode assumes a median ladder replenished around historical consensus. It lacks explicit submission/arrival latency, pending-order/cancel accounting, failures and a common action limiter.
- It calls `broker.tick()` but not `strategy.tick()`, omitting timed live behavior such as leftover-exit retries.
- The uploaded archive contains no original book logs, so its calibrated ladder and live costs cannot be independently reconstructed from this package.

A synthetic one-share-per-refresh ladder across nine stocks generates **135 filled order-equivalents in 56 seconds**, including nine at goal submission. This proves the replay does not enforce a single key's 60-actions/minute budget. It does not quantify the PnL impact or how often the production strategy reaches that demand. Multiple-key comparisons need identical capacity assumptions.

Evidence: `bot/broker.py:68–136`; `bot/replay.py:42–74`; `bot/book_profile.py:30–55`; `bot/live.py:211–218`.

### 7. HIGH — Announcement behavior is unvalidated and can reinstate obsolete bets

The announcement follower is enabled by default, despite the README acknowledging **zero real announcement examples** in the research/testnet sample (`README.md:330–338`; `strategy.py:74`).

It reacts to six seconds of midpoint movement, targets half the limit, and subsequently returns to the **pre-announcement position**. In a reproduced GEN example:

- Initial position: long 600.
- Follower target: short 300, requiring a 900-share sale.
- Timed exit target: long 600 again, requiring a 900-share purchase if entry filled.

These are legal net positions. The concern is large two-way turnover through a moving book and restoration of an old-economics position. Additionally, setting `follow_on=False` after a watcher is queued does not stop that watcher: the reproduction still creates the short target (`strategy.py:317–338`).

The ordinary announcement pause propagates to ETFs, but not all related companies. A GRID sensitivity change can affect GEN, RECON and insurer economics even if only GRID is named. Those remaining models can trade phantom gaps using obsolete parameters. Waiting for the house price to adjust does not repair a changed payoff function.

## Further material issues

| Finding | Concrete consequence and evidence |
|---|---|
| Feed liveness is not data freshness | Any WebSocket frame refreshes `last_frame`; there is no per-symbol book-age gate. Failed position reads log and return without stopping acquisition. A heartbeat can coexist with stale executable/account state. `live.py:237–238`; `broker.py:245–272,376–379`. |
| Rate limiter is not a strict rolling limiter | The 55/minute bucket starts with ten burst tokens. Reproduction accepts **64 actions in 60 seconds**; three independent keys allow 192 without a shared 180/team limiter. Actual server rejection behavior was not queried. `broker.py:139–152,193,230–234`. |
| Snapshot recovery misses the final trading trigger | Reproduced: snapshot restores round-3 final damage, then returns without calling `final_snipe`; action count remains zero. Later snapshots see the report as already applied. `strategy.py:412–420`. |
| Final valuation is not unconditionally exact | Reported insured-loss totals are logged but not passed into valuation; losses are rebuilt from old fixed asset coefficients. REIN explicitly has an unseen upper layer, yet is extrapolated without a cap. Outages rely on reconstructed tracks. Matching supplied no-announcement seasons does not prove exact values under changed economics or unseen loss ranges. `engine.py:84–120`; `value.py:14–16,36–59`. |
| Pattern probabilities ignore regime changes | Six direct advisories followed by an erratic advisory still give over **99.9998% direct** in the constructed posterior. No switching hazard or forgetting exists. Other trajectory/cone features can compensate, so this does not prove the whole model is blind. `analysis/features.py:11–24`. |
| Forecast physics has approximations | Late-age intensity is double-decayed; projected exposure omits the observed model's 2r truncation; nonlinear insurer/rebuilder values are evaluated at mean damage rather than averaged across scenarios. Useful learned features, but not exact expected payoffs. `nowcast.py:49–60`; `engine.py:68,106–113`. |
| Team attribution needs raw evidence | Fill polling accumulates returned trades without an owned-order filter; positions are account positions. If other bots share the team, this is not isolated bot PnL. It does not prove the README included other bots. `broker.py:355–371`. |

## What the performance documentation actually supports

| Supplied claim | Correct reading |
|---|---|
| USD 9,014 per season | Selected historical simulation mean, not a demonstrated actual/testnet average |
| USD 76,300 total, live seasons 71–86 | Approximately **USD 4,769 per season**, 16 seasons, multiple settings versions |
| USD 31,200 total, live seasons 80–86 | Approximately **USD 4,457 per season**, seven seasons on newer settings |
| USD 29,663 in season 70 | A single highlighted season on older settings |

The live aggregates overlap and cannot be pooled. They are arithmetic from rounded README claims, not reconciled exchange records: **this archive also has no `bot/logs/` files**. The README explicitly says `models.pkl` was trained on all 600 seasons, including the testnet weather. Its testnet performance is therefore not unseen-weather validation.

Our small actual sample and the teammate's documented samples cover different storms and settings. Neither establishes which bot is better. Calling a loss “variance” because a replay also loses is not independent proof of calibrated risk; both can share the same misspecified model.

## What is genuinely good

The storm-age clock, official cone/probability features, cross-stock physical-minus-price residuals, persistent damage state and capped acquisitions are sensible ideas. The source shows substantial logging and evaluation effort and documents rejected variants. I found no create/redeem call in this trading bot: its main in-storm approach is not the old ETF conversion route.

Full position limits are not inherently wrong when the objective is expected PnL. Generic tighter sizing or stop-losses can lower expected profit. The priority is to make the intended forecast, current executable edge, account state and order behavior agree, and then establish the strategy's incremental net value with credible evidence.

## Recommendation

**Fix the control/execution defects before more threshold tuning or trusting the headline mean.** The highest priorities are reliable pause/goal invalidation, conservative reconciliation of unknown orders, anchored remaining edge, fresh per-symbol/account state, shared action limits and a coherent announcement policy. Any later evaluation should freeze settings, fit every learned component within training partitions, and distinguish actual owned-fill PnL from hypothetical house-price fills.

This review applies no fixes and initiates no trial. To substantiate the teammate's latest actual average, the missing evidence is raw decision/order/fill/book logs plus exact settings/model version—not another API key. If a materially revised bot was intended, the new upload does not contain changed trading code or models.

## Reproduction and provenance

- `INVENTORY-v1.json`: new and previous archive hashes, 46 file comparisons, no live logs, exact original source hashes.
- `CONTROL-CHECKS-v2.json`: fresh results against the **new** archive for stale anchoring, ineffective symbol pause, retained rejected goals, follower behavior, snapshot recovery and static pattern posterior.
- `EXECUTION-CHECKS-v1.json`: fresh new-archive broker reproductions for unknown order outcomes and rate/replay capacity.
- `check_controls_v2.py` / `check_execution.py`: reproducible offline micro-tests. Place the teammate bot root under `source/` beside them; run each with Python, NumPy/pandas installed. They do not start live workers, fit models, replay seasons or call the exchange.

All source paths/line numbers refer to this uploaded archive's unchanged files. Source and saved model bytes were not modified.
