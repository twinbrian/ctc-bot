# Hurricane bot

Trades the edges in `analysis/FINDINGS.md` on the Cornell Trading Competition exchange
(`https://cornelltradingcompetition.org`, the same site as the web terminal):

- **In-storm model.** At storm ages 10, 30, …, 470 s it predicts how far the house price will move by
  the end of the storm. The bot works toward the full position limit in that direction, paying at
  most mid + half the predicted move per share. It flattens when the storm dissipates.
- **Round-3 final report.** The final damage report fixes every settlement exactly. The bot takes
  every resting order on the wrong side of value ± a safety margin, then holds to settlement.

Everything is keyed on storm age, never the clock, so it works with either round timing.

## How it trades, round by round

The same in-storm model runs in every round. Round is one of its inputs, and its fair values include
the storms still to come.

| | Rounds 1–2 | Round 3 |
|---|---|---|
| During the storm (ages 10–470 s) | Full position limit toward the predicted house move, paying ≤ half the predicted move | same |
| Storm ends | Close at the model's fair value (± half-spread + 0.15%); never chase a runaway book | **Hold**: no later storm to carry risk into |
| Damage reports | Re-close leftovers within 0.3% of mid | Final report: exact settlement values; take mispriced quotes, hold everything to settlement |

The policy was chosen on 100 held-out seasons with storm-time book depth (`bot/compare_variants.py`):

| Policy | Mean per season | Profitable seasons |
|---|---|---|
| Hold in round 3, flatten in rounds 1–2 (**used**) | $5,729 | 72% |
| Flatten every round | $5,440 | 71% |
| Hold every round | $3,013 | 63% |
| Add the formation front-run (built, tested) | −$280 vs the same policy without it | |

**Entry threshold (changed after 8 live seasons).** The live bot paid about $59k in execution costs
(0.4–1.4% per share against fair value) over 8 testnet seasons, because it traded every prediction
whose half cleared 0.1% of price. Re-tested out-of-sample (seasons 501–600) with costs calibrated
to the live fills:

| Setting | Mean per season | Profitable seasons |
|---|---|---|
| 0.1% threshold + flips (old) | $5,537 | 72% |
| 1% threshold | $6,930 | 78% |
| **1% threshold, no flips (used)** | **$7,085** | **79%** |
| 2% threshold | $6,216 | 83% |

No symbol loses on average out-of-sample. But any single symbol in any single round loses 25–45% of
the time (PORT and HAVEN swing the most), so a few thousand dollars of swing per round is normal.

**First entry at storm age 30, not 10 (changed after season 79).** At age 10 the model has no skill
(out-of-sample R² ≈ 0), and the house still makes a big repricing 16–28 s after the storm forms. In
season 79 that cost $7.3k in one round. Waiting for the first official advisory:

| Held-out seasons | Start at age 10 | Start at age 30 |
|---|---|---|
| 501–600 | $7,122/season, 79% profitable | $8,264/season, 86% profitable |
| 401–500 (separately trained model) | $7,656/season, 82% profitable | $8,603/season, 89% profitable |

**Validation of every remaining setting** (two independent held-out sets, live-calibrated costs):

| Setting | Result |
|---|---|
| Exit margin (0.05–0.5%) | No effect |
| Last decision age (230–470) | Little effect |
| Share of the predicted move paid to enter | **Matters**: 0.3 → $7.0k, 0.5 → $8.3–8.6k, **0.7 → $8.8–9.2k (used)**, plateau to 0.85, falls at 1.0 |

**Final configuration, 200 unseen seasons:**
- $9,014 per season, profitable 88% of the time
- 24–34% of rounds lose; worst season −$14.7k, best +$32.6k
- Every symbol is positive in every round, with nearly the same averages in both sets

The front-run loses money and is **off**. The jump is real, but crossing the thin storm-time book
twice costs more than the jump pays. REIN is not traded in-storm, because its edge is below its cost.

## Run it

```bash
export HURRICANE_BOT_KEY=...                 # Team page -> API keys -> create a "bot" key
export HURRICANE_EXTRA_KEYS=key2,key3        # optional: more bot keys = more order throughput (60/min each, 180/team)
PY=/opt/anaconda3/bin/python3
$PY -I bot/live.py --smoke-test              # 1-share order far from market, then cancel (checks key + API)
$PY -I bot/live.py --dry-run                 # real feed and real book, paper fills, sends nothing
$PY -I bot/live.py                           # live trading; leave it running, it follows season after season
```

Stop it with Ctrl-C. **Start it before round 1 and leave it running.** It records every feed
message and order-book update to `bot/logs/`. On a restart it first replays its own recording,
rebuilding storms, outage history, season damage and pre-storm prices exactly, then resumes trading.

If it was *not* running for an earlier storm of the season, the outage history is unknown. It then:

- marks the season approximate
- skips in-storm trading
- limits the final-report trade to the outage-free symbols (ISLE, AQUA, HAVEN, REIN, RECON, MAT)

## Manual controls (`bot/control.json`)

Write one JSON command to `bot/control.json`. The bot reads it within 2 s and deletes it, and its state is kept.

| Command | Effect |
|---|---|
| `{"halt": true}` | **The brake.** No automatic trading at all, and positions are held as they are. This includes the after-storm exit, the leftover retry and the announcement follower. |
| `{"halt": false}` | Release the brake (this also ends a flatten). |
| `{"flatten": true}` | Close every position within 1% of mid, retrying from the current mid every 20 s until flat, then stay halted. Any entry still being worked is stopped. |
| `{"close": ["ARCH"]}` | Close these positions within 1% of mid. |
| `{"pause": ["GEN"]}` / `{"resume": ["GEN"]}` | Stop or restart trading these symbols. Announcements pause their symbols automatically. |

The brake and pauses are not saved: **a restart releases them.** Nothing in the bot turns the brake on by itself;
only these commands do. The change was checked on 10 replayed seasons with fixed inputs, where results were
identical to the cent, and with fake-exchange tests 9a–9e.

## Round monitor

```bash
OMP_NUM_THREADS=1 nohup /opt/anaconda3/bin/python3 -I -u bot/monitor.py > bot/logs/monitor.log 2>&1 &
```

Every ALERT goes to `bot/logs/monitor.log`, `bot/logs/alerts.log` (live alerts with their diagnosis also
go to `bot/logs/live_alerts.md`) and a macOS notification. Logs are read incrementally, so CPU use is about 0%.

**During a round (`bot/watch.py`, checked every 5 s).** The round's PnL comes from one consistent source:
- the bot's fills, ordered by the exchange's execution time
- valued at the recorded book mid
- reconciled with the exchange's positions
- evaluated 5 s behind real time, so every fill has been logged (logging lag: median 1.1 s, max 2.5 s)

The bot's own status PnL is not used. It mixes positions (polled every 1 s) with fill cash (polled every 2 s), so
during fast entries it showed phantom swings of up to −$15k. Those caused false alerts and used up the alert levels
that the real Lena drop needed three minutes later.

Validated on 2026-10-18:
- On the 99.9% of status lines with no fills in flight, it matches the bot's exchange ledger (median difference $0).
- It matches the exchange dashboard to $66 (−$6,182.71 vs −$6,116.82, with the same positions).

**A drop is an alert when it is material** (about one standard deviation of a round's PnL):
- the round is down $5k, or
- the round is $8k below its best so far, or
- the round lost $6k within 60 s

There is one alert per incident. The next alert comes only if the incident gets $4k worse. A new incident needs every
rule to clear and a $3k recovery. Replayed over the whole day (44 rounds), the watch raised 14 alerts in 9 rounds:
every round with a material drop, and none without one.

**Each alert is diagnosed immediately:**
- the loss split into price moves on held positions, execution cost (vs the day's typical 46 bp), and positions the
  bot holds against its own plan
- each symbol's position and price change
- the model's predictions at the start of the window and now
- the storm now: heading change, how far it is from where its motion pointed 60 s ago, distance to each island, and
  the official advisory then vs now
- on the testnet, the honest replay of the same storm over the same storm ages
- bot health near the drop: rejects, disconnects, errors, restarts, silence, unlogged fills

It then gives one CAUSE:

| CAUSE | When |
|---|---|
| **BOT** | Something went wrong near the drop, or positions held against the bot's own plan lost at least 25% of it |
| **EXECUTION** | Costs are at least 40% of the loss and above 2× typical |
| **STORM (variance)** | Most of the loss is price moves on the model's positions, and the replay lost at least half as much over the same ages |
| **MARKET, BUT NOT IN THE REPLAY** | Price moves on the model's positions, but the replay did not lose: check |
| **UNCLEAR** | No single explanation |

Validate with: `OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3 -I bot/monitor.py --replay 06:54` (runs the watch over
today's logs and prints every alert it would have raised).

**Also during a round:**
- BOT SILENT: no status line for 90 s
- BOT EVENT: errors, rejected orders, disconnects, implausible predictions

**After every round** it writes `bot/logs/round_reports.md` (and `.jsonl`): the exact live PnL by symbol (rounds run
from season reset to season reset), the honest replay of the same storm, a verdict (OK / VARIANCE / LIVE-SPECIFIC),
an accounting check against the bot's exchange ledger, storm track surprise, direction, execution, leftovers, and a
running live-minus-replay gap per symbol (an ALERT for each further −$2.5k).

When you change the bot's settings, add a line to `SETTINGS` in `monitor.py`. The monitor only
reports; it never changes the bot.

**Monitor accounting bug (fixed 2026-10-18 16:30):** the monitor charged every fill to the window between
two storm formations. Trades in the gap between a season reset and the new season's first storm were charged
to the *old* season's round 3, then wiped at its settlement. That left phantom positions in the new season.
The season-80 pre-storm false snipes (PORT +200, HAVEN +250, ARCH +50) made season 79 r3 look like +$8.2k
(really −$2.1k) and made season 80 r3 look like −$6.6k (really +$0.8k, which matches the bot's own exchange
ledger). Rounds now run from season reset to season reset. Fills are also reconciled against the exchange's
positions: a restart's position snapshot, or a gap that lasts 3 status lines, is booked as an
"unlogged fill" anomaly.

**Audit after the fix (34 live rounds, 11 complete seasons 71–81):**
- 9 of 11 seasons positive. Season 79 (−$7.0k) and season 80 (−$16.4k) were negative; the honest replays
  of the same seasons lost about the same (−$6.3k and −$16.1k).
- All 8 rounds below −$3k were **VARIANCE**. The worst was season 80 r2 Wren: −$18.1k live vs −$18.6k
  in replay (a direct hit that turned erratically mid-storm).
- Live beat the replay overall: +$34.6k vs +$24.4k.
- The only persistent gap is **PORT: −$5.5k over 34 rounds (about −$160/round)** worse live (thin book).
- The dashboard often shows a negative total mid-season because rounds 1–2 are frequently negative,
  while round 3 (held to settlement) is usually strongly positive.

**Removed / tested and rejected:**
- The **mispriced-order sniper** is OFF. Replaying every recorded book through it lost money (old
  rules −$53k, hardened −$2.9k). Its fair-value estimate (the house quote ladder) can be wrong and
  stuck, and at the season-80 open it fired 5 false snipes.
- **Stricter PORT entry / no PORT** (`bot/port_variants.py`; 200 held-out seasons, paired against
  production): a PORT buffer of 2% changes PnL by −$26 ± 48 per season, 3% by −$121 ± 71, and dropping
  PORT by −$1,225 ± 166. PORT earns about $1,300/season in replay, well above its live gap (about
  $480/season), so PORT stays at the 1% buffer.
- A **round stop-loss** (−$3k / −$5k) is off. It lowered mean profit and the share of profitable
  seasons, and did not reduce the worst rounds, because big losses come from sudden repricings.

## Keep the machine awake

macOS idle sleep froze the bot for 13–18 minutes at a time on the testnet. It missed whole storms,
and a missed storm means no in-storm trading for the rest of that season. Always run it under
`caffeinate -is` (as below), keep the laptop plugged in with the lid open, or run it on a server.

```bash
caffeinate -is /opt/anaconda3/bin/python3 -I bot/live.py
```

The bot also forces a reconnect after 40 s with no feed frame, fills short observation gaps from the
feed snapshot, and detects a new season by competition id or by a jump in the market date.

## Watching it

- **Web terminal:** your team's positions and PnL. This is authoritative.
- **Console:** a `STATUS` line every 10 s with positions, the season PnL from the exchange's own fills
  (`GET /api/v1/trades`, undocumented but works), and the exchange's unrealized PnL.
- **Results:** `ROUND RESULT` and `SEASON RESULT` lines; print them with `$PY -I bot/summary.py`. After a
  restart the bot only knows PnL since the restart (profit realized earlier is on the website).
- **Order log:** `bot/logs/run-*.jsonl` has every decision (predictions and goals), order, cancel,
  position change and error.
- **Feed recording:** `bot/logs/feed-*.jsonl` is a feed recording. Verify it with
  `python starter_kit/check_recording.py bot/logs/feed-*.jsonl history/`.
- **Book recording:** `bot/logs/books-*.jsonl` records the order book. `bot/book_profile.py` turns it
  into a depth model for `replay.py --book`.
- **Execution gap:** the testnet replays research season N (the market date tells you which).
  `$PY -I bot/replay.py --seasons N-N --models bot/models.pkl --book` is what the same code makes
  with modelled fills; the gap to the live PnL is real execution cost.

## What the testnet showed (2026-10-18)

- The API works with this key on `cornelltradingcompetition.org`. `hurricane.cornelltradingcompetition.org`
  rejects it.
- Formats, all confirmed:
  - order response `{order_id, status, ...}`
  - `/orders/open` → `{orders: [...filled_quantity...]}`
  - `/positions` → `{positions: [{symbol, quantity, average_price, current_price, unrealized_pnl,
    realized_pnl}]}`; flat symbols drop out of the list
  - websocket frames: `quote {bid, ask, bid_size, ask_size}`, `orderbook {bids, asks: [{price,
    quantity}]}`, `trade`, `heartbeat`, `news`
- **Prices must be multiples of $0.01.** The packet says 6 decimals; the server rejects anything
  else. The broker rounds caps the safe way (buy caps down, sell caps up).
- **The book is thin, and thinner during storms.**
  - Median spread is about 0.22–0.32% between storms and 0.28–0.60% during them (HAVEN is widest).
  - Depth within 1% of the best price falls about 30% during storms. PORT is about 94 shares and
    HAVEN about 103; the rest range up to about 380.
  - The book refreshes every ~4 s, which is why the broker works each goal in pieces.
- **Realistic expectation.** The 100 held-out research seasons, filling against storm-time book
  depth, average about **$5.7k per season, profitable in 70% of seasons** (about $5.1k in-storm and
  $0.6k from the final report). With unlimited depth at the touch it would be about $14k, so depth is
  the binding constraint. Extra bot keys and patient execution are the main levers.

## Live testnet results (2026-10-18)

| Season | PnL | Notes |
|---|---|---|
| 65 | ≈ +$6.0k | First live season. Rounds 1–2 lost to the tick bug and a restart (both fixed). |
| 66–69 | partial | The Mac slept and froze the bot for 13–18 minutes at a time, so whole storms were missed. The bot then correctly stopped in-storm trading for those seasons. Season 69 round 1 alone was about +$11.7k. |
| **70** | **+$29,663** | First clean season on the final code: round 1 +$10,237, round 2 +$4,559, round 3 held to settlement ≈ +$14,900. 0 rejected orders, 0 errors, 0 disconnects. |

One season is one draw; the honest long-run expectation is the replay figure, about $5.7k/season and
profitable ~72% of the time.

**Bugs found by running live, all fixed:**

1. Prices must be multiples of $0.01.
2. A missing lock crashed the broker threads.
3. Feed sequence numbers restart every season.
4. Late-join and restart state loss (fixed by catch-up from the bot's own recording).
5. Season state leaked across seasons when round 1 was missed.
6. Exits paid up to 1% regardless of edge, and the live book can run $5–7 from fair value during a
   storm (fixed by fair-value exits).
7. The PnL ledger guessed fill prices (now exact fills from `GET /api/v1/trades`).
8. The final-report values used site damage capped at 100% (now the reported totals; checked against
   all 600 settlements).
9. Sleep and restarts (keep the machine awake with a standalone `caffeinate -dis`).
10. Stale exit caps: after a damage report, an exit's price cap was fixed at that moment. If the price
    drifted past it, the position stayed open into the next storm's formation jump. In season 74 r1,
    HAVEN −142 was carried into round 2 and cost about $1.1k. Now, in rounds 1–2, any position left after
    a report is re-flattened from the current mid every 20 s (`LEFTOVER_RETRY`), so the bot is flat
    before the next storm. ETF positions are included; the announcement follower is excluded; round 3
    is still held to settlement. This is fake-exchange test 8. Deployed 2026-10-18 between storms.

**Season 74 r1 (live $737 vs replay $5,049), explained.** The settings in use then were later
replaced. At storm age 10 (where the model has no skill), the bot bought PORT 200 and HAVEN 250. At
age 30, the formation repricing crashed both, and with flips allowed it reversed to full shorts in a
falling book: PORT sold at 62.5–68.4 against a 69 mid seconds earlier. That is most of the
"$5,450 execution cost": prices moving during aggressive fills, not a broken order engine. It was fixed
by starting at age 30, turning flips off, and the 1% entry buffer. Execution cost per round fell from
$1,723 (morning settings, 25 rounds) to $809 (current settings, 8 rounds).

## Files

| File | Contents |
|---|---|
| `live.py` | Websocket runner (`--smoke-test`, `--dry-run`, live) |
| `strategy.py` | When to trade; goals = (target, price cap, priority) |
| `broker.py` | Works goals against the book: a limit order at the cap for the depth available within it, cancel after 2 s, retry every 3 s, positions polled every 1 s, one rate budget per key, PnL ledger |
| `engine.py` | Live features, identical to the research features (`replay.py --parity`) |
| `train.py` | `models.pkl` (all 600 seasons) and `models_holdout100.pkl` (seasons 1–500, for honest replays) |
| `replay.py` | Runs research seasons through the same code (`--book` for realistic depth) |
| `book_profile.py` | Depth model from recorded books |

## Not in v1

- the formation front-run (t = 60 → 64)
- the dissipation and prelim trades
- final-report trades in rounds 1–2
- the opening fade
- announcement handling (it logs `ANNOUNCEMENT` and does nothing). In the live event, read the
  headline and change the matching constant in `analysis/value.py`.
