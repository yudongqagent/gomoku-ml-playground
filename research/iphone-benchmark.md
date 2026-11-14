# Measuring on-device speed on iPhone (A20 Pro)

This is the only way to get real on-device speed numbers for the scoring
kernel: there is no Apple hardware on the training VM, so everything
measured there is CPU-only. The numbers from this page, run on your iPhone,
are the real ones. **No on-device numbers exist yet — all values below are
to be measured.**

## What it measures

The ⚡ Benchmark tab in the Gomoku ML Playground web app runs the v0
scoring kernel over a fixed set of 60 positions (20 tactical, 3 structured,
37 random midgame boards) and reports:

- **WebGPU throughput** — positions/second and ms/position for the WGSL
  compute shader running on the A20 Pro GPU. One "position" is a full
  225-cell scoring pass. Timing includes reading the results back.
- **JS reference throughput** — the same 60 positions scored by plain
  JavaScript on the same phone, for comparison.
- **Self-check** — the shader's scores are compared against the JavaScript
  reference and against frozen Python scores on all 13,500 cells (60
  positions × 225). Both mismatch counts must be **0**; if either is not 0,
  the numbers are not trustworthy — report the mismatch counts instead.

## Requirements

- Your iPhone (A20 Pro chip).
- **Safari, iOS 26 or later.** WebGPU needs Safari 26+. On older iOS the
  page will say WebGPU is unavailable — that is expected, not an error
  (see "If WebGPU is unavailable" below).
- The hosted Gomoku ML Playground page — the same link you use to play
  the game. (If you are reading this in the repo, the page is the
  deployed web app; the Benchmark tab is part of it.)

## Steps

1. **Charge the phone** (or plug it in) and turn **Low Power Mode off**
   (Settings → Battery). Low Power Mode throttles the GPU and the numbers
   will be lower and less stable.
2. **Close other apps** and extra Safari tabs if convenient — not
   required, but it keeps the numbers stable.
3. Open the Gomoku page in **Safari** and tap the **⚡ Benchmark** tab at
   the top (next to Play and Trainer).
4. Tap **Run benchmark**. The page first self-checks the shader against
   the JavaScript reference and the frozen Python scores, then runs 3
   warmup runs and 20 timed runs over the 60 positions. It shows progress
   while it runs; it usually takes well under a minute.
5. When it finishes, the results table shows the WebGPU and JS numbers
   plus the self-check counts. The page also shows a copy-ready summary
   line at the bottom of the results.
6. **Run it 2–3 times** (tap Run benchmark again) and note each run's
   numbers — the first run can differ from later ones as the phone warms
   up. Report all runs, not just the fastest.

## What to report back

Copy the summary lines from the results, plus your phone details. Example
of what to send (values to be measured):

```
iPhone model: (e.g. iPhone 17 Pro) / iOS version: (e.g. 26.1)
Adapter: (the adapter line shown in the results)
Run 1: WebGPU ___ positions/sec (___ ms/position); JS ___ positions/sec (___ ms/position). Self-check: 0 mismatches.
Run 2: WebGPU ___ positions/sec (___ ms/position); JS ___ positions/sec (___ ms/position). Self-check: 0 mismatches.
Run 3: WebGPU ___ positions/sec (___ ms/position); JS ___ positions/sec (___ ms/position). Self-check: 0 mismatches.
```

If the self-check shows any mismatches (not 0), report the exact mismatch
counts and the two lines anyway — that means the shader disagrees with the
reference on this device, which is a bug to fix, not a number to use.

## If WebGPU is unavailable

If the page says "WebGPU is not available here", it means this browser
does not expose WebGPU — on iPhone that almost always means iOS older than
26. In that case:

1. Tap **Run JS-reference benchmark only** and report those numbers
   (labelled JS, not GPU).
2. Report your iOS version (Settings → General → About → iOS Version).
3. Updating to iOS 26+ and re-running is the way to get the GPU numbers.

## What the numbers will be used for

Comparing the on-device GPU speed against the VM's CPU kernel numbers, to
decide how much of the self-play scoring can move on-device and whether
the portable kernel (WebGPU now, Metal later) is fast enough to score
positions during live play. No decision is made from estimated or
simulated numbers — only from what this page measures on the phone.
