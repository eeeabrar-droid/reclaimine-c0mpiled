# Reclaimine: a self-improving lost-item describer (c0mpiled-15, Transpose Track 1A)

**Problem.** Lost-and-found claims are verified by details only the real owner knows. A vision model has to describe a found item well enough to match those details: the scratch, the sticker, the handwritten name. It must also never invent one, because a hallucinated detail can hand a wallet to the wrong person.

**Task.** 16 everyday items photographed from 2-6 angles. For each item, ground-truth tags were frozen before any agent ran, weighted by tier:
tier 1 general (1 pt), tier 2 distinguishable (2 pts), tier 4 personalizing / owner-only (4 pts). Wrong claims cost the same weight; claims marked as known-false cost 2.
The split is by item: 10 dev items for optimization and 6 held-out items never seen during optimization.

**Agents (DeepSeek V4.1 Flash on Luminal Cloud; GLM 5.3 Flash as second judge).**
- *Describer*: writes the item description from the photos (reasoning off).
- *3 parallel proposers* per round, each with a different strategy (coverage / format / precision). Each sees only scores, tier-miss counts and the lessons log, never the tag text.
- *Two-model judge*: per-tag verdict plus an exact quote, verified mechanically. A penalty counts only if both judges flag it.
- *Selector*: screens candidates on dev, confirms the winner with extra runs, and accepts only if the paired item-level 95% bootstrap CI is above 0.

**Video + slides:** `slides/`. Repo: https://github.com/eeeabrar-droid/reclaimine-c0mpiled

**Results** (raw round logs in `results/`, slides in `slides/`):

| Run | What the agents could change | Dev baseline -> kept | Gate | Fresh items (6, never seen) |
|---|---|---|---|---|
| Main run | prompt text (coverage / format / precision proposers) | 48.0 -> 58.2 (+10.1, 95% CI [+2.7, +16.1]) | 1 kept, 4 rejected | +0.9, CI [-5.4, +6.6]: not claimed |
| Pipeline run | pipeline: prompt rewrite / per-photo + merge / marks pass | 50.4 -> 60.3 (+9.9, CI [+1.2, +16.8]), per-photo + merge | 1 kept, 2 rejected | +2.6, CI [-11.3, +13.3]: not claimed |
| Control (swap) | optimized on the 6 held-out items | nothing passed the gate | 0 kept, 5 rejected | baseline vs itself on 10 items: +1.8, CI [-2.8, +5.8] (A/A check) |

Scores are net % of max (gains minus penalties). Each prompt/pipeline is a mean of 5 runs; CIs are paired, item-level bootstrap. Feedback guides later rounds through a lessons log (each proposer's hypothesis, whether it was supported, and the delta), and proposers only ever see scores and tier-miss counts.

**Honest caveats.** Run-to-run noise is large (the same prompt spans ~15 points), so every number is a mean over 5 runs with item-level CIs. Held-out is small (6 items). Dataset: `dataset/tags.json` (frozen tags, splits, known-false claims) + `dataset/photos_neutral/` (1800x1350, EXIF-free). Items are the author's own household items used as stand-ins for lost property.

## Run
```
export OPENROUTER_API_KEY=...   # GLM judge
export LUMINAL_API_KEY=...      # https://flash.luminal.cloud/v1
DEEPSEEK_PROVIDER=luminal DATASET_DIR=/path/to/dataset python3 harness.py --rounds 5    # prompt-level search
DEEPSEEK_PROVIDER=luminal DATASET_DIR=/path/to/dataset python3 harness2.py --rounds 3   # pipeline-level search
python3 analyze.py runs/<run_dir>                                                       # overview + held-out by tier
```
`dataset/tags.json` format: `{"items":[{"id","split":"dev|held_out","photos_neutral":[...],"tags":[{"text","tier","weight"}],"known_false":[...],"inferences":[...]}]}`.
