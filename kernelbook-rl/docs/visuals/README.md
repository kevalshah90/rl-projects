# Visuals

Self-contained HTML (inline CSS/JS/SVG, no network, light and dark mode). Open in a browser.
They are interactive: Pause / Next stage / Restart, or click a stage number.

| File | Shows |
|---|---|
| [`prime_rl_async_loop.html`](prime_rl_async_loop.html) | The prime-rl loop: vLLM inference, orchestrator, trainer, agent sandboxes, grader. Amber stages (5, 6, 7, 9) are the async overlap: step 2 samples with v1 while the trainer makes v2, so batch B arrives 1 version stale (kept, since `max_staleness = 1`). |
| [`trainer_zoom.html`](trainer_zoom.html) | Zoom into the trainer during stages 5-7 of the loop: tokens -> forward pass -> ratio p_train / p_vLLM and IPO mask -> mismatch KL (monitor only) -> loss mean(-A x ratio), and why the minus and the ratio -> backward into LoRA only -> AdamW -> save adapter -> vLLM v1 to v2. |

GitHub shows the source, not the page. To view without cloning:
`https://htmlpreview.github.io/?<raw file URL>`, or download and open the file.
Context: [NOTEBOOK.md](../NOTEBOOK.md), log entries 2026-11-09 and 2026-11-16.
