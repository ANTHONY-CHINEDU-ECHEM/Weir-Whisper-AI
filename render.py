"""Render investigation reports to Markdown and a self contained HTML register."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from ..simulator.scenarios import CAUSE_LABELS
from ..utils.io import write_json

SECTION_TITLES = [
    ("summary", "Summary"),
    ("root_cause", "Why we believe this"),
    ("alternatives", "Alternatives considered"),
    ("impact", "Impact"),
    ("recurrence", "Recurrence"),
    ("recommendations", "Recommendations"),
]


def cite(ids: list[str]) -> str:
    return " ".join(f"[{i}]" for i in ids)


def report_markdown(rep: dict) -> str:
    rc = rep["root_cause"]
    lines = [f"# Investigation {rep['episode_id']}: {rep['column']} {rep['column_name']}", ""]
    lines.append(f"**{rep['headline']}**")
    lines.append("")
    lines.append(f"Window {rep['start'][:16].replace('T', ' ')} to {rep['end'][:16].replace('T', ' ')} ({rep['duration_h']:.1f} h). "
                 f"Detector class: {rep['episode_kind'].replace('_', ' ')}. Writer backend: {rep['backend']}.")
    lines.append("")
    by_sec: dict[str, list[dict]] = {}
    for c in rep["claims"]:
        by_sec.setdefault(c["section"], []).append(c)
    for key, title in SECTION_TITLES:
        if key not in by_sec:
            continue
        lines += [f"## {title}", ""]
        for c in by_sec[key]:
            lines.append(f"* {c['text']} {cite(c['evidence'])}")
        lines.append("")
        if key == "root_cause":
            lines += ["| Hypothesis | Posterior |", "|---|---|"]
            for cause, p in list(rc["posterior"].items())[:5]:
                lines.append(f"| {CAUSE_LABELS[cause]} | {p:.3f} |")
            lines.append("")
    lines += ["## Evidence timeline", "", "| Time | Source | Record | Detail |", "|---|---|---|---|"]
    for t in rep["timeline"]:
        txt = t["text"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {t['time'][:16].replace('T', ' ')} | {t['source'].replace('_', ' ')} | {t['id']} | {txt} |")
    lines.append("")
    v = rep["verification"]
    lines += ["## Verification", "",
              f"{v['claims_supported']} of {v['claims_total']} claims passed the verifier (citation coverage {v['citation_coverage']:.0%}); "
              f"{v['numbers_checked']} numbers checked against the fact registry; {v['evidence_items']} evidence items indexed."]
    for r in v["claims_removed"]:
        lines.append(f"* Removed: {r['text']} ({'; '.join(r['reasons'])})")
    lines.append("")
    return "\n".join(lines)


def write_report(folder: Path, rep: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{rep['episode_id']}.md").write_text(report_markdown(rep), encoding="utf8")
    write_json(folder / f"{rep['episode_id']}.json", rep)


CSS = """
:root{--bg:#f7f7f4;--fg:#1d2321;--muted:#5d6763;--card:#fff;--line:#d9ddd8;--accent:#0f6b5c;--warn:#a4471f}
@media (prefers-color-scheme: dark){:root{--bg:#141817;--fg:#e8ece9;--muted:#9aa5a0;--card:#1c2120;--line:#2d3532;--accent:#4fb7a3;--warn:#e08a60}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:18px;margin:28px 0 10px}
.muted{color:var(--muted)} .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.kpi b{display:block;font-size:22px} table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top;font-size:14px}
th{color:var(--muted);font-weight:600} tr:last-child td{border-bottom:none} .num{text-align:right;font-variant-numeric:tabular-nums}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:10px 0}
summary{cursor:pointer;font-weight:600} .tag{display:inline-block;padding:1px 8px;border-radius:999px;border:1px solid var(--line);font-size:12px;color:var(--muted)}
.wrap{overflow-x:auto}
"""


def write_register(folder: Path, reports: list[dict], recurrence: list[dict]) -> None:
    reps = sorted(reports, key=lambda r: r["start"])
    total = sum(r["impact"]["total_cost_gbp"] for r in reps)
    weep = [r for r in reps if r["root_cause"]["cause"] in CAUSE_LABELS and r["episode_kind"] == "weeping"]
    rows = []
    for r in reps:
        rows.append(
            f"<tr><td>{r['episode_id']}</td><td>{r['start'][:10]}</td><td>{r['column']}</td><td>{html.escape(r['root_cause']['label'])}</td>"
            f"<td class='num'>{list(r['root_cause']['posterior'].values())[0]:.2f}</td><td class='num'>{r['duration_h']:.0f}</td>"
            f"<td class='num'>{r['impact']['total_cost_gbp']:,.0f}</td><td class='num'>{r['verification']['citation_coverage']:.0%}</td></tr>")
    fam_rows = []
    for f in recurrence:
        if not f["recurring"]:
            continue
        fam_rows.append(f"<tr><td>{f['column']}</td><td>{html.escape(f['cause_label'])}</td><td>{f['asset']}</td><td class='num'>{f['occurrences']}</td>"
                        f"<td class='num'>{(f['mean_interval_days'] or 0):.0f}</td><td class='num'>{f['total_cost_gbp']:,.0f}</td></tr>")
    details = []
    for r in reps[:400]:
        claims = "".join(f"<li>{html.escape(c['text'])} <span class='tag'>{html.escape(' '.join(c['evidence'][:4]))}</span></li>" for c in r["claims"] if c["section"] in ("summary", "recommendations"))
        details.append(f"<details><summary>{r['episode_id']} {r['column']} {html.escape(r['headline'])}</summary><ul>{claims}</ul></details>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Investigation Register</title><style>{CSS}</style></head><body><main>
<h1>Weir Whisper AI investigation register</h1><p class="muted">Every detected efficiency loss episode, its diagnosed cause, cost and verifier coverage.</p>
<div class="kpis"><div class="kpi"><span class="muted">Episodes investigated</span><b>{len(reps)}</b></div>
<div class="kpi"><span class="muted">Weeping episodes</span><b>{len(weep)}</b></div>
<div class="kpi"><span class="muted">Estimated cost</span><b>GBP {total:,.0f}</b></div>
<div class="kpi"><span class="muted">Recurring families</span><b>{len(fam_rows)}</b></div></div>
<h2>Recurring causes</h2><div class="wrap"><table><tr><th>Column</th><th>Cause</th><th>Asset</th><th class="num">Occurrences</th><th class="num">Mean interval (days)</th><th class="num">Cost (GBP)</th></tr>{''.join(fam_rows)}</table></div>
<h2>All episodes</h2><div class="wrap"><table><tr><th>Episode</th><th>Start</th><th>Column</th><th>Root cause</th><th class="num">Posterior</th><th class="num">Hours</th><th class="num">Cost (GBP)</th><th class="num">Verified</th></tr>{''.join(rows)}</table></div>
<h2>Findings</h2>{''.join(details)}
</main></body></html>"""
    (folder / "register.html").write_text(page, encoding="utf8")
    df = pd.DataFrame([{"episode_id": r["episode_id"], "start": r["start"], "end": r["end"], "column": r["column"], "kind": r["episode_kind"],
                        "root_cause": r["root_cause"]["cause"], "posterior": list(r["root_cause"]["posterior"].values())[0],
                        "cost_gbp": r["impact"]["total_cost_gbp"], "excess_steam_t": r["impact"]["excess_steam_t"], "offspec_t": r["impact"]["offspec_t"],
                        "family_id": r["family_id"]} for r in reps])
    df.to_csv(folder / "register.csv", index=False)
