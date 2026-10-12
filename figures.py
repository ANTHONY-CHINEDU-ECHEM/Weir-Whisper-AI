"""README figures.

Colours follow a validated categorical order (blue, orange, aqua, yellow, magenta, green,
violet, red); sequential encodings use a single blue ramp. Text is always ink, never the
series colour, and every multi series chart carries a legend.
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from ..simulator.scenarios import WEEP_CAUSES  # noqa: E402
from ..utils.io import read_json, read_pickle  # noqa: E402

LOG = logging.getLogger("weir_whisper.figures")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
GREY = "#b9b8b2"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
CAUSE_COLOR = {c: SERIES[i] for i, c in enumerate(WEEP_CAUSES)}
CAUSE_COLOR.update({"flooding": "#9a9993", "foaming": "#52514e", "analyzer_fault": "#cfcec8"})
SHORT = {"turndown": "Turndown", "steam_valve_stiction": "Valve stiction", "reboiler_fouling": "Reboiler fouling", "tray_damage": "Tray damage",
         "pressure_rise": "Pressure rise", "subcooled_feed": "Subcooled feed", "flooding": "Flooding", "foaming": "Foaming", "analyzer_fault": "Analyzer fault"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def save(fig, folder, name):
    fig.savefig(folder / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("figure written: %s", name)


def fig_architecture(folder):
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.7)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=color, lw=2))
        ax.text(x + 0.12, y + h - 0.2, title, fontsize=10.5, fontweight="bold", va="top", color=INK)
        ax.text(x + 0.12, y + h - 0.55, body, fontsize=8.6, va="top", color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))

    box(0.1, 3.4, 2.4, 2.6, "Plant data", "Historian (5 min, 20 tags\nper column)\nAlarms and operator actions\nShift logs (free text)\nWork orders, lab, gamma\nscans, MOCs\nEngineering library", SERIES[0])
    box(2.95, 3.4, 2.5, 2.6, "Signal analytics", "Hydraulic soft sensor\n(weep margin, flood, dP)\nNormal behaviour models\nGradient boosted detector\n(5 classes)\nEpisode segmentation\nCost and efficiency model", SERIES[1])
    box(5.9, 3.4, 2.5, 2.6, "Knowledge graph", "Assets, tags, records,\nsymptoms, causes, episodes\nEntity linking with aliases\nTemporal OVERLAPS edges\nSymptom to cause weights", SERIES[2])
    box(8.85, 3.4, 2.5, 2.6, "Hybrid GraphRAG", "BM25 lexical channel\nLSA dense channel\nGraph and time channel\nKnowledge channel\nReciprocal rank fusion\nLearned linear reranker", SERIES[3])
    box(0.1, 0.25, 3.6, 2.6, "Agents", "Signal analyst\nRecords analyst\nRoot cause (Bayesian over 9\nhypotheses)\nImpact, recurrence", SERIES[6])
    box(4.15, 0.25, 3.0, 2.6, "Writer", "Offline templates (default)\nor Claude via the\nAnthropic API\nClaims cite evidence ids", SERIES[4])
    box(7.6, 0.25, 2.6, 2.6, "Verifier", "Citation exists\nCitation inside window\nNumbers match the\nfact registry\nUnsupported claims removed", SERIES[7])
    box(10.65, 0.25, 2.25, 2.6, "Outputs", "Investigation reports\n(Markdown, JSON)\nRecurrence register\nHTML register\nREST API and Q and A", SERIES[5])
    arrow(2.5, 4.7, 2.95, 4.7)
    arrow(5.45, 4.7, 5.9, 4.7)
    arrow(8.4, 4.7, 8.85, 4.7)
    arrow(10.1, 3.4, 2.4, 2.85)
    arrow(3.7, 1.55, 4.15, 1.55)
    arrow(7.15, 1.55, 7.6, 1.55)
    arrow(10.2, 1.55, 10.65, 1.55)
    ax.text(0.1, 6.65, "Weir Whisper AI architecture", fontsize=14, fontweight="bold", va="top")
    save(fig, folder, "architecture.png")


def fig_year_overview(settings, folder, column="C201"):
    raw = settings.path("raw")
    h = pd.read_parquet(raw / "historian" / f"{column}.parquet").set_index("timestamp")
    tags = pd.read_csv(raw / "tag_dictionary.csv")
    roles = tags[tags["column"] == column].set_index("tag")["role"].to_dict()
    h = h.rename(columns=roles)
    feats = pd.read_parquet(settings.path("processed") / f"features_{column}.parquet")
    truth = read_json(raw / "ground_truth" / "episodes.json")
    d = h.resample("2h").mean()
    f = feats.resample("2h").mean()
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.2), sharex=True)
    design = settings.column(column)
    axes[0].plot(d.index, d["feed_flow"] / design["feed_design_t_h"] * 100, color=SERIES[0], lw=1.2)
    axes[0].set_ylabel("Feed, % design")
    axes[1].plot(d.index, d["steam_flow"], color=SERIES[0], lw=1.2)
    axes[1].set_ylabel("Steam, t/h")
    axes[2].plot(f.index, f["dp_strip_resid"] * 100, color=SERIES[0], lw=1.2)
    axes[2].set_ylabel("Strip dP vs\nexpected, %")
    seen = set()
    for t in truth:
        if t["column"] != column:
            continue
        for ax in axes:
            ax.axvspan(pd.Timestamp(t["start"]), pd.Timestamp(t["end"]), color=CAUSE_COLOR[t["cause"]], alpha=0.28, lw=0,
                       label=SHORT[t["cause"]] if ax is axes[0] and t["cause"] not in seen else None)
        seen.add(t["cause"])
    axes[0].legend(loc="lower left", ncol=5, fontsize=9)
    axes[0].set_title(f"{column} {design['name']}: one year of historian data with the true scenario windows shaded")
    save(fig, folder, "year_overview_c201.png")


def fig_signatures(settings, folder):
    raw = settings.path("raw")
    truth = read_json(raw / "ground_truth" / "episodes.json")
    tags = pd.read_csv(raw / "tag_dictionary.csv")
    picks = []
    for cause in ("steam_valve_stiction", "turndown", "reboiler_fouling", "subcooled_feed"):
        t = next((x for x in truth if x["cause"] == cause and x["material_weeping"]), None)
        if t:
            picks.append(t)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, t in zip(axes.ravel(), picks):
        col = t["column"]
        h = pd.read_parquet(raw / "historian" / f"{col}.parquet").set_index("timestamp")
        h = h.rename(columns=tags[tags["column"] == col].set_index("tag")["role"].to_dict())
        gt = pd.read_parquet(raw / "ground_truth" / f"signals_{col}.parquet").set_index("timestamp")
        if t["cause"] == "reboiler_fouling":
            t0, t1 = pd.Timestamp(t["start"]) - pd.Timedelta(days=3), pd.Timestamp(t["end"]) + pd.Timedelta(days=3)
        elif t["cause"] == "steam_valve_stiction":
            t0 = pd.Timestamp(t["weep_start"]) - pd.Timedelta(hours=3)
            t1 = t0 + pd.Timedelta(hours=14)
        else:
            t0 = pd.Timestamp(t["weep_start"]) - pd.Timedelta(hours=10)
            t1 = t0 + pd.Timedelta(hours=34)
        w = h.loc[t0:t1]
        g = gt.loc[t0:t1]
        wf = np.maximum(g["true_wf_rect"], g["true_wf_strip"])
        if t["cause"] == "steam_valve_stiction":
            ax2 = ax.twinx()
            ax2.plot(w.index, w["steam_flow"], color=SERIES[0], lw=1.3, label="Steam flow, t/h", zorder=2)
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
            ax.plot(w.index, w["steam_valve_op"], color=SERIES[1], lw=1.6, label="Valve output, %")
            ax2.set_ylabel("Steam, t/h")
            ax.set_ylabel("Valve output, %")
        elif t["cause"] == "reboiler_fouling":
            ax.plot(w.index, w["steam_valve_op"], color=SERIES[1], lw=1.2, label="Valve output, %")
            ax2 = ax.twinx()
            ax2.plot(w.index, w["chest_pressure"], color=SERIES[0], lw=1.2, label="Chest pressure, barg")
            ax2.set_ylabel("Chest pressure, barg")
            ax.set_ylabel("Valve output, %")
        else:
            ax.plot(w.index, w["dp_rect"], color=SERIES[0], lw=1.4, label="Rectifying dP, mbar")
            ax.plot(w.index, w["dp_strip"], color=SERIES[1], lw=1.4, label="Stripping dP, mbar")
            ax.set_ylabel("dP, mbar")
            ax2 = None
        top = ax.get_ylim()[1]
        ax.fill_between(w.index, ax.get_ylim()[0], top, where=(wf.reindex(w.index).fillna(0) > 0.03).to_numpy(), color=SERIES[7], alpha=0.10, lw=0, label="True weeping")
        ax.set_title(f"{SHORT[t['cause']]} on {col} ({t['scenario_id']})")
        handles, labels = ax.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels += l2
            ax2.grid(False)
        ax.legend(handles, labels, loc="upper left" if t["cause"] == "reboiler_fouling" else "lower left", fontsize=8)
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Four weeping causes leave four different fingerprints", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, folder, "cause_signatures.png")


def fig_detection(settings, folder):
    proc = settings.path("processed")
    te = np.load(proc / "forward_test_mask.npy")
    from ..evaluation.truth import sample_labels
    ys, P = [], {"Gradient boosted detector (forward split)": [], "Same detector, unseen column": [], "Physics rule": [], "Isolation forest": []}
    for c in settings.columns:
        ys.append((sample_labels(settings, c) == 1)[te])
        P["Gradient boosted detector (forward split)"].append(np.load(proc / f"probs_forward_{c}.npy")[te, 1])
        P["Same detector, unseen column"].append(np.load(proc / f"probs_loco_{c}.npy")[te, 1])
        P["Physics rule"].append(np.load(proc / f"baseline_physics_rule_{c}.npy")[te])
        P["Isolation forest"].append(np.load(proc / f"baseline_isolation_forest_{c}.npy")[te])
    y = np.concatenate(ys)
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    for i, (name, ps) in enumerate(P.items()):
        p = np.concatenate(ps)
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=SERIES[i], label=name)
    ax.set_xlabel("Recall (weeping samples)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Weeping detection on the held out period (June to September)")
    ax.legend(loc="lower left", fontsize=9)
    save(fig, folder, "detection_pr.png")


def fig_confusion(settings, folder):
    rc = read_json(settings.path("metrics") / "root_cause.json")
    cm = np.array(rc["signals_and_records"]["confusion_matrix"]["matrix"])
    labels = [SHORT[c] for c in rc["signals_and_records"]["confusion_matrix"]["labels"]]
    keep = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
    cm = cm[keep][:, keep]
    labels = [l for l, k in zip(labels, keep) if k]
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    ax.imshow(cm, cmap=BLUES)
    ax.grid(False)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() * 0.55 else INK, fontsize=10)
    ax.set_xlabel("Diagnosed cause")
    ax.set_ylabel("True cause")
    ax.set_title("Root cause diagnosis on every detected episode")
    save(fig, folder, "root_cause_confusion.png")


def fig_retrieval(settings, folder):
    rt = read_json(settings.path("metrics") / "retrieval.json")
    modes = ["bm25", "dense", "hybrid", "hybrid_graph", "full"]
    names = ["BM25", "Dense (LSA)", "Hybrid (RRF)", "Hybrid + graph", "Hybrid + graph\n+ reranker"]
    metrics = [("capped_recall@10", "Capped recall at 10"), ("mrr", "MRR"), ("ndcg@10", "nDCG at 10")]
    x = np.arange(len(modes))
    wbar = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, (m, lab) in enumerate(metrics):
        vals = [rt["modes"][mo][m] for mo in modes]
        bars = ax.bar(x + (i - 1) * wbar, vals, wbar - 0.03, color=SERIES[i], label=lab)
        for b, v in zip(bars, vals):
            if m == "ndcg@10":
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Evidence retrieval for {rt['queries']} investigations (ablation)")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, folder, "retrieval_ablation.png")


def fig_cost(settings, folder):
    im = read_json(settings.path("metrics") / "impact.json")
    df = pd.DataFrame(im["rows"])
    df = df[(df["true_cost"] > 500) & (df["est_cost"] > 0)]
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for i, cause in enumerate(WEEP_CAUSES):
        s = df[df["cause"] == cause]
        if len(s):
            ax.scatter(s["true_cost"], s["est_cost"], s=60, color=CAUSE_COLOR[cause], edgecolor=SURFACE, linewidth=2, label=SHORT[cause], zorder=3)
    lo, hi = 500, max(df["true_cost"].max(), df["est_cost"].max()) * 1.5
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
    ax.fill_between([lo, hi], [lo * 0.75, hi * 0.75], [lo * 1.25, hi * 1.25], color=GRID, alpha=0.6, lw=0, zorder=0, label="Within 25 percent")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True cost from the counterfactual simulation, GBP")
    ax.set_ylabel("Estimated cost from plant data only, GBP")
    ax.set_title("Cost estimate per weeping scenario")
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, folder, "cost_parity.png")


def fig_cost_by_cause(settings, folder):
    reps = read_json(settings.path("reports") / "all_reports.json")
    df = pd.DataFrame([{"cause": r["root_cause"]["cause"], "cost": r["impact"]["total_cost_gbp"]} for r in reps])
    s = df.groupby("cause")["cost"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.barh([SHORT[c] for c in s.index], s.values / 1000, color=SERIES[0], height=0.62)
    for b, v in zip(bars, s.values):
        ax.text(max(v / 1000, 0) + s.max() / 1000 * 0.01, b.get_y() + b.get_height() / 2, f"GBP {v / 1000:,.0f}k", va="center", fontsize=9, color=INK2)
    ax.set_xlabel("Estimated cost over the year, thousand GBP")
    ax.set_title("Where the money went, by diagnosed cause")
    ax.set_xlim(0, s.max() / 1000 * 1.18)
    ax.grid(axis="y", visible=False)
    save(fig, folder, "cost_by_cause.png")


def fig_recurrence(settings, folder):
    reps = read_json(settings.path("reports") / "all_reports.json")
    cols = settings.columns
    fig, ax = plt.subplots(figsize=(13, 4.6))
    seen = set()
    for r in reps:
        if not r.get("material", True):
            continue
        y = cols.index(r["column"])
        c = r["root_cause"]["cause"]
        t0, t1 = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        width = max((t1 - t0).total_seconds() / 86400, 0.8)
        ax.barh(y, width, left=t0, height=0.55, color=CAUSE_COLOR[c], edgecolor=SURFACE, linewidth=1.5, label=SHORT[c] if c not in seen else None)
        seen.add(c)
    fams = read_json(settings.path("reports") / "recurrence.json")
    for f in fams:
        if f["recurring"] and f["cause"] in ("steam_valve_stiction", "turndown", "reboiler_fouling", "subcooled_feed"):
            y = cols.index(f["column"])
            ax.text(pd.Timestamp(f["last"]) + pd.Timedelta(days=4), y + 0.33, f"{f['occurrences']}x {SHORT[f['cause']].lower()}", fontsize=8, color=INK2)
    ax.set_yticks(range(len(cols)), cols)
    ax.invert_yaxis()
    ax.set_title("Material episodes per column, coloured by diagnosed cause (minor blips hidden)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=9, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    save(fig, folder, "recurrence_timeline.png")


def fig_graph(settings, folder):
    g = read_pickle(settings.path("graph") / "graph.pkl")
    reps = read_json(settings.path("reports") / "all_reports.json")
    rep = next((r for r in reps if r["root_cause"]["cause"] == "steam_valve_stiction" and r["recurrence"]["earlier_episodes"]), reps[0])
    eid = rep["episode_id"]
    cited = {e for c in rep["claims"] for e in c["evidence"]}
    nodes = {eid, rep["column"]}
    for _, n, d in g.out_edges(eid, data=True):
        if n in cited:
            nodes.add(n)
    for n in list(nodes):
        for _, m, d in g.out_edges(n, data=True):
            if d.get("rel") in ("MENTIONS", "ON_ASSET", "REPORTS") and g.nodes[m].get("kind") in ("equipment", "symptom", "column"):
                nodes.add(m)
    for n in list(nodes):
        if g.nodes[n].get("kind") == "symptom":
            for _, m, d in g.out_edges(n, data=True):
                if d.get("rel") == "INDICATES" and m.endswith(rep["root_cause"]["cause"]):
                    nodes.add(m)
    for k in rep["knowledge_refs"]:
        if k in g:
            nodes.add(k)
            nodes.add(f"CAUSE:{rep['root_cause']['cause']}")
    for e in rep["recurrence"]["earlier_episodes"][-2:]:
        nodes.add(e)
    sub = nx.DiGraph(g.subgraph(nodes))
    kind_color = {"episode": SERIES[7], "column": SERIES[0], "equipment": SERIES[0], "instrument": SERIES[0], "shift_log": SERIES[1],
                  "work_order": SERIES[3], "operator_action": SERIES[3], "alarm": SERIES[4], "gamma_scan": SERIES[3], "lab": SERIES[4],
                  "symptom": SERIES[2], "cause": SERIES[6], "knowledge": SERIES[5], "moc": SERIES[3]}
    pos = nx.spring_layout(sub, seed=4, k=1.4)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    nx.draw_networkx_edges(sub, pos, ax=ax, edge_color=GREY, width=1.0, arrows=False)
    for kind in sorted({sub.nodes[n].get("kind") for n in sub}):
        ns = [n for n in sub if sub.nodes[n].get("kind") == kind]
        nx.draw_networkx_nodes(sub, pos, nodelist=ns, node_color=kind_color.get(kind, GREY), node_size=[900 if n == eid else 380 for n in ns],
                               edgecolors=SURFACE, linewidths=2, ax=ax, label=kind.replace("_", " "))
    labels = {n: (n.replace("SYM:", "").replace("CAUSE:", "").replace("_", " ") if ":" in n else n) for n in sub}
    nx.draw_networkx_labels(sub, pos, labels=labels, font_size=7.5, font_color=INK, ax=ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8.5, ncol=1, markerscale=0.7)
    ax.set_title(f"Evidence subgraph behind investigation {eid} ({rep['column']}, {SHORT[rep['root_cause']['cause']].lower()})", loc="left")
    save(fig, folder, "evidence_subgraph.png")


def make_all(settings) -> None:
    style()
    folder = settings.path("figures")
    fig_architecture(folder)
    for fn in (fig_year_overview, fig_signatures, fig_detection, fig_confusion, fig_retrieval, fig_cost, fig_cost_by_cause, fig_recurrence, fig_graph):
        try:
            fn(settings, folder)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("figure %s failed: %s", fn.__name__, exc)
