"""Build the final project report (report/final_report.pdf) from the result files in results/.

    python demo.py                 # makes results/demo_closure.png (Figure 6)
    python report/make_report.py

Every number in the figures is read from results/; numbers in the text were copied from the same files.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
OUT = ROOT / "report" / "final_report.pdf"

GROUP = "Machine Learning INET - 6"
MEMBERS = ""          # e.g. "Name One, Name Two, Name Three"; left out of the title block when empty

# Chart colours: reference categorical slots 1-3 (validated all-pairs) + neutral greys for baselines
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, LIGHT_GREY, INK, MUTED, GRID = "#52514e", "#9a9993", "#0b0b0b", "#52514e", "#e6e5e0"


# ----------------------------------------------------------------------------------------------- figures
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(LIGHT_GREY)
    ax.spines["bottom"].set_color(LIGHT_GREY)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


def fig_impact():
    df = pd.read_csv(RES / "incident_impact_summary.csv")
    df = df[df.hours == "all hours"].set_index("kind")
    order = ["closure", "other", "collision", "hazard", "sigalert", "no incident"]
    names = {"closure": "Closure", "other": "Other", "collision": "Collision", "hazard": "Hazard",
             "sigalert": "SigAlert", "no incident": "No incident (control)"}
    d = df.loc[order]
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=200)
    y = np.arange(len(d))[::-1]
    col = [BLUE if k != "no incident" else LIGHT_GREY for k in order]
    ax.barh(y, d.speed_drop_mph, height=0.55, color=col)
    ax.errorbar(d.speed_drop_mph, y, xerr=[d.speed_drop_mph - d.speed_drop_lo, d.speed_drop_hi - d.speed_drop_mph],
                fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    for yi, (k, r) in zip(y, d.iterrows()):
        ax.text(max(r.speed_drop_hi, 0) + 0.15, yi, f"{r.speed_drop_mph:.2f}  (n = {int(r.incidents):,})",
                va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y, [names[k] for k in order])
    ax.axvline(0, color=LIGHT_GREY, lw=0.8)
    ax.set_xlabel("mean speed drop in the first hour after the report (mph, 95% CI)")
    ax.set_xlim(-4, 8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_impact.png")
    plt.close(fig)


def fig_mae():
    d = json.loads((RES / "incident_eval_nograph.json").read_text())
    groups = ["normal", "closure", "collision", "hazard", "sigalert"]
    labels = ["Normal", "Closure", "Collision", "Hazard", "SigAlert"]
    series = [("last value", "Last value", LIGHT_GREY), ("dcrnn_check_plain_ft", "DCRNN (graph)", BLUE),
              ("dcrnn_check_nograph_ft", "DCRNN, no graph", AQUA)]
    fig, ax = plt.subplots(figsize=(6.4, 2.7), dpi=200)
    x = np.arange(len(groups))
    w = 0.26
    for i, (k, name, c) in enumerate(series):
        vals = [d[k][g]["mae"] for g in groups]
        ax.bar(x + (i - 1) * w, vals, width=w - 0.03, color=c, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE over the next hour (mph)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, 5)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_mae_by_kind.png")
    plt.close(fig)


def fig_phase():
    d = json.loads((RES / "closure_errors.json").read_text())
    phases = ["0-15 min", "15-30 min", "30-60 min", "60+ min"]
    series = [("last value", "Last value", GREY, ":"), ("dcrnn_check_plain_ft", "DCRNN (graph)", BLUE, "-"),
              ("dcrnn_check_nograph_ft", "DCRNN, no graph", AQUA, "-")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), dpi=200)
    for ax, (metric, title) in zip(axes, [("mae", "MAE (mph)"), ("bias", "mean signed error (mph; > 0 = too fast)")]):
        for k, name, c, ls in series:
            ax.plot(phases, [d[k]["phase"][p][metric] for p in phases], color=c, ls=ls, lw=2, marker="o",
                    ms=4, label=name)
        ax.set_ylabel(title, fontsize=7.5)
        ax.set_xlabel("time since the closure was reported", fontsize=7.5)
        ax.grid(axis="y", color=GRID, lw=0.6)
        if metric == "bias":
            ax.axhline(0, color=LIGHT_GREY, lw=0.8)
        style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_closure_phase.png")
    plt.close(fig)


def fig_gating():
    d = json.loads((RES / "graph_gating.json").read_text())["closure"]
    rows = [("graph", "DCRNN (graph), reference"), ("graph, gate 0", "Gate both directions"),
            ("gate-trained, gate 0", "Gate both, fine-tuned with gate"),
            ("graph, gate upstream only", "Gate upstream only"), ("graph, gate downstream only", "Gate downstream only"),
            ("graph, gate 0.5", "Half gate (g = 0.5)"), ("graph, gate every sensor (placebo)", "Gate every sensor (placebo)"),
            ("no graph", "DCRNN trained without graph"), ("last value", "Last value")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
    y = np.arange(len(rows))[::-1]
    ref = d["graph"]["disrupted"]["bias"]
    ax.axvline(ref, color=BLUE, lw=1, ls="--")
    ax.axvline(0, color=LIGHT_GREY, lw=0.8)
    for yi, (k, name) in zip(y, rows):
        s = d[k]["disrupted"]
        c = BLUE if k == "graph" else (ORANGE if "gate" in k else LIGHT_GREY)
        ax.plot(s["bias_ci"], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(s["bias"], yi, "o", color=c, ms=6, mec="white", mew=1.5)
        ax.annotate(f"{s['bias']:+.2f}", (1.0, yi), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y, [r[1] for r in rows])
    ax.set_xlim(-1, 13)
    ax.set_xlabel("bias on disrupted closure readings (mph; > 0 = too fast)", fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_gating.png")
    plt.close(fig)


def fig_gap():
    d = json.loads((RES / "graph_gating.json").read_text())["by_gap_down"]
    bins = ["neighbours slower", "0-5 mph faster", "5-15 mph faster", ">15 mph faster"]
    labels = ["slower", "0-5 faster", "5-15 faster", ">15 faster"]
    series = [("last value", "Last value", GREY, ":"), ("graph", "DCRNN (graph)", BLUE, "-"),
              ("gate-trained, gate 0", "Gated (fine-tuned)", ORANGE, "-"), ("no graph", "DCRNN, no graph", AQUA, "-")]
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=200)
    for k, name, c, ls in series:
        ax.plot(labels, [d[k][b]["bias"] for b in bins], color=c, ls=ls, lw=2, marker="o", ms=4, label=name)
    n = d["graph"]
    ax.set_xticks(range(len(bins)), [f"{l}\n({n[b]['events']} closures)" for l, b in zip(labels, bins)],
                  fontsize=7)
    ax.set_xlabel("downstream neighbours' speed relative to the closure sensor at forecast time (mph)", fontsize=7.5)
    ax.set_ylabel("bias (mph; > 0 = too fast)", fontsize=7.5)
    ax.axhline(0, color=LIGHT_GREY, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_gap.png")
    plt.close(fig)


def fig_recovery():
    d = json.loads((RES / "recovery_prior.json").read_text())["groups"]
    groups = ["usual congestion", "unusual slowdown", "closure"]
    labels = ["Usual congestion", "Unusual slowdown\n(no incident)", "Closure"]
    series = [("actual", "Actual (test)", INK, "D"), ("base", "Training base rate", LIGHT_GREY, "s"),
              ("dcrnn_check_plain_ft", "DCRNN (graph)", BLUE, "o"), ("dcrnn_check_nograph_ft", "DCRNN, no graph", AQUA, "o"),
              ("dcrnn_check_slowdown_ft", "DCRNN, slowdowns weighted 5×", ORANGE, "o")]
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=200)
    y = np.arange(len(groups))[::-1]
    for i, (k, name, c, m) in enumerate(series):
        off = (i - 2) * 0.13
        if k == "actual":
            vals = [d["last value"][g]["actual_recovery_60"] for g in groups]
        elif k == "base":
            vals = [d["last value"][g]["base_rate_recovery_60"] for g in groups]
        else:
            vals = [d[k][g]["pred_recovery_60"] for g in groups]
        ax.plot(vals, y + off, m, color=c, ms=6, mec="white", mew=1, label=name, ls="none")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("speed gain over the next 60 minutes (mph), from slowed traffic", fontsize=8)
    ax.axvline(0, color=LIGHT_GREY, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_recovery.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ report
def fonts():
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(base / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(base / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Italic", str(base / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-BoldItalic", str(base / "DejaVuSans-BoldOblique.ttf")))
    from reportlab.lib.fonts import addMapping
    addMapping("DejaVu", 0, 0, "DejaVu")
    addMapping("DejaVu", 1, 0, "DejaVu-Bold")
    addMapping("DejaVu", 0, 1, "DejaVu-Italic")
    addMapping("DejaVu", 1, 1, "DejaVu-BoldItalic")


def styles():
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="DejaVu", fontSize=9.2, leading=13.2,
                          spaceAfter=6, textColor=colors.HexColor("#1a1a19"))
    return {
        "title": ParagraphStyle("title", parent=body, fontName="DejaVu-Bold", fontSize=17, leading=22,
                                alignment=TA_CENTER, spaceAfter=6),
        "sub": ParagraphStyle("sub", parent=body, alignment=TA_CENTER, textColor=colors.HexColor(MUTED),
                              fontSize=9.5, spaceAfter=2),
        "h1": ParagraphStyle("h1", parent=body, fontName="DejaVu-Bold", fontSize=12.5, leading=16, spaceBefore=10,
                             spaceAfter=5),
        "h2": ParagraphStyle("h2", parent=body, fontName="DejaVu-Bold", fontSize=10.2, leading=14, spaceBefore=7,
                             spaceAfter=3),
        "body": body,
        "abstract": ParagraphStyle("abstract", parent=body, leftIndent=18, rightIndent=18, fontSize=8.9,
                                   leading=12.6),
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=3),
        "caption": ParagraphStyle("caption", parent=body, fontSize=8, leading=10.8, textColor=colors.HexColor(MUTED),
                                  spaceAfter=10),
        "cell": ParagraphStyle("cell", parent=body, fontSize=7.8, leading=10, spaceAfter=0),
        "cellb": ParagraphStyle("cellb", parent=body, fontName="DejaVu-Bold", fontSize=7.8, leading=10, spaceAfter=0),
        "code": ParagraphStyle("code", parent=body, fontName="Courier", fontSize=7.6, leading=9.8, leftIndent=10,
                               spaceAfter=1),
    }


def table(rows, S, widths, bold_rows=()):
    data = [[Paragraph(str(c), S["cellb"] if i == 0 or i in bold_rows else S["cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.HexColor(INK)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor(LIGHT_GREY)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor(INK)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def figure(path, caption, S, width=6.4):
    img = Image(str(path))
    img.drawHeight = width * inch * img.imageHeight / img.imageWidth
    img.drawWidth = width * inch
    return KeepTogether([img, Spacer(1, 3), Paragraph(caption, S["caption"])])


def build():
    fonts()
    S = styles()
    P = lambda text, s="body": Paragraph(text, S[s])
    B = lambda text: Paragraph(text, S["bullet"], bulletText="•")
    gg = json.loads((RES / "graph_gating.json").read_text())
    story = []

    # Title
    story += [Spacer(1, 6),
              P("Where Traffic GNNs Fail During Road Closures, and Why", "title"),
              P("Forecasting freeway speeds through recorded closures with DCRNN", "sub"),
              P(f"ML Fall 2026 · Final project report · Group: {GROUP}", "sub")]
    if MEMBERS:
        story.append(P(MEMBERS, "sub"))
    story.append(P("November 2026", "sub"))
    story.append(Spacer(1, 8))

    story += [P("<b>Abstract.</b> We study how a graph neural network for traffic forecasting (DCRNN) behaves "
                "during recorded freeway closures and other incidents, using six months of California PeMS speeds "
                "(TraffiDent, District 8, 898 sensors) with California Highway Patrol (CHP) incident records as ground truth. Closures cause the "
                "largest slowdown of any incident type: 3.6 mph on average in the first hour, against 0.06 mph at "
                "the same sensors on incident-free days. DCRNN beats simple baselines in normal traffic (1.84 vs "
                "2.01 mph MAE for repeating the last value) but not during closures (3.01 vs 2.89). A failure "
                "analysis shows where the error comes from: on closure readings where traffic is actually slowed, the model "
                "forecasts 5.4 mph too fast, i.e. it expects recovery too early. We hypothesised that neighbouring "
                "sensors that still look normal cause this, and tested the idea with a graph-gating experiment that "
                "cuts the closure sensor off from its neighbours. The hypothesis was <b>not supported</b>: gating did "
                "not reduce the bias, the downstream neighbours we blamed pull the forecast slower rather than "
                "faster, and the bias is largest where neighbours are already slower. A second test supports a "
                "different cause: the model learned how fast slowed traffic usually recovers, and closures are the "
                "exception (in training only 17% of slowed closure readings were back to normal within the hour, "
                "against 83% for everyday congestion). On test closures it predicts +8.5 mph of recovery, close to "
                "that learned rate (+10.2) and far above what happened (+3.6). Fine-tuning with unusual slowdowns "
                "weighted more lowered the closure bias by 1.3 mph, but only by expecting less recovery everywhere. "
                "Adding incident inputs and up-weighting closures in the loss gave no gain at our training budget; "
                "a fixed blend with the last value did.", "abstract")]

    # 1 Introduction
    story += [P("1  Question and approach", "h1"),
              P("Traffic forecasting GNNs are usually compared on average error over a test period. Averages are "
                "dominated by normal traffic, and road closures, where a forecast matters most, are rare. "
                "We therefore asked four questions:"),
              B("<b>Q1</b> How much do recorded closures change traffic, compared with other incidents?"),
              B("<b>Q2</b> How well does a standard traffic GNN (DCRNN) forecast through them, against simple baselines?"),
              B("<b>Q3</b> When, where and in which direction does it fail?"),
              B("<b>Q4</b> Do simple changes help: incident inputs, loss weighting, blending, removing or gating the graph?"),
              P("Our approach is diagnostic rather than leaderboard-driven: every model is scored separately on "
                "normal traffic and on each incident type, against persistence and seasonal baselines, with "
                "signed errors and with confidence intervals that resample whole closure events. Section 4 follows the "
                "investigation in the order we ran it, including the parts that did not work.")]

    # 2 Data
    story += [P("2  Data", "h1"),
              P("<b>Traffic.</b> TraffiDent (Kaggle <i>gpxlcj/xtraffic</i>, PeMS 2023, CC BY-NC 4.0), 5-minute "
                "readings. We use Caltrans District 8 (San Bernardino and Riverside), mainline sensors only: "
                "898 sensors, January to June 2023, speed in mph (0.02% missing). Inputs are one hour of speed and "
                "time of day; the target is the next hour (12 steps)."),
              P("<b>Incidents.</b> CHP incident records, classified from their description into closure, SigAlert, "
                "collision, hazard and other. Each is matched to the nearest mainline sensor on the same freeway and "
                "direction within 0.5 miles, and active from its report for its <i>recorded</i> duration (no default "
                "durations are filled in). Ramp incidents are excluded from scoring. This yields 195 freeway closures, "
                "8,951 collisions, 9,634 hazards, 117 SigAlerts and 1,297 other incidents."),
              P("<b>Road graph.</b> Directed, built from Caltrans postmiles: sensor <i>i</i> links to <i>j</i> if "
                "<i>j</i> is downstream on the same freeway and direction within 10 km; sensors on different "
                "freeways within 0.5 km are linked both ways (interchanges). Weights are a Gaussian kernel of "
                "driving distance, thresholded at 0.1, as in DCRNN."),
              P("<b>Splits.</b> Time-ordered 60/20/20. The test period is 25 May to 30 June 2023 and contains 35 "
                "closures. Input scaling and each sensor's <i>typical speed</i> (median per 5-minute slot of the "
                "week) use the training period only.")]

    # 3 Methods
    story += [P("3  Models, baselines and evaluation", "h1"),
              P("<b>DCRNN</b> (Li et al., 2018), re-implemented in PyTorch: an encoder-decoder of diffusion-"
                "convolutional GRU cells, 2 layers × 64 units, 2 diffusion steps along and against traffic, "
                "372k weights. The road graph enters only through the diffusion matrices, so it can be changed at "
                "inference time without retraining, which the ablation and gating experiments rely on."),
              P("<b>Training budget.</b> All runs fit a 16 GB Mac (Apple GPU): 3 epochs on every 8th training "
                "window, then 2 fine-tuning epochs at learning rate 10<super>-4</super>. The paper's recipe is up "
                "to 100 epochs; our longer 6-epoch run reached 1.79 mph test MAE against 1.85 here, so absolute "
                "numbers are not converged. All comparisons are made at an equal budget."),
              P("<b>Baselines.</b> <i>Last value</i> (every future step equals the last reading); <i>typical "
                "speed</i> for that sensor and time of week; and <i>blend</i>: while an incident is active, the "
                "model's forecast is mixed with last value, with the model's share growing from 1/12 at 5 minutes "
                "to all of it at 60 minutes (fixed in advance, not tuned)."),
              P("<b>Evaluation.</b> MAE over the forecast hour, split by what is happening at the sensor (normal "
                "traffic, or an incident of each type active at that sensor), plus the 95th-percentile error and "
                "the mean signed error (prediction minus truth; positive means the forecast is too fast). With "
                "only 35 test closures, intervals resample closure events rather than readings, and differences "
                "below about 0.05 mph are treated as ties.")]

    # 4 Investigation
    story += [P("4  The investigation", "h1"),
              P("4.1  Starting point and a pivot: METR-LA", "h2"),
              P("We began with the authors' pretrained DCRNN on METR-LA (Los Angeles, 207 sensors), a closure "
                "simulator, and \"events\" defined as unusual slowdowns. A first check showed the model depends "
                "on its graph: replacing the road graph by the identity at inference raised the 15-minute MAE in "
                "normal traffic from 2.53 to 3.58 mph. Feedback on the proposal pointed out the core problem: "
                "METR-LA has no incident labels, so an unusual slowdown is not evidence of a closure (it may be "
                "a crash, an event, weather or a sensor fault). We switched to TraffiDent, which ships recorded "
                "CHP incidents, and rebuilt the pipeline around them. <b>Lesson:</b> settle the ground truth "
                "before the model."),
              P("4.2  A closure-simulation pilot", "h2"),
              P("Our original plan was to simulate closures by editing the model's inputs and graph, and to compare "
                "the simulated slowdowns with real ones. On 69 held-out closures, feeding the closed sensor's real "
                "readings reproduced its own slowdown reasonably (predicted 5.5 vs observed 9.1 mph, correlation "
                "0.89 across closures), but the model predicted essentially <b>no spillover</b> to upstream sensors "
                "(-0.03 mph predicted vs 3.3 mph observed; 9.1 mph on 21 clearly disrupted closures). Cutting the "
                "closed sensor's graph edges changed nothing (0.2 mph). A model that has not learned how a "
                "closure propagates cannot be used to simulate one, so we dropped simulation and measured "
                "forecast error on real closures instead."),
              P("4.3  Q1: how much do closures change traffic?", "h2"),
              P("For 17,062 incidents we compared the speed in the first hour after the report with the same "
                "sensor and time of week one week earlier and later, with 16,933 incident-free controls (Figure 1). "
                "Closures cause the largest drop (3.6 mph, 95% CI 2.0-5.3), about three times a collision. The "
                "average is modest because most closures are at night: only 25% produce a visible slowdown, "
                "against 9% of controls. The effect is real but concentrated in a minority of closures, which "
                "is why later analyses separate disrupted from undisrupted readings.")]
    story.append(figure(FIG / "fig1_impact.png",
                        "Figure 1. Mean speed drop in the first hour after an incident report, against the same "
                        "sensor and time one week earlier and later (bars: 95% CI). Closures slow traffic most.", S))

    story += [P("4.4  Q2: forecast error by incident type", "h2"),
              P("Table 1 and Figure 2 split the test error by what is happening at the sensor. The ranking flips: "
                "DCRNN is the best model in normal traffic but loses to last value during closures and "
                "SigAlerts. The overall test MAE (1.85 vs 2.02 mph for last value) hides this completely.")]
    rows = [["Model", "Normal", "Closure", "Collision", "Hazard", "SigAlert"],
            ["Last value", "2.01", "2.89", "4.21", "3.22", "3.53"],
            ["Typical speed for time of week", "2.76", "7.56", "6.15", "4.58", "8.99"],
            ["DCRNN (3 + 2 epochs)", "1.84", "3.01", "4.09", "3.01", "3.82"],
            ["DCRNN, blended with last value during incidents", "1.84", "2.87", "4.08", "3.04", "3.70"],
            ["DCRNN trained without the graph", "1.88", "2.94", "4.11", "3.10", "3.57"]]
    story.append(KeepTogether([P("<b>Table 1.</b> MAE over the next hour (mph), test period. Readings: 112 million "
                                 "normal, 19,536 closure, 4,848 SigAlert.", "caption"),
                               table(rows, S, [2.6 * inch] + [0.75 * inch] * 5)]))
    story.append(Spacer(1, 8))
    story.append(figure(FIG / "fig2_mae_by_kind.png",
                        "Figure 2. Forecast error by situation. DCRNN wins in normal traffic and mild incidents; "
                        "last value wins during closures and SigAlerts.", S))

    story += [P("4.5  Q4: incident information and loss weighting", "h2"),
              P("We gave the model four extra inputs (which incident type is active at the sensor and how recently "
                "it started, both known at forecast time) and fine-tuned with closure readings weighted 20× in "
                "the loss. Closure MAE fell from 3.08 to 3.02 mph, which looked like a gain. An <b>equal-epochs "
                "control</b>, the plain model fine-tuned for the same 2 extra epochs with no incident information, "
                "reached 3.01. The improvement came from training longer, not from the incident information. "
                "Without that control we would have reported a false positive. The fixed blend with last value was the only "
                "change that clearly helped on closures (2.87 mph), at no cost in normal traffic."),
              P("4.6  Q4: does the graph help? Graph ablation", "h2"),
              P("We trained the same model, with the same budget, with all road edges removed, so each sensor sees only its "
                "own history. The graph helps in normal traffic (1.84 vs 1.88 mph) and mild incidents, but the "
                "model without it is better in severe ones: closures 3.01 → 2.94 and SigAlerts 3.82 → 3.57, "
                "with the SigAlert 95th-percentile error falling from 19.8 to 14.6 mph.")]

    story += [P("4.7  Q3: failure analysis (when, where, which direction)", "h2"),
              P("We split every test reading taken during a closure by four factors: whether traffic is actually "
                "disrupted (at least 5 mph below typical), time since the report, day or night, and horizon. "
                "Signed error was the most informative measure:"),
              B("<b>What:</b> on undisrupted closure readings DCRNN beats last value (1.50 vs 1.65 mph); on disrupted "
                "ones it loses (7.24 vs 6.38) and is biased <b>+5.4 mph too fast</b> (last value +2.9, no-graph "
                "model +4.5). It forecasts recovery too early."),
              B("<b>When:</b> the bias is largest in the first 30 minutes after the report (+4.7 mph) and the gap to last "
                "value is widest 30-60 minutes after it (4.92 vs 4.04 mph, Figure 3). It is larger in daytime "
                "(+1.9 vs +0.6 at night)."),
              B("<b>Horizon:</b> the bias grows with lead time, from +0.2 mph at 5 minutes to +1.9 at 60 minutes."),
              P("Together with the ablation, this suggested a mechanism: neighbouring sensors that still look normal "
                "pull the closure sensor's forecast toward recovery. Feedback on our checkpoint called the analysis thorough but "
                "noted that this explanation was still untested. Section 4.8 tests it.")]
    story.append(figure(FIG / "fig3_closure_phase.png",
                        "Figure 3. Error during closures by time since the report (all closure readings). The models "
                        "are too fast early in a closure; DCRNN with the graph is the most biased.", S))

    gd = gg["closure"]
    story += [P("4.8  Testing the explanation: graph gating", "h2"),
              P("<b>Design.</b> We added a per-sensor gate to DCRNN: each diffusion matrix <i>S</i> becomes "
                "<i>g·S + (1-g)·I</i> for that sensor. At <i>g</i> = 1 the model is unchanged (checked by a unit "
                "test); at <i>g</i> = 0 the sensor hears its own readings in place of its neighbours', so no information "
                "from other sensors reaches it, while its inputs stay in their usual range (zeroing the messages "
                "instead would feed the model values it never saw). The gate closes at every sensor with an "
                "incident active at forecast time, which is known information. Upstream and downstream messages use separate "
                "matrices, so each direction can be gated alone. We wrote down what the explanation predicts "
                "<i>before</i> looking at the results:"),
              B("P1. Closing the gate removes most of the fast bias."),
              B("P2. Gating <i>downstream</i> neighbours, which sit on the emptier road past the closure, removes more than gating "
                "upstream ones, where the queue builds."),
              B("P3. The graph model's extra bias over the no-graph model grows with how much faster the "
                "neighbours look."),
              B("P4. The gain does not disappear when the model is fine-tuned with the gate (so it is not an "
                "artefact of an unfamiliar input)."),
              P("A placebo, gating every sensor everywhere, checks whether the graph is simply unhelpful in general.")]
    rows = [["Model (disrupted closure readings)", "MAE", "Bias (95% CI)", "MAE change vs graph (95% CI)"]]
    order = [("graph", "DCRNN (graph), reference"), ("graph, gate 0", "Gate both directions"),
             ("gate-trained, gate 0", "Gate both, fine-tuned with the gate"),
             ("graph, gate upstream only", "Gate upstream only"), ("graph, gate downstream only", "Gate downstream only"),
             ("graph, gate 0.5", "Half gate (g = 0.5)"), ("graph, gate every sensor (placebo)", "Gate every sensor (placebo)"),
             ("no graph", "DCRNN trained without the graph"), ("last value", "Last value")]
    for k, name in order:
        s = gd[k]["disrupted"]
        ci = f"{s['bias']:+.2f} ({s['bias_ci'][0]:+.1f} to {s['bias_ci'][1]:+.1f})"
        dm = "" if k == "graph" else f"{s['d_mae']:+.2f} ({s['d_mae_ci'][0]:+.2f} to {s['d_mae_ci'][1]:+.2f})"
        rows.append([name, f"{s['mae']:.2f}", ci, dm])
    n = gd["graph"]["disrupted"]
    story.append(KeepTogether([P(f"<b>Table 2.</b> Graph gating on the {n['readings']:,} disrupted closure "
                                 f"readings from {gd['graph']['all']['events']} test closures (mph). Intervals "
                                 "resample closures.", "caption"),
                               table(rows, S, [2.45 * inch, 0.55 * inch, 1.55 * inch, 1.85 * inch])]))
    story.append(Spacer(1, 8))
    story.append(figure(FIG / "fig4_gating.png",
                        "Figure 4. Bias on disrupted closure readings under each gate (95% CI over closures). "
                        "Dashed line: the ungated graph model. Gating does not move the bias toward zero.", S))
    e = gg["extra_bias_vs_gap_down"]
    story += [P("<b>Result: the explanation is not supported.</b>"),
              B(f"<b>P1 fails.</b> Closing the gate raised the bias slightly ({gd['graph']['disrupted']['bias']:+.2f} → "
                f"{gd['graph, gate 0']['disrupted']['bias']:+.2f} mph) and the MAE (+0.51 mph, CI -0.07 to +1.21)."),
              B(f"<b>P2 is reversed.</b> Gating downstream neighbours made the bias <i>worse</i> "
                f"({gd['graph, gate downstream only']['disrupted']['bias']:+.2f} mph); it is the only change whose "
                "MAE interval excludes zero (+0.73, CI +0.27 to +1.45). Downstream messages pull the forecast "
                "slower, not faster. Upstream gating had no clear effect."),
              B(f"<b>P3 fails.</b> The graph model's extra bias over the no-graph model does not grow with the "
                f"downstream speed gap (correlation {e['corr']:+.2f}, {e['readings']:,} readings). The bias is "
                "largest (+11 mph) where neighbours are already <i>slower</i> than the closure sensor (Figure 5), "
                "when the slowdown is still growing."),
              B(f"<b>P4 fails.</b> Fine-tuning with the gate did not rescue it "
                f"({gd['gate-trained, gate 0']['disrupted']['bias']:+.2f} mph)."),
              B("<b>Placebo.</b> Gating every sensor raised normal-traffic MAE from 1.84 to 1.94 mph: the graph is "
                "useful outside incidents."),
              P("The half gate (<i>g</i> = 0.5) lowered the bias to +4.53 mph, but the full gate raised it, so there "
                "is no consistent trend between gate strength and bias. With nine conditions and wide intervals we "
                "treat it as noise rather than a finding.")]
    story.append(figure(FIG / "fig5_gap.png",
                        "Figure 5. Bias on disrupted closure readings by how much faster the downstream neighbours "
                        "were than the closure sensor when the forecast was made. Under the explanation the graph "
                        "model's line would rise above the others to the right; it does not.", S))
    story += [P("<b>What explains the bias then?</b> Every learned model is too fast during disrupted closures, and "
                "last value is too, to a lesser degree. The no-graph model's lower bias comes from training "
                "without the graph, not from neighbour messages at forecast time, since removing those "
                "messages from the trained graph model does not reproduce it. The pattern fits a "
                "<b>learned expectation that slowdowns recover</b>: in training, most slow readings are recurring "
                "congestion that clears within the hour, and the model applies that expectation to closures, which "
                "do not clear on schedule. Section 4.9 tests this second explanation."),
              P("<b>One example is not evidence.</b> Figure 6 shows the test closure with the deepest slowdown. "
                "There, the explanation looks convincing: the graph model forecasts a fast recovery and the gated model "
                "stays slower. Had we stopped at this example we would have confirmed a mechanism that the "
                "35-closure test rejects.")]
    story.append(figure(RES / "demo_closure.png",
                        "Figure 6. A single test closure (I-15 N at Temescal Canyon Rd, 23 June 2023), forecast "
                        "15 minutes after the report. On this case gating helps (next-hour MAE 22.5 → 14.3 mph); "
                        "across all closures it does not (Table 2). Last value is best here (2.8 mph).", S, width=6.0))

    rp = json.loads((RES / "recovery_prior.json").read_text())
    p1, rg, cdis = rp["p1_training"], rp["groups"], rp["closure_disrupted"]
    sw = cdis["dcrnn_check_slowdown_ft"]["vs dcrnn_check_plain_ft"]
    story += [P("4.9  Testing the second explanation: a learned expectation of recovery", "h2"),
              P("<b>Design.</b> A <i>forecast origin</i> is a moment when a sensor is slowed (below 50 mph). We "
                "split origins into three groups using only what is known at that moment: <i>usual congestion</i> "
                "(within 5 mph of the typical speed for that time of week, no incident), <i>unusual slowdown</i> "
                "(10+ mph below typical, no recorded incident) and <i>closure</i> (10+ mph below typical, closure "
                "active). For each origin we compare three numbers for the next 60 minutes: how much speed actually "
                "recovered, how much the model predicted, and the <i>base rate</i>, which is the average recovery in "
                "the training period of all readings at the same speed and hour of day. The base rate is what "
                "\"slow now, faster soon\" learned from speed and time alone would predict. Predictions, written "
                "before running the test:"),
              B("P1. In training, everyday congestion recovers as the base rate says; closures recover much less."),
              B("P2. On test closures the model's predicted recovery is close to the base rate and well above the actual."),
              B("P3. The too-fast bias is near zero in usual congestion and large in unusual slowdowns and closures."),
              B("P4. Fine-tuning with unusual slowdowns weighted 5× in the loss lowers the closure bias, against the "
                "same fine-tuning without weights. Only 0.15% of those readings are closures, so a gain would come "
                "from the general recovery expectation, not from learning closures.")]
    rows = [["Slowed origins", "Training: recovered within 1 h", "Training: actual / base rate",
             "Test: actual / base rate", "Test: DCRNN predicted", "Test: DCRNN bias (95% CI)"]]
    for g, name in [("usual congestion", "Usual congestion"), ("unusual slowdown", "Unusual slowdown"),
                    ("closure", "Closure")]:
        t, lv, m = p1[g], rg["last value"][g], rg["dcrnn_check_plain_ft"][g]
        unit = "closures" if g == "closure" else "sensor-days"
        rows.append([f"{name} ({lv['origins']:,} test origins, {lv['clusters']:,} {unit})",
                     f"{t['recovered_share']:.0%}",
                     f"{t['actual_recovery_60']:+.1f} / {t['base_rate_recovery_60']:+.1f}",
                     f"{lv['actual_recovery_60']:+.1f} / {lv['base_rate_recovery_60']:+.1f}",
                     f"{m['pred_recovery_60']:+.1f}",
                     f"{m['bias']:+.2f} ({m['bias_ci'][0]:+.2f} to {m['bias_ci'][1]:+.2f})"])
    story.append(KeepTogether([P("<b>Table 3.</b> Speed gain over the next 60 minutes from slowed traffic (mph). "
                                 "Bias is the mean signed error over the forecast hour (> 0 = too fast).", "caption"),
                               table(rows, S, [1.75 * inch, 0.8 * inch, 0.95 * inch, 0.95 * inch, 0.8 * inch,
                                               1.15 * inch])]))
    story.append(Spacer(1, 8))
    story.append(figure(FIG / "fig7_recovery.png",
                        "Figure 7. Predicted and actual speed gain over the next hour from slowed traffic, test "
                        "period. For closures, the graph model's prediction sits near the training base rate and far "
                        "from what happened.", S))
    story += [P("<b>Result: supported, with a refinement.</b>"),
              B(f"<b>P1 holds for closures.</b> In training, {p1['usual congestion']['recovered_share']:.0%} of slowed "
                f"everyday-congestion readings were back to normal within the hour, but only "
                f"{p1['closure']['recovered_share']:.0%} of slowed closure readings were. Closures gained "
                f"{p1['closure']['actual_recovery_60']:+.1f} mph against a base rate of "
                f"{p1['closure']['base_rate_recovery_60']:+.1f} mph for traffic at the same speed and hour. "
                f"Everyday congestion matched the base rate only roughly ({p1['usual congestion']['actual_recovery_60']:+.1f} "
                f"vs {p1['usual congestion']['base_rate_recovery_60']:+.1f} mph)."),
              B(f"<b>P2 holds.</b> On test closures DCRNN predicted {rg['dcrnn_check_plain_ft']['closure']['pred_recovery_60']:+.1f} mph, "
                f"close to the base rate ({rg['last value']['closure']['base_rate_recovery_60']:+.1f}) and far above "
                f"the actual {rg['last value']['closure']['actual_recovery_60']:+.1f}. The no-graph model predicted "
                f"{rg['dcrnn_check_nograph_ft']['closure']['pred_recovery_60']:+.1f}: it learned a weaker recovery "
                "expectation in every group (Figure 7). This explains its lower closure bias in Sections 4.6 and 4.8, "
                "which neighbour messages could not."),
              B(f"<b>P3 only partly holds, and that sharpens the conclusion.</b> The bias is small in everyday "
                f"congestion ({rg['dcrnn_check_plain_ft']['usual congestion']['bias']:+.2f} mph), as predicted. But in "
                f"unusual slowdowns without an incident the model is slightly <i>too slow</i> "
                f"({rg['dcrnn_check_plain_ft']['unusual slowdown']['bias']:+.2f} mph), because those recover even "
                "faster than the base rate. The model is not too fast on unusual slowdowns in general. It is too fast "
                "on closures, the one kind of slowdown that does not recover the way traffic at that speed usually "
                "does, and it cannot tell them apart from their speed history alone."),
              B(f"<b>P4 holds, at a cost.</b> With unusual slowdowns weighted 5×, the bias on disrupted closure "
                f"readings fell from {cdis['dcrnn_check_plain_ft']['bias']:+.2f} to "
                f"{cdis['dcrnn_check_slowdown_ft']['bias']:+.2f} mph (change {sw['d_bias']:+.2f}, CI "
                f"{sw['d_bias_ci'][0]:+.2f} to {sw['d_bias_ci'][1]:+.2f}) and the MAE by {-sw['d_mae']:.2f} mph "
                f"(CI {sw['d_mae_ci'][0]:+.2f} to {sw['d_mae_ci'][1]:+.2f}). Both intervals exclude zero, so "
                "lowering the learned recovery expectation lowers the closure bias. But the model "
                "now expects less recovery <i>everywhere</i>: its bias in everyday congestion went from "
                f"{rg['dcrnn_check_plain_ft']['usual congestion']['bias']:+.2f} to "
                f"{rg['dcrnn_check_slowdown_ft']['usual congestion']['bias']:+.2f} mph, normal-traffic MAE rose from "
                "1.84 to 1.86, and on disrupted closures it still loses to last value "
                f"({cdis['dcrnn_check_slowdown_ft']['mae']:.2f} vs {cdis['last value']['mae']:.2f} mph).")]
    story += [P("Caveats: the closure group in Table 3 has 11 closures (P4 uses all 35), and checkpoint selection kept "
                "the weighted model after 1 of its 2 fine-tuning epochs while the control used both. The conclusion is "
                "that the closure bias comes from a recovery expectation that is right for almost all slowed traffic "
                "and wrong for closures. Fixing it properly needs the model to <i>recognise</i> a closure, not just "
                "to expect less recovery everywhere. Our incident inputs (Section 4.5) did not achieve that, plausibly "
                f"because training had only {p1['closure']['origins']} slowed closure moments against "
                f"{p1['usual congestion']['origins'] + p1['unusual slowdown']['origins']:,} other slowed ones.")]

    # 5 What worked
    story += [P("5  What worked and what did not", "h1")]
    rows = [["Attempt", "Outcome", "Verdict"],
            ["METR-LA with unusual slowdowns as events", "No incident labels, so events were not valid ground truth", "Replaced by TraffiDent"],
            ["Closure simulation (edit inputs / graph)", "Closed sensor reproduced, spillover not (≈0 vs 3.3 mph)", "Dropped"],
            ["Per-incident evaluation vs baselines", "Revealed that DCRNN loses to last value on closures", "Worked; core method"],
            ["Incident inputs + 20× closure weighting", "3.02 vs 3.01 for equal-epoch control", "No gain"],
            ["Blend with last value during incidents", "Closure MAE 3.01 → 2.87, normal unchanged", "Worked"],
            ["Removing the graph (ablation)", "Better on closures/SigAlerts, worse on normal", "Trade-off"],
            ["Signed-error failure analysis", "Found a +5.4 mph too-fast bias on disrupted readings", "Worked; key finding"],
            ["Graph gating (inference and fine-tuned)", "Bias not reduced; downstream gating worse", "Refuted the explanation"],
            ["Recovery analysis (predicted vs actual vs base rate)", "Closures recover far less than the model's learned rate", "Supported the 2nd explanation"],
            ["Fine-tune with unusual slowdowns weighted 5×", "Closure bias −1.3 mph, but less recovery expected everywhere", "Partial fix, with a cost"]]
    story.append(table(rows, S, [2.3 * inch, 2.75 * inch, 1.35 * inch]))
    story.append(Spacer(1, 6))

    # 6 Insights
    story += [P("6  Key insights and what we learned", "h1"),
              B("<b>Average error hides the failure that matters.</b> On overall MAE DCRNN is the best model; during "
                "closures, repeating the last reading beats it. Evaluation has to be split by situation and compared "
                "against persistence."),
              B("<b>The failure has a direction.</b> Reporting signed error, not only MAE, turned \"worse on closures\" "
                "into a specific behaviour: forecasting recovery too early, most strongly in the first hour."),
              B("<b>The failure comes from what the model learned, not what it hears from neighbours.</b> It "
                "learned how fast slowed traffic usually recovers, which is right for almost all slowdowns and wrong "
                "for closures. The graph model learned a stronger version of that expectation, which is why "
                "removing the graph helped on closures. Cutting off neighbour messages did not."),
              B("<b>A plausible mechanism can be wrong.</b> Our first explanation fitted the ablation and a vivid example. Writing down "
                "predictions that could fail (direction, gate strength, a placebo, a fine-tuned control) is what "
                "showed it was wrong. The same approach then supported the second explanation, and the prediction "
                "that failed (P3) told us more precisely where the problem lies."),
              B("<b>Controls decide the conclusion.</b> The equal-epochs control removed a false gain from incident "
                "inputs, the no-incident controls calibrated the impact analysis, and resampling whole closures kept "
                "35 events from looking like 5,000 independent readings."),
              B("<b>At a small budget, simple fixes win.</b> A fixed, untuned blend with last value beat every "
                "learned change we tried on closures.")]

    # 7 Limitations
    story += [P("7  Limitations and next steps", "h1"),
              B("<b>Training budget.</b> Models were trained for 3 + 2 epochs on 1/8 of the windows; conclusions are "
                "comparisons at equal budget and could change with fully trained models."),
              B("<b>Few closures.</b> 35 test closures give wide intervals; some bins in Figure 5 rest on 5 closures. "
                "More districts and years of TraffiDent would multiply the sample."),
              B("<b>Label noise.</b> CHP report times, recorded durations and sensor matching (within 0.5 miles) are "
                "imperfect; 75% of closures cause no visible slowdown, which is why we split by disruption."),
              B("<b>Next step.</b> The model needs to recognise a closure, not just expect less recovery everywhere. "
                "Options: pool closures from more districts and years so incident inputs have enough examples to "
                "learn from; give the decoder an explicit \"closure regime\" that switches off the learned recovery; "
                "or use the closure's recorded or estimated duration as an input."),
              B("<b>Spillover.</b> The model did not learn how closures propagate (Section 4.2). Learning it likely "
                "needs many more closures or physics-informed structure.")]

    # 8 Reproducibility
    story += [P("8  Code and reproducibility", "h1"),
              P("All code is in the submitted repository; <font face='Courier'>README.md</font> lists setup and "
                "every command. Main entry points:")]
    for line in ["python prepare_data.py          # data, graph, incident matching",
                 "python train.py --epochs 3 --stride 8 --batch-size 16 --out checkpoints/dcrnn_check_plain.pt",
                 "python finetune.py --base checkpoints/dcrnn_check_plain.pt --epochs 2 --lr 1e-4 ...",
                 "python incident_impact.py       # Q1, Figure 1",
                 "python incident_eval.py checkpoints/*.pt --blend   # Q2, Table 1",
                 "python closure_errors.py <checkpoints>            # failure analysis, Figure 3",
                 "python graph_gating.py --gated checkpoints/dcrnn_check_gated_ft.pt   # Table 2, Figures 4-5",
                 "python finetune.py ... --slowdown-weight 5 --out checkpoints/dcrnn_check_slowdown_ft.pt",
                 "python recovery_prior.py <plain_ft> <nograph_ft> <slowdown_ft>   # Table 3, Figure 7",
                 "python demo.py                  # one closure, Figure 6",
                 "python -m pytest tests          # graph direction, isolation, gate, incident parsing"]:
        story.append(Paragraph(line.replace(" ", "&nbsp;"), S["code"]))
    story += [Spacer(1, 6),
              P("<b>References.</b> Y. Li, R. Yu, C. Shahabi, Y. Liu. Diffusion Convolutional Recurrent Neural "
                "Network: Data-Driven Traffic Forecasting. ICLR 2018. · TraffiDent / XTraffic dataset, Kaggle "
                "<i>gpxlcj/xtraffic</i> (CC BY-NC 4.0), built from Caltrans PeMS and CHP incident records.")]

    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.HexColor(LIGHT_GREY))
        canvas.drawRightString(letter[0] - 0.8 * inch, 0.5 * inch, f"{doc.page}")
        canvas.drawString(0.8 * inch, 0.5 * inch, f"Traffic GNNs under road closures · {GROUP}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.8 * inch,
                            title="Where Traffic GNNs Fail During Road Closures, and Why", author=GROUP)
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    if not (RES / "demo_closure.png").exists():
        sys.exit("run `python demo.py` first (Figure 6)")
    fig_impact(); fig_mae(); fig_phase(); fig_gating(); fig_gap(); fig_recovery()
    build()
