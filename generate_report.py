"""
Generate all figures and updated report for Graph WaveNet contextual experiments.

Outputs:
  - figures/performance_comparison_bar.png
  - figures/horizon_line_plots.png
  - figures/convergence_curves.png
  - figures/architecture_diagram.png
  - REPORT_UPDATED.pdf
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

FIGURE_DIR = 'figures'
GARAGE_DIR = 'garage'
os.makedirs(FIGURE_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Parse checkpoint filenames for convergence curves
# ------------------------------------------------------------------
def parse_checkpoints(prefix):
    """Extract (epoch, valid_loss) from checkpoint filenames."""
    pattern = re.compile(rf'{re.escape(prefix)}_epoch_(\d+)_(\d+\.?\d*)\.pth')
    points = []
    if not os.path.isdir(GARAGE_DIR):
        return np.array([]), np.array([])
    for fname in os.listdir(GARAGE_DIR):
        m = pattern.match(fname)
        if m:
            epoch = int(m.group(1))
            loss = float(m.group(2))
            points.append((epoch, loss))
    points.sort()
    if not points:
        return np.array([]), np.array([])
    epochs, losses = zip(*points)
    return np.array(epochs), np.array(losses)

epoch_base, loss_base = parse_checkpoints('metr')
epoch_weather, loss_weather = parse_checkpoints('metr_weather')
epoch_contextual, loss_contextual = parse_checkpoints('metr_contextual')

# ------------------------------------------------------------------
# 2. Performance comparison bar chart (7 report models + 3 runs)
# ------------------------------------------------------------------
models = ['LSTM', 'GRU', 'TCN', 'DCRNN', 'STGCN', 'Graph\nWaveNet', 'GMAN',
          'GWN\nBaseline', 'GWN\n+Weather', 'GWN\n+Contextual']
mae_60min = [4.37, 4.11, 3.96, 3.60, 4.59, 3.53, 3.04,
             3.54, 3.58, 3.53]
colors_bar = ['#ff9999']*3 + ['#66b3ff']*4 + ['#2ca02c', '#ff7f0e', '#9467bd']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(models, mae_60min, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_ylabel('MAE @ 60 min (mph)', fontsize=12)
ax.set_title('Performance Comparison: 60-Minute Forecasting Horizon on METR-LA', fontsize=14, fontweight='bold')
ax.set_ylim(0, 5.0)
ax.axhline(y=3.53, color='red', linestyle='--', alpha=0.5, label='Graph WaveNet (report)')

# Add value labels on bars
for bar, val in zip(bars, mae_60min):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#ff9999', label='Temporal Baselines'),
                   Patch(facecolor='#66b3ff', label='Graph-Based Models (Report)'),
                   Patch(facecolor='#2ca02c', label='Our Baseline'),
                   Patch(facecolor='#ff7f0e', label='Our +Weather'),
                   Patch(facecolor='#9467bd', label='Our +Contextual')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'performance_comparison_bar.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved performance_comparison_bar.png")

# ------------------------------------------------------------------
# 3. MAE/MAPE vs horizon line plots
# ------------------------------------------------------------------
horizons = [15, 30, 60]

# Report model 60-min values (only have 60min for most; use scaling for 15/30)
# Our actual results from EXPERIMENTS.md
our_base_mae = [2.72, 3.10, 3.54]
our_base_mape = [6.97, 8.53, 10.19]
our_weather_mae = [2.72, 3.10, 3.58]
our_weather_mape = [7.14, 8.57, 10.39]
our_contextual_mae = [2.70, 3.08, 3.53]
our_contextual_mape = [7.15, 8.53, 10.06]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(horizons, our_base_mae, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='Baseline')
ax1.plot(horizons, our_weather_mae, 's-', color='#ff7f0e', linewidth=2, markersize=8, label='+Weather')
ax1.plot(horizons, our_contextual_mae, '^-', color='#9467bd', linewidth=2, markersize=8, label='+Contextual')
ax1.set_xlabel('Forecasting Horizon (minutes)', fontsize=12)
ax1.set_ylabel('MAE (mph)', fontsize=12)
ax1.set_title('MAE vs Forecasting Horizon', fontsize=13, fontweight='bold')
ax1.set_xticks(horizons)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

ax2.plot(horizons, our_base_mape, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='Baseline')
ax2.plot(horizons, our_weather_mape, 's-', color='#ff7f0e', linewidth=2, markersize=8, label='+Weather')
ax2.plot(horizons, our_contextual_mape, '^-', color='#9467bd', linewidth=2, markersize=8, label='+Contextual')
ax2.set_xlabel('Forecasting Horizon (minutes)', fontsize=12)
ax2.set_ylabel('MAPE (%)', fontsize=12)
ax2.set_title('MAPE vs Forecasting Horizon', fontsize=13, fontweight='bold')
ax2.set_xticks(horizons)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'horizon_line_plots.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved horizon_line_plots.png")

# ------------------------------------------------------------------
# 4. Convergence curves
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Use precise best valid losses from EXPERIMENTS.md (filenames round to 2 decimals)
best_base = 2.7418
best_weather = 2.7425
best_contextual = 2.7411

if len(epoch_base) > 0:
    ax.plot(epoch_base, loss_base, color='#2ca02c', linewidth=1.5, alpha=0.8, label=f'Baseline (best {best_base:.4f})')
if len(epoch_weather) > 0:
    ax.plot(epoch_weather, loss_weather, color='#ff7f0e', linewidth=1.5, alpha=0.8, label=f'+Weather (best {best_weather:.4f})')
if len(epoch_contextual) > 0:
    ax.plot(epoch_contextual, loss_contextual, color='#9467bd', linewidth=1.5, alpha=0.8, label=f'+Contextual (best {best_contextual:.4f})')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Loss (MAE)', fontsize=12)
ax.set_title('Training Convergence: Validation Loss Across Epochs', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 100)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'convergence_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved convergence_curves.png")

# ------------------------------------------------------------------
# 5. Architecture diagram for 3-input model
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Proposed 3-Stream Contextual Graph WaveNet Architecture', fontsize=16, fontweight='bold', pad=20)

def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Input streams
draw_box(ax, 0.5, 7.5, 2.5, 1.2, 'Traffic Stream\n(speed + time-of-day)', '#e6f3ff', 9)
draw_box(ax, 0.5, 5.5, 2.5, 1.2, 'Weather Stream\n(temp + precip + humidity)', '#fff2e6', 9)
draw_box(ax, 0.5, 3.5, 2.5, 1.2, 'Road Stream\n(type + lanes)', '#e6ffe6', 9)

# Concatenate
draw_box(ax, 4.0, 5.0, 2.0, 1.5, 'Feature\nConcatenation\nin_dim = 9', '#f0f0f0', 10)
draw_arrow(ax, 3.0, 8.1, 4.0, 5.75)
draw_arrow(ax, 3.0, 6.1, 4.0, 5.75)
draw_arrow(ax, 3.0, 4.1, 4.0, 5.75)

# Graph WaveNet Core
draw_box(ax, 6.5, 5.0, 2.5, 1.5, 'Graph WaveNet Core\n(Adaptive Graph Conv +\nDilated Temporal Conv)', '#d9d9d9', 10)
draw_arrow(ax, 6.0, 5.75, 6.5, 5.75)

# Output
draw_box(ax, 9.5, 5.0, 2.0, 1.5, 'Output\n12-step forecast\n(207 sensors)', '#ffcccc', 10)
draw_arrow(ax, 9.0, 5.75, 9.5, 5.75)

# Key features text
ax.text(6.0, 2.0, 
        'Key Design:\n'
        '  - Streams concatenated at input layer (no core changes)\n'
        '  - Self-adaptive adjacency matrix learns hidden spatial correlations\n'
        '  - Modular: weather/road features added without modifying graph convolutions',
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'architecture_diagram.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved architecture_diagram.png")

# ------------------------------------------------------------------
# 6. Generate updated REPORT.pdf
# ------------------------------------------------------------------
pdf_path = 'REPORT_UPDATED.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                        rightMargin=72, leftMargin=72,
                        topMargin=72, bottomMargin=18)
styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=18,
    textColor=colors.HexColor('#1a1a2e'),
    spaceAfter=12,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading2_style = ParagraphStyle(
    'CustomHeading2',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=colors.HexColor('#16213e'),
    spaceAfter=10,
    spaceBefore=12,
    fontName='Helvetica-Bold'
)

body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['BodyText'],
    fontSize=10,
    leading=14,
    alignment=TA_JUSTIFY
)

caption_style = ParagraphStyle(
    'Caption',
    parent=styles['Italic'],
    fontSize=9,
    alignment=TA_CENTER,
    textColor=colors.grey
)

story = []

# Title
story.append(Paragraph("A Comparative Analysis of Spatiotemporal Deep Learning Models<br/>"
                       "for Short-Term Traffic Forecasting", title_style))
story.append(Paragraph("<b>Updated Report — Contextual Enhancement Results</b>", body_style))
story.append(Spacer(1, 0.2*inch))
story.append(Paragraph("Sairam Jammu, Brent Stanfield, M.Lalitha Akshaya<br/>"
                       "Dept. of Computer Science, Kent State University, Kent, OH, USA", body_style))
story.append(Spacer(1, 0.3*inch))

# Abstract update
story.append(Paragraph("<b>Abstract</b>", heading2_style))
abstract_text = (
    "This report updates our prior comparative analysis with empirical validation of a proposed "
    "contextual hybrid augmentation framework for Graph WaveNet. We extend the baseline model "
    "with two additional input streams: (1) weather features (temperature, precipitation, humidity) "
    "retrieved from the Open-Meteo API, and (2) static road features (road classification and lane count) "
    "collected from OpenStreetMap. Three model variants are trained and evaluated on the METR-LA "
    "benchmark: a bug-fixed Graph WaveNet baseline, a weather-enhanced model (in_dim = 5), and a "
    "full contextual model (in_dim = 9). Results show that contextual features achieve the best validation "
    "loss (2.7411) but offer only marginal improvement over the baseline (2.7418), likely due to the "
    "92%% freeway sensor bias in METR-LA and near-zero linear correlation between weather and traffic speed."
)
story.append(Paragraph(abstract_text, body_style))
story.append(Spacer(1, 0.2*inch))

# Section I: Overview
story.append(Paragraph("I. EXPERIMENTAL OVERVIEW", heading2_style))
story.append(Paragraph(
    "All experiments use the METR-LA dataset (207 sensors, Mar–Jun 2012, 5-minute intervals). "
    "Each model is trained for 100 epochs with batch size 64, learning rate 0.001, dropout 0.3, "
    "and weight decay 0.0001. The Graph WaveNet core (adaptive adjacency, dilated causal convolutions) "
    "remains unchanged across all three runs; only the input dimension varies.", body_style))
story.append(Spacer(1, 0.15*inch))

# Section II: Results Table
story.append(Paragraph("II. RESULTS SUMMARY", heading2_style))

table_data = [
    ['Model', 'Epochs', 'MAE@15min', 'MAE@30min', 'MAE@60min', 'Avg MAE', 'Avg MAPE', 'Best Valid Loss'],
    ['Graph WaveNet (baseline)', '100', '2.72', '3.10', '3.54', '3.06', '8.35%', '2.7418'],
    ['+ Weather', '100', '2.72', '3.10', '3.58', '3.07', '8.50%', '2.7425'],
    ['+ Weather + Road (Contextual)', '100', '2.70', '3.08', '3.53', '3.04', '8.39%', '2.7411'],
]

t = Table(table_data, colWidths=[2.4*inch, 0.7*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.1*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f4f8')]),
]))
story.append(t)
story.append(Spacer(1, 0.1*inch))
story.append(Paragraph("Table 1. Performance of three Graph WaveNet variants on METR-LA test set.", caption_style))
story.append(Spacer(1, 0.2*inch))

# Figures
story.append(Paragraph("III. PERFORMANCE COMPARISON", heading2_style))
story.append(Paragraph(
    "Figure 1 places our three model variants in the context of the seven models from the original "
    "comparative analysis. The contextual model matches the original Graph WaveNet report result (3.53) "
    "at 60 minutes while offering the best overall validation loss.", body_style))
story.append(Spacer(1, 0.1*inch))
story.append(Image(os.path.join(FIGURE_DIR, 'performance_comparison_bar.png'), width=6.5*inch, height=3.25*inch))
story.append(Paragraph("Figure 1. 60-minute MAE comparison across all ten models.", caption_style))
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("IV. HORIZON ANALYSIS", heading2_style))
story.append(Paragraph(
    "Figure 2 shows MAE and MAPE across the three standard forecasting horizons. "
    "The contextual model achieves modest improvements at 15 and 30 minutes but converges to "
    "the same 60-minute performance as the baseline, suggesting diminishing returns from "
    "contextual features at longer horizons.", body_style))
story.append(Spacer(1, 0.1*inch))
story.append(Image(os.path.join(FIGURE_DIR, 'horizon_line_plots.png'), width=6.5*inch, height=2.7*inch))
story.append(Paragraph("Figure 2. MAE and MAPE vs forecasting horizon for our three model variants.", caption_style))
story.append(Spacer(1, 0.2*inch))

story.append(PageBreak())

story.append(Paragraph("V. TRAINING CONVERGENCE", heading2_style))
story.append(Paragraph(
    "Figure 3 plots validation loss per epoch, reconstructed from saved model checkpoints. "
    "All three runs converge within the first 20–30 epochs and fluctuate around a similar loss floor, "
    "confirming that contextual features do not destabilize training.", body_style))
story.append(Spacer(1, 0.1*inch))
story.append(Image(os.path.join(FIGURE_DIR, 'convergence_curves.png'), width=6.0*inch, height=3.6*inch))
story.append(Paragraph("Figure 3. Validation loss convergence over 100 epochs.", caption_style))
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("VI. ARCHITECTURE", heading2_style))
story.append(Paragraph(
    "Figure 4 illustrates the proposed 3-stream input architecture. Traffic, weather, and road features "
    "are concatenated at the input layer and fed into the standard Graph WaveNet core. "
    "No graph convolution or temporal convolution modules are modified, preserving the "
    "modularity and computational efficiency of the original model.", body_style))
story.append(Spacer(1, 0.1*inch))
story.append(Image(os.path.join(FIGURE_DIR, 'architecture_diagram.png'), width=6.5*inch, height=4.3*inch))
story.append(Paragraph("Figure 4. 3-stream contextual Graph WaveNet architecture.", caption_style))
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("VII. DISCUSSION", heading2_style))
discussion_text = (
    "<b>Marginal Improvement.</b> The contextual model achieves the best validation loss (2.7411) "
    "but the improvement over baseline (2.7418) is only 0.0007 — well within training noise. "
    "This suggests that for the METR-LA freeway-dominant network, historical speed patterns "
    "are already highly predictive and weather/road features add limited information.<br/><br/>"
    "<b>Weather Impact.</b> Correlation analysis between weather variables (temperature, precipitation, humidity) "
    "and traffic speed reveals near-zero linear relationships over the full dataset. Rain events are rare "
    "in Los Angeles during the study period (Mar–Jun 2012), limiting the model's opportunity to learn "
    "weather-induced traffic deviations.<br/><br/>"
    "<b>Road Feature Bias.</b> OpenStreetMap data shows that 191 of 207 sensors (92%%) are located on "
    "freeways (motorway/motorway_link). With only 4 arterial and 12 local sensors, the road type feature "
    "has low variance, making it difficult for the model to learn distinct congestion propagation patterns "
    "by road classification.<br/><br/>"
    "<b>Conclusion.</b> While the proposed contextual augmentation framework is technically sound and modular, "
    "its benefits on METR-LA are constrained by dataset characteristics. Future work should evaluate "
    "this architecture on datasets with higher weather variability (e.g., rainy climates) and more diverse "
    "road networks (e.g., mixed urban arterials and local streets). Alternative fusion strategies — such as "
    "attention-based gating or GNN-level feature interaction — may also unlock greater improvements."
)
story.append(Paragraph(discussion_text, body_style))
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("VIII. REFERENCES", heading2_style))
refs = [
    "[1] Y. Li et al., 'Diffusion Convolutional Recurrent Neural Network,' ICLR, 2018.",
    "[2] B. Yu et al., 'Spatio-Temporal Graph Convolutional Networks,' IJCAI, 2018.",
    "[3] Z. Wu et al., 'Graph WaveNet for Deep Spatial-Temporal Graph Modeling,' IJCAI, 2019.",
    "[4] C. Zheng et al., 'GMAN: A Graph Multi-Attention Network for Traffic Prediction,' AAAI, 2020.",
    "[5] Open-Meteo API, https://open-meteo.com, accessed Apr 2026.",
    "[6] OpenStreetMap contributors, https://www.openstreetmap.org, accessed Apr 2026.",
]
for r in refs:
    story.append(Paragraph(r, body_style))
    story.append(Spacer(1, 0.05*inch))

doc.build(story)
print(f"Saved {pdf_path}")
print("\nAll deliverables generated successfully!")
