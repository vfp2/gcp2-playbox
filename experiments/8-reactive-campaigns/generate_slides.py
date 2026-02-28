#!/usr/bin/env python3
"""
Generate 2 simple, glanceable charts for the 3-slide assembly deck.
Slide 2: Negative event (Eggs + Jan 2025 crisis)
Slide 3: Positive event (Champagne + Oct 10 Gaza ceasefire)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

GCP2_DIR = Path("/home/soliax/sites/gcp2-playbox/gcp2.net-rng-data-downloaded/network/global_network/2025")
OUT_DIR = Path("/home/soliax/sites/gcp2-playbox/experiments/8-reactive-campaigns/slides")
OUT_DIR.mkdir(exist_ok=True)

# Dark, clean style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#0d1117',
    'text.color': '#e6edf3',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#7d8590',
    'ytick.color': '#7d8590',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 13,
})


def load_gcp2_daily():
    print("Loading GCP2 data...")
    records = []
    for f in sorted(GCP2_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        df['date'] = pd.to_datetime(df['epoch_time_utc'], unit='s', utc=True).dt.date
        daily = df.groupby('date').agg(
            mean_nc=('network_coherence', 'mean'),
            cumsum=('network_coherence', 'sum'),
        ).reset_index()
        records.append(daily)
    result = pd.concat(records, ignore_index=True)
    result['date'] = pd.to_datetime(result['date'])
    result = result.sort_values('date').reset_index(drop=True)
    result['cumsum_smooth'] = result['cumsum'].rolling(3, center=True, min_periods=1).mean()
    return result


# ============================================================
# SLIDE 2: Negative — Eggs "CRACKED" (Jan 2025)
# ============================================================
def slide2_eggs(gcp):
    print("Generating Slide 2: Eggs crisis...")

    # Egg price curve (known points, Jan-Apr window)
    egg_dates = pd.to_datetime([
        '2025-01-01', '2025-01-05', '2025-01-10', '2025-01-15',
        '2025-02-01', '2025-02-15', '2025-03-01', '2025-03-15',
        '2025-04-01', '2025-04-15',
    ])
    egg_prices = [7.80, 8.20, 8.45, 8.53, 8.10, 7.20, 5.80, 4.08, 3.90, 3.75]
    eggs = pd.DataFrame({'date': egg_dates, 'price': egg_prices})

    # Expand to daily
    full_range = pd.DataFrame({'date': pd.date_range('2025-01-01', '2025-04-15')})
    eggs = full_range.merge(eggs, on='date', how='left')
    eggs['price'] = eggs['price'].interpolate()

    # GCP2 for same window
    window = gcp[(gcp['date'] >= '2025-01-01') & (gcp['date'] <= '2025-04-15')].copy()

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Egg price — bold, warm color
    ax1.plot(eggs['date'], eggs['price'], color='#F5DEB3', linewidth=3.5, label='Egg price ($/dozen)', zorder=3)
    ax1.fill_between(eggs['date'], eggs['price'], eggs['price'].min(), alpha=0.15, color='#F5DEB3')
    ax1.set_ylabel('Egg Price ($/dozen)', color='#F5DEB3', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#F5DEB3', labelsize=12)
    ax1.set_ylim(3, 9.5)

    # GCP2 — cyan bars
    ax2 = ax1.twinx()
    # Normalize for visual clarity — show as bars
    ax2.bar(window['date'], window['cumsum'].clip(lower=0), width=1, alpha=0.35,
            color='#00E5FF', label='GCP2 coherence spike')
    ax2.set_ylabel('GCP2 Network Coherence', color='#00E5FF', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#00E5FF', labelsize=12)

    # Mark Jan 5 — the convergence point
    ax1.annotate('JAN 5\nEggs peak at $8.53\nGCP2 top-5 spike\nPost-attack + pre-wildfire',
                 xy=(pd.to_datetime('2025-01-05'), 8.20),
                 xytext=(100, -20), textcoords='offset points',
                 fontsize=11, color='#FF6B6B', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a0000',
                           edgecolor='#FF6B6B', alpha=0.95))

    # Mark key events
    for d, label in [('2025-01-01', 'Bourbon St.\nattack'), ('2025-01-07', 'LA wildfires'),
                     ('2025-01-27', 'Nvidia\ncrash')]:
        ax1.axvline(pd.to_datetime(d), color='#FF6B6B', alpha=0.3, linewidth=1, linestyle='--')
        ax1.text(pd.to_datetime(d), 9.2, label, color='#FF6B6B', fontsize=8,
                 ha='center', va='top', alpha=0.7)

    ax1.set_title('When the world hurts AND your grocery bill hurts — we step in.',
                  fontsize=16, fontweight='bold', color='white', pad=15)

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_xlim(pd.to_datetime('2025-01-01'), pd.to_datetime('2025-04-15'))

    # Simple legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#F5DEB3', linewidth=3, label='Egg price ($/dozen)'),
        Line2D([0], [0], color='#00E5FF', linewidth=8, alpha=0.4, label='GCP2 consciousness spike'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11,
               facecolor='#161b22', edgecolor='#30363d')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'slide2_eggs_crisis.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: slide2_eggs_crisis.png")


# ============================================================
# SLIDE 3: Positive — Champagne + Oct 10 Gaza Ceasefire
# ============================================================
def slide3_champagne(gcp):
    print("Generating Slide 3: Champagne celebration...")

    # GCP2 for Sep-Nov window
    window = gcp[(gcp['date'] >= '2025-09-01') & (gcp['date'] <= '2025-11-30')].copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    # GCP2 as bars — green for positive, subtle for negative
    colors = ['#00FF88' if v > 200 else '#1a3a2a' if v > 0 else '#162030' for v in window['cumsum']]
    ax.bar(window['date'], window['cumsum'], width=1, color=colors, alpha=0.6)
    ax.plot(window['date'], window['cumsum_smooth'], color='#00FF88', linewidth=2.5, alpha=0.9)

    # Mark Oct 10 — the big one
    oct10 = window[window['date'] == '2025-10-10']
    if not oct10.empty:
        ax.scatter(oct10['date'], oct10['cumsum'], color='#FFD700', s=300, zorder=5,
                   edgecolors='white', linewidth=2)
        ax.annotate('OCT 10\nGaza ceasefire begins\nNobel Peace Prize announced\n#1 GCP2 day of the year',
                     xy=(oct10['date'].iloc[0], oct10['cumsum'].iloc[0]),
                     xytext=(-160, 40), textcoords='offset points',
                     fontsize=11, color='#FFD700', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#FFD700', lw=2),
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a00',
                               edgecolor='#FFD700', alpha=0.95))

    # Mark other joy events in the window
    joy_events = [
        ('2025-09-27', 'Ryder Cup\nEurope wins'),
        ('2025-10-20', 'Diwali\n1.5B celebrate'),
        ('2025-11-02', 'India wins\nCricket WC'),
    ]
    for d, label in joy_events:
        ax.axvline(pd.to_datetime(d), color='#00FF88', alpha=0.25, linewidth=1, linestyle='--')
        ypos = ax.get_ylim()[1] * 0.85
        ax.text(pd.to_datetime(d), ypos, label, color='#00FF88', fontsize=8,
                ha='center', va='top', alpha=0.6)

    ax.set_title('When the world celebrates together — we make the toast cheaper.',
                 fontsize=16, fontweight='bold', color='white', pad=15)
    ax.set_ylabel('GCP2 Network Coherence', color='#00FF88', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#00FF88', labelsize=12)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_xlim(pd.to_datetime('2025-09-01'), pd.to_datetime('2025-11-30'))

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'slide3_champagne_celebration.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: slide3_champagne_celebration.png")


if __name__ == '__main__':
    gcp = load_gcp2_daily()
    slide2_eggs(gcp)
    slide3_champagne(gcp)
    print("\nDone. Output in:", OUT_DIR)
