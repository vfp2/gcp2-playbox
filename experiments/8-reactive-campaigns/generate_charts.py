#!/usr/bin/env python3
"""
Generate charts for the Reactive Data Campaigns assembly presentation.
Full spectrum: crisis-side + celebration-side + unexplained anomalies.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime, timezone
import yfinance as yf

# Paths
GCP2_DIR = Path("/home/soliax/sites/gcp2-playbox/gcp2.net-rng-data-downloaded/network/global_network/2025")
OUT_DIR = Path("/home/soliax/sites/gcp2-playbox/experiments/8-reactive-campaigns/charts")
OUT_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'figure.facecolor': '#0a0a0a',
    'axes.facecolor': '#0a0a0a',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#888888',
    'ytick.color': '#888888',
    'axes.edgecolor': '#333333',
    'grid.color': '#1a1a1a',
    'grid.alpha': 0.8,
    'font.family': 'sans-serif',
    'font.size': 11,
})

GOLD_COLOR = '#FFD700'
COFFEE_COLOR = '#D2691E'
EGG_COLOR = '#F5DEB3'
GCP_COLOR = '#00E5FF'
GCP_FILL = '#00E5FF'
CRISIS_COLOR = '#FF4444'
JOY_COLOR = '#00FF88'
MYSTERY_COLOR = '#BB88FF'
CHAMPAGNE_COLOR = '#F5E6CC'
FLOWER_COLOR = '#FF69B4'
TICKET_COLOR = '#FF8C00'


# ============================================================
# DATA LOADERS
# ============================================================

def load_gcp2_daily():
    """Load all 2025 GCP2 data and compute daily aggregates."""
    print("Loading GCP2 data...")
    daily_records = []
    for csv_path in sorted(GCP2_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['epoch_time_utc'], unit='s', utc=True).dt.date
        daily = df.groupby('date').agg(
            mean_nc=('network_coherence', 'mean'),
            max_nc=('network_coherence', 'max'),
            std_nc=('network_coherence', 'std'),
            cumsum=('network_coherence', 'sum'),
            count=('network_coherence', 'count'),
            active_devices=('active_devices', 'mean'),
        ).reset_index()
        daily_records.append(daily)

    result = pd.concat(daily_records, ignore_index=True)
    result['date'] = pd.to_datetime(result['date'])
    global_mean = result['mean_nc'].mean()
    global_std = result['mean_nc'].std()
    result['z_score'] = (result['mean_nc'] - global_mean) / global_std
    result = result.sort_values('date').reset_index(drop=True)
    result['z_smooth'] = result['z_score'].rolling(7, center=True, min_periods=1).mean()
    result['cumsum_smooth'] = result['cumsum'].rolling(7, center=True, min_periods=1).mean()
    print(f"  {len(result)} days loaded")
    return result


def fetch_gold_prices():
    print("Fetching gold prices...")
    gold = yf.download("GC=F", start="2025-01-01", end="2026-01-01", progress=False)
    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.droplevel(1)
    gold = gold[['Close']].rename(columns={'Close': 'gold_close'})
    gold.index = gold.index.tz_localize(None)
    gold = gold.reset_index().rename(columns={'Date': 'date'})
    gold['date'] = pd.to_datetime(gold['date'])
    print(f"  {len(gold)} days")
    return gold


def fetch_coffee_prices():
    print("Fetching coffee prices...")
    coffee = yf.download("KC=F", start="2025-01-01", end="2026-01-01", progress=False)
    if isinstance(coffee.columns, pd.MultiIndex):
        coffee.columns = coffee.columns.droplevel(1)
    coffee = coffee[['Close']].rename(columns={'Close': 'coffee_close'})
    coffee.index = coffee.index.tz_localize(None)
    coffee = coffee.reset_index().rename(columns={'Date': 'date'})
    coffee['date'] = pd.to_datetime(coffee['date'])
    print(f"  {len(coffee)} days")
    return coffee


def construct_egg_prices():
    print("Constructing egg price curve...")
    known_points = pd.DataFrame({
        'date': pd.to_datetime([
            '2025-01-01', '2025-01-15', '2025-02-01', '2025-02-15',
            '2025-03-01', '2025-03-15', '2025-04-01', '2025-04-15',
            '2025-05-01', '2025-05-15', '2025-06-01', '2025-06-15',
            '2025-07-01', '2025-07-15', '2025-08-01', '2025-08-15',
            '2025-09-01', '2025-09-15', '2025-10-01', '2025-10-15',
            '2025-11-01', '2025-11-15', '2025-12-01', '2025-12-15',
            '2025-12-31'
        ]),
        'egg_price': [
            7.80, 8.53, 8.10, 7.20, 5.80, 4.08, 3.90, 3.75,
            3.60, 3.50, 3.40, 3.30, 3.20, 3.15, 3.10, 3.05,
            3.00, 2.95, 2.90, 2.85, 2.80, 2.78, 2.75, 2.73, 2.71
        ]
    })
    date_range = pd.date_range('2025-01-01', '2025-12-31', freq='D')
    daily = pd.DataFrame({'date': date_range})
    daily = daily.merge(known_points, on='date', how='left')
    daily['egg_price'] = daily['egg_price'].interpolate(method='linear')
    print(f"  {len(daily)} days")
    return daily


# ============================================================
# CHART 1: Full-Year Timeline — Crisis + Celebration + Mystery
# ============================================================
def chart1_full_timeline(gcp):
    print("\nGenerating Chart 1: Full Timeline (Crisis + Celebration + Mystery)...")
    fig, ax = plt.subplots(figsize=(20, 8))

    # GCP2 data
    ax.fill_between(gcp['date'], 0, gcp['cumsum'], alpha=0.12, color=GCP_FILL)
    ax.plot(gcp['date'], gcp['cumsum'], color=GCP_COLOR, linewidth=0.5, alpha=0.4)
    ax.plot(gcp['date'], gcp['cumsum_smooth'], color=GCP_COLOR, linewidth=2, alpha=0.9)

    # Dec 11 marker
    dec11 = gcp[gcp['date'] == '2025-12-11']
    if not dec11.empty:
        ax.scatter(dec11['date'], dec11['cumsum'], color=MYSTERY_COLOR, s=250, zorder=5,
                   edgecolors='white', linewidth=2)
        ax.annotate('DEC 11\nMost anomalous day\nof 2025 (4,676)\nNo known event',
                     xy=(dec11['date'].iloc[0], dec11['cumsum'].iloc[0]),
                     xytext=(25, 15), textcoords='offset points',
                     fontsize=8.5, color=MYSTERY_COLOR, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=MYSTERY_COLOR, lw=1.5))

    # CRISIS events (red/orange)
    crisis_events = [
        ('2025-01-01', 'Bourbon St. attack'),
        ('2025-01-07', 'LA wildfires begin'),
        ('2025-01-27', 'Nvidia crash $589B'),
        ('2025-03-28', 'Myanmar M7.7 quake'),
        ('2025-04-02', 'Tariff crash'),
        ('2025-04-21', 'Pope Francis dies'),
        ('2025-05-07', 'India strikes Pakistan'),
        ('2025-06-22', 'US strikes Iran nukes'),
        ('2025-07-30', 'Kamchatka M8.8 quake'),
        ('2025-08-31', 'Afghanistan quake'),
        ('2025-12-14', 'Bondi Beach shooting'),
    ]

    # CELEBRATION events (green/gold)
    joy_events = [
        ('2025-01-29', 'Lunar New Year\n2B celebrate'),
        ('2025-02-09', 'Super Bowl LIX\n133.5M viewers'),
        ('2025-03-03', 'Rio Carnival\nMoon landing'),
        ('2025-03-30', 'Eid al-Fitr\n2B celebrate'),
        ('2025-05-08', 'Pope Leo XIV elected'),
        ('2025-05-10', 'India-Pak ceasefire'),
        ('2025-05-17', 'Eurovision Final\n160M viewers'),
        ('2025-06-08', 'French Open epic\n5h 29m final'),
        ('2025-07-04', 'US July 4th\nOasis reunion'),
        ('2025-10-10', 'Gaza ceasefire\nNobel Peace Prize'),
        ('2025-10-20', 'Diwali\n1.5B celebrate'),
    ]

    # MYSTERY events (purple)
    mystery_events = [
        ('2025-08-10', 'UNEXPLAINED\nZ = 42.32'),
    ]

    ymin, ymax = ax.get_ylim()

    for date_str, label in crisis_events:
        d = pd.to_datetime(date_str)
        ax.axvline(d, color=CRISIS_COLOR, alpha=0.25, linewidth=0.7, linestyle='--')
        y_pos = ymin + (ymax - ymin) * 0.60
        ax.annotate(label, xy=(d, y_pos), fontsize=6.5, color=CRISIS_COLOR, alpha=0.8,
                    ha='center', va='bottom', rotation=90, xytext=(0, 5), textcoords='offset points')

    for date_str, label in joy_events:
        d = pd.to_datetime(date_str)
        ax.axvline(d, color=JOY_COLOR, alpha=0.25, linewidth=0.7, linestyle='--')
        y_pos = ymin + (ymax - ymin) * 0.35
        ax.annotate(label, xy=(d, y_pos), fontsize=6.5, color=JOY_COLOR, alpha=0.8,
                    ha='center', va='bottom', rotation=90, xytext=(0, 5), textcoords='offset points')

    for date_str, label in mystery_events:
        d = pd.to_datetime(date_str)
        ax.axvline(d, color=MYSTERY_COLOR, alpha=0.4, linewidth=1.2, linestyle='-')
        y_pos = ymin + (ymax - ymin) * 0.75
        ax.annotate(label, xy=(d, y_pos), fontsize=7.5, color=MYSTERY_COLOR, alpha=0.9,
                    ha='center', va='bottom', rotation=90, fontweight='bold',
                    xytext=(0, 5), textcoords='offset points')

    # Legend
    legend_elements = [
        Line2D([0], [0], color=GCP_COLOR, linewidth=2, label='GCP2 Daily Coherence (7d smooth)'),
        Line2D([0], [0], color=CRISIS_COLOR, linewidth=1, linestyle='--', label='Crisis / Negative Event'),
        Line2D([0], [0], color=JOY_COLOR, linewidth=1, linestyle='--', label='Celebration / Positive Event'),
        Line2D([0], [0], color=MYSTERY_COLOR, linewidth=1.5, label='Unexplained Anomaly'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              facecolor='#1a1a1a', edgecolor='#333333')

    ax.set_title('GCP2 Global Network Coherence — 2025\n'
                 'The Full Picture: Crises, Celebrations, and Mysteries',
                 fontsize=16, fontweight='bold', color='white', pad=20)
    ax.set_xlabel('2025', fontsize=12)
    ax.set_ylabel('Daily Coherence Sum', fontsize=12)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(gcp['date'].min(), gcp['date'].max())

    fig.tight_layout()
    fig.savefig(OUT_DIR / '01_full_timeline_2025.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: 01_full_timeline_2025.png")


# ============================================================
# CHART 2: Gold "Collective Nerve" (updated Oct 10 annotation)
# ============================================================
def chart2_gold(gcp, gold):
    print("\nGenerating Chart 2: Gold — Collective Nerve...")
    merged = gcp.merge(gold, on='date', how='inner')
    fig, ax1 = plt.subplots(figsize=(18, 7))

    ax1.plot(merged['date'], merged['gold_close'], color=GOLD_COLOR, linewidth=2.5,
             label='Gold Price ($/oz)', zorder=3)
    ax1.set_ylabel('Gold Price ($/oz)', color=GOLD_COLOR, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=GOLD_COLOR)
    ax1.set_ylim(merged['gold_close'].min() * 0.95, merged['gold_close'].max() * 1.05)

    ax2 = ax1.twinx()
    ax2.fill_between(merged['date'], 0, merged['cumsum'].clip(lower=0), alpha=0.12, color=GCP_FILL)
    ax2.plot(merged['date'], merged['cumsum_smooth'], color=GCP_COLOR, linewidth=1.5, alpha=0.8,
             label='GCP2 Coherence (7d smooth)')
    ax2.set_ylabel('GCP2 Daily Coherence Sum', color=GCP_COLOR, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=GCP_COLOR)

    # Activation zones
    threshold = gcp['cumsum'].quantile(0.9)
    for _, row in merged[merged['cumsum'] > threshold].iterrows():
        ax1.axvspan(row['date'] - pd.Timedelta(hours=12), row['date'] + pd.Timedelta(hours=12),
                     alpha=0.08, color=CRISIS_COLOR, zorder=0)

    # Oct 10 — updated to reflect ceasefire + gold milestone
    oct10 = merged[merged['date'] == '2025-10-10']
    if not oct10.empty:
        ax1.annotate('OCT 10: #1 GCP2 anomaly\nGaza ceasefire takes effect\n'
                     'Nobel Peace Prize announced\nGold breaks $4,000\n'
                     '"COLLECTIVE NERVE: ZERO PREMIUM"',
                     xy=(oct10['date'].iloc[0], oct10['gold_close'].iloc[0]),
                     xytext=(-160, 50), textcoords='offset points',
                     fontsize=8, color=CRISIS_COLOR, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=CRISIS_COLOR, lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a0000',
                               edgecolor=CRISIS_COLOR, alpha=0.9))

    # May 7-10
    may_data = merged[(merged['date'] >= '2025-05-07') & (merged['date'] <= '2025-05-10')]
    if not may_data.empty:
        row = may_data.iloc[-1]
        ax1.annotate('MAY 7-10\nIndia-Pakistan conflict\n"COLLECTIVE NERVE: ACTIVE"',
                     xy=(row['date'], row['gold_close']),
                     xytext=(-80, -80), textcoords='offset points',
                     fontsize=8, color='#FFAA44', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#FFAA44', lw=1.2),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1000',
                               edgecolor='#FFAA44', alpha=0.9))

    ax1.set_title('Campaign 1: "THE COLLECTIVE NERVE" — Gold Price vs. Global Consciousness\n'
                  'Red zones = campaign activation (premium drops to 0%)',
                  fontsize=14, fontweight='bold', color='white', pad=20)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(merged['date'].min(), merged['date'].max())

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10,
               facecolor='#1a1a1a', edgecolor='#333333')

    fig.tight_layout()
    fig.savefig(OUT_DIR / '02_gold_collective_nerve.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: 02_gold_collective_nerve.png")


# ============================================================
# CHART 3: Coffee "The World Pause"
# ============================================================
def chart3_coffee(gcp, coffee):
    print("\nGenerating Chart 3: Coffee — The World Pause...")
    merged = gcp.merge(coffee, on='date', how='inner')
    fig, ax1 = plt.subplots(figsize=(18, 7))

    ax1.plot(merged['date'], merged['coffee_close'], color=COFFEE_COLOR, linewidth=2.5,
             label='Arabica Coffee (cents/lb)', zorder=3)
    ax1.set_ylabel('Arabica Coffee Price (cents/lb)', color=COFFEE_COLOR, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=COFFEE_COLOR)

    ax2 = ax1.twinx()
    ax2.fill_between(merged['date'], 0, merged['cumsum'].clip(lower=0), alpha=0.12, color=GCP_FILL)
    ax2.plot(merged['date'], merged['cumsum_smooth'], color=GCP_COLOR, linewidth=1.5, alpha=0.8,
             label='GCP2 "World Pulse" (7d smooth)')
    ax2.set_ylabel('GCP2 Daily Coherence Sum', color=GCP_COLOR, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=GCP_COLOR)

    threshold = gcp['cumsum'].quantile(0.9)
    for _, row in merged[merged['cumsum'] > threshold].iterrows():
        ax1.axvspan(row['date'] - pd.Timedelta(hours=12), row['date'] + pd.Timedelta(hours=12),
                     alpha=0.07, color=JOY_COLOR, zorder=0)

    # Aug 10
    aug10 = merged[(merged['date'] >= '2025-08-08') & (merged['date'] <= '2025-08-12')]
    if not aug10.empty:
        row = aug10.iloc[len(aug10)//2]
        ax1.annotate('AUG 10: Z = 42.32\nMost extreme hour of 2025\n'
                     '"THE WORLD PAUSE: 50% OFF"',
                     xy=(row['date'], row['coffee_close']),
                     xytext=(60, 60), textcoords='offset points',
                     fontsize=9, color=JOY_COLOR, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=JOY_COLOR, lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#001a0a',
                               edgecolor=JOY_COLOR, alpha=0.9))

    # Apr 2-4
    apr = merged[(merged['date'] >= '2025-04-01') & (merged['date'] <= '2025-04-05')]
    if not apr.empty:
        row = apr.iloc[-1]
        ax1.annotate('APR 2-4: Tariff crash\n"WORLD PAUSE: ACTIVE"',
                     xy=(row['date'], row['coffee_close']),
                     xytext=(-90, -70), textcoords='offset points',
                     fontsize=8, color='#88DDAA', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#88DDAA', lw=1.2),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#001a0a',
                               edgecolor='#88DDAA', alpha=0.9))

    ax1.set_title('Campaign 2: "THE WORLD PAUSE" — Coffee Price vs. Global Consciousness\n'
                  'Green zones = campaign activation (coffee up to 50% off)',
                  fontsize=14, fontweight='bold', color='white', pad=20)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(merged['date'].min(), merged['date'].max())

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10,
               facecolor='#1a1a1a', edgecolor='#333333')

    fig.tight_layout()
    fig.savefig(OUT_DIR / '03_coffee_world_pause.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: 03_coffee_world_pause.png")


# ============================================================
# CHART 4: Eggs "Cracked" (Dual Signal)
# ============================================================
def chart4_eggs(gcp, eggs):
    print("\nGenerating Chart 4: Eggs — Cracked...")
    merged = gcp.merge(eggs, on='date', how='inner')
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [2, 2, 1.5]})

    # Panel 1: Egg Price
    ax = axes[0]
    ax.fill_between(merged['date'], merged['egg_price'], merged['egg_price'].min(), alpha=0.3, color=EGG_COLOR)
    ax.plot(merged['date'], merged['egg_price'], color=EGG_COLOR, linewidth=2.5, label='Egg Price ($/dozen)')
    ax.axhline(y=5.0, color=CRISIS_COLOR, linestyle=':', alpha=0.5, linewidth=1)
    ax.text(merged['date'].iloc[5], 5.15, 'Crisis threshold ($5.00)', color=CRISIS_COLOR, fontsize=8, alpha=0.7)
    peak_row = merged.loc[merged['egg_price'].idxmax()]
    ax.annotate(f'PEAK: ${peak_row["egg_price"]:.2f}/dozen\nJan 2025 — Avian flu crisis',
                xy=(peak_row['date'], peak_row['egg_price']),
                xytext=(80, -10), textcoords='offset points',
                fontsize=9, color=CRISIS_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CRISIS_COLOR, lw=1.5))
    ax.set_ylabel('WALLET SIGNAL\nEgg Price ($/dozen)', fontsize=11, color=EGG_COLOR)
    ax.tick_params(axis='y', labelcolor=EGG_COLOR)
    ax.set_title('Campaign 3: "CRACKED" — Dual Signal Reactive Pricing for Eggs\n'
                 'World Signal (GCP2) + Wallet Signal (USDA Egg Index)',
                 fontsize=14, fontweight='bold', color='white', pad=20)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(merged['date'].min(), merged['date'].max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend(loc='upper right', fontsize=10, facecolor='#1a1a1a', edgecolor='#333333')

    # Panel 2: GCP2
    ax = axes[1]
    ax.fill_between(merged['date'], 0, merged['cumsum'].clip(lower=0), alpha=0.15, color=GCP_FILL)
    ax.plot(merged['date'], merged['cumsum_smooth'], color=GCP_COLOR, linewidth=2, label='GCP2 Coherence (7d smooth)')
    ax.plot(merged['date'], merged['cumsum'], color=GCP_COLOR, linewidth=0.4, alpha=0.4)
    ax.set_ylabel('WORLD SIGNAL\nGCP2 Coherence Sum', fontsize=11, color=GCP_COLOR)
    ax.tick_params(axis='y', labelcolor=GCP_COLOR)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(merged['date'].min(), merged['date'].max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend(loc='upper left', fontsize=10, facecolor='#1a1a1a', edgecolor='#333333')

    # Panel 3: Cracked Index
    ax = axes[2]
    egg_5yr_avg = 3.50
    wallet_signal = ((merged['egg_price'] - egg_5yr_avg) / egg_5yr_avg * 100).clip(0, 100)
    cs_min = merged['cumsum_smooth'].quantile(0.05)
    cs_max = merged['cumsum_smooth'].quantile(0.95)
    world_signal = ((merged['cumsum_smooth'] - cs_min) / (cs_max - cs_min) * 100).clip(0, 100)
    cracked_index = np.sqrt(wallet_signal * world_signal)

    ax.fill_between(merged['date'], 0, cracked_index, alpha=0.3,
                    color=CRISIS_COLOR, where=cracked_index > 70)
    ax.fill_between(merged['date'], 0, cracked_index, alpha=0.2,
                    color='#FFAA44', where=(cracked_index > 30) & (cracked_index <= 70))
    ax.fill_between(merged['date'], 0, cracked_index, alpha=0.15,
                    color=JOY_COLOR, where=cracked_index <= 30)
    ax.plot(merged['date'], cracked_index, color='white', linewidth=1.5, label='Cracked Index')
    ax.axhline(y=70, color=CRISIS_COLOR, linestyle=':', alpha=0.6, linewidth=1)
    ax.axhline(y=30, color=JOY_COLOR, linestyle=':', alpha=0.6, linewidth=1)
    ax.text(merged['date'].iloc[-15], 73, 'FULL CRACK: Eggs $2.99', color=CRISIS_COLOR, fontsize=8, ha='right')
    ax.text(merged['date'].iloc[-15], 23, 'Stable: No subsidy needed', color=JOY_COLOR, fontsize=8, ha='right')

    jan5_idx = merged[merged['date'] == '2025-01-05'].index
    if len(jan5_idx) > 0:
        ci_val = cracked_index.iloc[jan5_idx[0]] if jan5_idx[0] < len(cracked_index) else 50
        ax.annotate('JAN 5: FULL CRACK\nEggs $8.53 + Top-5 GCP2\n= Eggs subsidized to $2.99',
                    xy=(merged['date'].iloc[jan5_idx[0]], ci_val),
                    xytext=(120, 15), textcoords='offset points',
                    fontsize=8, color=CRISIS_COLOR, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=CRISIS_COLOR, lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a0000',
                              edgecolor=CRISIS_COLOR, alpha=0.9))

    ax.set_ylabel('CRACKED INDEX\n(0-100)', fontsize=11, color='white')
    ax.set_xlabel('2025', fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(merged['date'].min(), merged['date'].max())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend(loc='upper right', fontsize=10, facecolor='#1a1a1a', edgecolor='#333333')

    fig.tight_layout()
    fig.savefig(OUT_DIR / '04_eggs_cracked.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: 04_eggs_cracked.png")


# ============================================================
# CHART 5: Celebration-Side — GCP2 + Joy Events overlay
# ============================================================
def chart5_celebrations(gcp):
    print("\nGenerating Chart 5: Celebration Events overlay...")
    fig, ax = plt.subplots(figsize=(18, 7))

    ax.fill_between(gcp['date'], 0, gcp['cumsum'].clip(lower=0), alpha=0.10, color=JOY_COLOR)
    ax.plot(gcp['date'], gcp['cumsum'], color=GCP_COLOR, linewidth=0.5, alpha=0.4)
    ax.plot(gcp['date'], gcp['cumsum_smooth'], color=GCP_COLOR, linewidth=2, alpha=0.9)

    celebrations = [
        ('2025-01-29', 'Lunar New Year\n~2B celebrate', JOY_COLOR),
        ('2025-02-09', 'Super Bowl LIX\n133.5M viewers', JOY_COLOR),
        ('2025-03-03', 'Rio Carnival\n+ Moon landing', '#FFD700'),
        ('2025-03-14', 'Holi\n~1B celebrate', '#FF69B4'),
        ('2025-03-30', 'Eid al-Fitr\n~2B celebrate', JOY_COLOR),
        ('2025-04-20', 'Easter\n2.4B Christians\n(East+West united)', JOY_COLOR),
        ('2025-05-08', 'Pope Leo XIV\nFirst American pope', '#FFD700'),
        ('2025-05-17', 'Eurovision Final\n160M viewers', '#FF69B4'),
        ('2025-05-31', 'Champions League\nPSG 5-0 Inter', JOY_COLOR),
        ('2025-06-06', 'Eid al-Adha\n~2B celebrate', JOY_COLOR),
        ('2025-06-08', 'French Open epic\nAlcaraz 5h 29m', '#FFD700'),
        ('2025-07-04', 'US July 4th\n+ Oasis reunion', '#FFD700'),
        ('2025-07-27', 'Tour de France\nPogacar wins', JOY_COLOR),
        ('2025-09-27', 'Ryder Cup\nEurope wins', JOY_COLOR),
        ('2025-10-10', 'GAZA CEASEFIRE\n+ Nobel Peace Prize', '#FFD700'),
        ('2025-10-20', 'Diwali\n~1.5B celebrate', '#FF69B4'),
        ('2025-11-02', 'India wins\nWomen\'s Cricket WC', JOY_COLOR),
    ]

    ymin, ymax = ax.get_ylim()
    for i, (date_str, label, color) in enumerate(celebrations):
        d = pd.to_datetime(date_str)
        ax.axvline(d, color=color, alpha=0.35, linewidth=0.8, linestyle='--')
        # Alternate heights to reduce overlap
        y_frac = 0.45 + (i % 3) * 0.15
        y_pos = ymin + (ymax - ymin) * y_frac
        ax.annotate(label, xy=(d, y_pos), fontsize=6.5, color=color, alpha=0.85,
                    ha='center', va='bottom', rotation=90,
                    xytext=(0, 5), textcoords='offset points')

    # Highlight Oct 10
    oct10 = gcp[gcp['date'] == '2025-10-10']
    if not oct10.empty:
        ax.scatter(oct10['date'], oct10['cumsum'], color='#FFD700', s=200, zorder=5,
                   edgecolors='white', linewidth=1.5)
        ax.annotate('OCT 10: Gaza ceasefire + Nobel Peace\n'
                    '#1 GCP2 anomaly of 2025\n'
                    'Not fear — RELIEF',
                    xy=(oct10['date'].iloc[0], oct10['cumsum'].iloc[0]),
                    xytext=(-140, 60), textcoords='offset points',
                    fontsize=9, color='#FFD700', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a00',
                              edgecolor='#FFD700', alpha=0.9))

    ax.set_title('GCP2 Global Network Coherence — 2025\n'
                 'The Celebration Side: Joy, Awe, and Collective Relief',
                 fontsize=16, fontweight='bold', color='white', pad=20)
    ax.set_xlabel('2025', fontsize=12)
    ax.set_ylabel('Daily Coherence Sum', fontsize=12)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(gcp['date'].min(), gcp['date'].max())

    fig.tight_layout()
    fig.savefig(OUT_DIR / '05_celebrations_overlay.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: 05_celebrations_overlay.png")


# ============================================================
# CHART 6: The Spectrum — All 6 Campaigns FEAR ←→ JOY
# ============================================================
def chart6_spectrum(gcp, gold, coffee, eggs):
    print("\nGenerating Chart 6: The Full Spectrum (6 campaigns)...")
    merged_gold = gcp.merge(gold, on='date', how='inner')
    merged_coffee = gcp.merge(coffee, on='date', how='inner')
    merged_eggs = gcp.merge(eggs, on='date', how='inner')

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # Row 1: CRISIS SIDE
    campaigns_crisis = [
        (axes[0, 0], '"THE COLLECTIVE NERVE"', 'Gold', GOLD_COLOR,
         merged_gold['date'], merged_gold['gold_close'], 'Gold $/oz',
         merged_gold['cumsum_smooth'],
         '"Your safety shouldn\'t cost more\nwhen you need it most"'),
        (axes[0, 1], '"THE WORLD PAUSE"', 'Coffee', COFFEE_COLOR,
         merged_coffee['date'], merged_coffee['coffee_close'], 'Arabica cents/lb',
         merged_coffee['cumsum_smooth'],
         '"When the world can\'t slow down,\nwe make calm more affordable"'),
        (axes[0, 2], '"CRACKED"', 'Eggs (Dual Signal)', EGG_COLOR,
         merged_eggs['date'], merged_eggs['egg_price'], 'Eggs $/dozen',
         gcp.set_index('date').reindex(merged_eggs['date'])['cumsum_smooth'].values,
         '"When the world cracks and your\nbudget does too, we\'ve got you"'),
    ]

    for ax, title, subtitle, color, dates, prices, ylabel, gcp_data, tagline in campaigns_crisis:
        ax_twin = ax.twinx()
        ax.plot(dates, prices, color=color, linewidth=2)
        ax_twin.fill_between(dates, 0, np.maximum(gcp_data, 0), alpha=0.12, color=GCP_FILL)
        ax_twin.plot(dates, gcp_data, color=GCP_COLOR, linewidth=1, alpha=0.7)
        ax.set_title(f'{title}\n{subtitle}', fontsize=12, fontweight='bold', color=color, pad=10)
        ax.set_ylabel(ylabel, color=color, fontsize=9)
        ax_twin.set_ylabel('GCP2', color=GCP_COLOR, fontsize=9)
        ax.tick_params(axis='y', labelcolor=color)
        ax_twin.tick_params(axis='y', labelcolor=GCP_COLOR)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.grid(True, alpha=0.2)
        ax.text(0.5, -0.13, tagline, transform=ax.transAxes, fontsize=8,
                color='#aaaaaa', ha='center', style='italic')

    # Row 2: CELEBRATION SIDE
    # These don't have commodity price overlays — use GCP2 with event markers

    # Campaign 4: Champagne
    ax = axes[1, 0]
    ax.fill_between(gcp['date'], 0, gcp['cumsum'].clip(lower=0), alpha=0.15, color=JOY_COLOR)
    ax.plot(gcp['date'], gcp['cumsum_smooth'], color=GCP_COLOR, linewidth=1.5, alpha=0.9)
    joy_dates = ['2025-01-29', '2025-02-09', '2025-03-03', '2025-05-17',
                 '2025-07-04', '2025-10-10', '2025-10-20']
    for d_str in joy_dates:
        d = pd.to_datetime(d_str)
        ax.axvline(d, color=CHAMPAGNE_COLOR, alpha=0.5, linewidth=1, linestyle='--')
        # Small champagne-glass emoji substitute: gold dot
        gcp_row = gcp[gcp['date'] == d_str]
        if not gcp_row.empty:
            ax.scatter(gcp_row['date'], gcp_row['cumsum'], color=CHAMPAGNE_COLOR, s=60, zorder=5,
                       edgecolors='white', linewidth=0.5)
    ax.set_title('"WHEN THE WORLD TOASTS"\nChampagne / Prosecco', fontsize=12,
                 fontweight='bold', color=CHAMPAGNE_COLOR, pad=10)
    ax.set_ylabel('GCP2 Coherence', color=GCP_COLOR, fontsize=9)
    ax.tick_params(axis='y', labelcolor=GCP_COLOR)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.grid(True, alpha=0.2)
    ax.set_xlim(gcp['date'].min(), gcp['date'].max())
    ax.text(0.5, -0.13, '"When the world celebrates,\nbubbles get cheaper"',
            transform=ax.transAxes, fontsize=8, color='#aaaaaa', ha='center', style='italic')

    # Campaign 5: Concert Tickets / Joy Pricing
    ax = axes[1, 1]
    ax.fill_between(gcp['date'], 0, gcp['cumsum'].clip(lower=0), alpha=0.15, color=TICKET_COLOR)
    ax.plot(gcp['date'], gcp['cumsum_smooth'], color=GCP_COLOR, linewidth=1.5, alpha=0.9)
    music_dates = ['2025-02-09', '2025-04-11', '2025-05-17', '2025-06-25',
                   '2025-07-04', '2025-09-13']
    for d_str in music_dates:
        d = pd.to_datetime(d_str)
        ax.axvline(d, color=TICKET_COLOR, alpha=0.5, linewidth=1, linestyle='--')
        gcp_row = gcp[gcp['date'] == d_str]
        if not gcp_row.empty:
            ax.scatter(gcp_row['date'], gcp_row['cumsum'], color=TICKET_COLOR, s=60, zorder=5,
                       edgecolors='white', linewidth=0.5)
    ax.set_title('"JOY PRICING"\nConcert Tickets', fontsize=12,
                 fontweight='bold', color=TICKET_COLOR, pad=10)
    ax.set_ylabel('GCP2 Coherence', color=GCP_COLOR, fontsize=9)
    ax.tick_params(axis='y', labelcolor=GCP_COLOR)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.grid(True, alpha=0.2)
    ax.set_xlim(gcp['date'].min(), gcp['date'].max())
    ax.text(0.5, -0.13, '"Everyone hates surge pricing.\nWe invented joy pricing."',
            transform=ax.transAxes, fontsize=8, color='#aaaaaa', ha='center', style='italic')

    # Campaign 6: Flowers
    ax = axes[1, 2]
    ax.fill_between(gcp['date'], 0, gcp['cumsum'].clip(lower=0), alpha=0.10, color=FLOWER_COLOR)
    ax.fill_between(gcp['date'], gcp['cumsum'].clip(upper=0), 0, alpha=0.10, color=CRISIS_COLOR)
    ax.plot(gcp['date'], gcp['cumsum_smooth'], color=GCP_COLOR, linewidth=1.5, alpha=0.9)
    # Mark both crisis AND joy events
    both_dates = [('2025-01-07', CRISIS_COLOR), ('2025-03-03', JOY_COLOR),
                  ('2025-04-21', CRISIS_COLOR), ('2025-05-11', FLOWER_COLOR),  # Mother's Day
                  ('2025-07-04', JOY_COLOR), ('2025-10-10', JOY_COLOR),
                  ('2025-12-14', CRISIS_COLOR)]
    for d_str, c in both_dates:
        d = pd.to_datetime(d_str)
        ax.axvline(d, color=c, alpha=0.5, linewidth=1, linestyle='--')
        gcp_row = gcp[gcp['date'] == d_str]
        if not gcp_row.empty:
            ax.scatter(gcp_row['date'], gcp_row['cumsum'], color=c, s=60, zorder=5,
                       edgecolors='white', linewidth=0.5)
    ax.set_title('"WHEN THE WORLD BLOOMS"\nFlowers (Both Sides)', fontsize=12,
                 fontweight='bold', color=FLOWER_COLOR, pad=10)
    ax.set_ylabel('GCP2 Coherence', color=GCP_COLOR, fontsize=9)
    ax.tick_params(axis='y', labelcolor=GCP_COLOR)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.grid(True, alpha=0.2)
    ax.set_xlim(gcp['date'].min(), gcp['date'].max())
    ax.text(0.5, -0.13, '"Flowers mark every moment that matters.\nAny signal. Any emotion."',
            transform=ax.transAxes, fontsize=8, color='#aaaaaa', ha='center', style='italic')

    # Row labels
    fig.text(0.01, 0.75, 'CRISIS\nSIDE', fontsize=14, fontweight='bold',
             color=CRISIS_COLOR, ha='left', va='center', rotation=0)
    fig.text(0.01, 0.28, 'JOY\nSIDE', fontsize=14, fontweight='bold',
             color=JOY_COLOR, ha='left', va='center', rotation=0)

    fig.suptitle('SIX REACTIVE DATA CAMPAIGNS POWERED BY GLOBAL CONSCIOUSNESS\n'
                 'FEAR <————————————————————> JOY',
                 fontsize=16, fontweight='bold', color='white', y=1.01)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    fig.savefig(OUT_DIR / '06_full_spectrum_6_campaigns.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: 06_full_spectrum_6_campaigns.png")


# ============================================================
# CHART 7: The Unexplained — Dec 11 + Aug 10 (closing slides)
# ============================================================
def chart7_unexplained(gcp):
    print("\nGenerating Chart 7: The Unexplained (closing slides)...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # --- Dec 11 ---
    ax = axes[0]
    # Show Nov 15 - Dec 31 window
    window = gcp[(gcp['date'] >= '2025-11-15') & (gcp['date'] <= '2025-12-31')]
    ax.fill_between(window['date'], 0, window['cumsum'], alpha=0.15, color=MYSTERY_COLOR)
    ax.plot(window['date'], window['cumsum'], color=GCP_COLOR, linewidth=1, alpha=0.6)
    ax.plot(window['date'], window['cumsum_smooth'], color=GCP_COLOR, linewidth=2, alpha=0.9)

    dec11 = window[window['date'] == '2025-12-11']
    if not dec11.empty:
        ax.scatter(dec11['date'], dec11['cumsum'], color=MYSTERY_COLOR, s=400, zorder=5,
                   edgecolors='white', linewidth=2)
        ax.annotate('DECEMBER 11\nCumulative sum: 4,676\nMean: 100x baseline\nZ = 15.15 sustained\n\nNo known event.',
                     xy=(dec11['date'].iloc[0], dec11['cumsum'].iloc[0]),
                     xytext=(-140, -20), textcoords='offset points',
                     fontsize=10, color=MYSTERY_COLOR, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=MYSTERY_COLOR, lw=2),
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#0a0015',
                               edgecolor=MYSTERY_COLOR, alpha=0.95))

    # Mark Dec 14
    ax.axvline(pd.to_datetime('2025-12-14'), color=CRISIS_COLOR, alpha=0.6, linewidth=1.5, linestyle='--')
    ax.annotate('DEC 14\nBondi Beach\nHanukkah shooting\n15 killed',
                xy=(pd.to_datetime('2025-12-14'), ax.get_ylim()[1] * 0.3),
                fontsize=8, color=CRISIS_COLOR, ha='center', va='bottom', rotation=90)

    # Arrow between dates
    if not dec11.empty:
        ax.annotate('', xy=(pd.to_datetime('2025-12-14'), dec11['cumsum'].iloc[0] * 0.5),
                    xytext=(dec11['date'].iloc[0], dec11['cumsum'].iloc[0] * 0.5),
                    arrowprops=dict(arrowstyle='<->', color='white', lw=1, alpha=0.5))
        ax.text(pd.to_datetime('2025-12-12') + pd.Timedelta(hours=12),
                dec11['cumsum'].iloc[0] * 0.52, '3 days', color='white',
                fontsize=9, ha='center', alpha=0.7)

    ax.set_title('THE UNEXPLAINED: DECEMBER 11, 2025\n'
                 'The most anomalous day of the year. No known cause.',
                 fontsize=13, fontweight='bold', color=MYSTERY_COLOR, pad=15)
    ax.set_ylabel('Daily Coherence Sum', fontsize=11, color=GCP_COLOR)
    ax.tick_params(axis='y', labelcolor=GCP_COLOR)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(True, alpha=0.2)

    # --- Aug 10 ---
    ax = axes[1]
    window2 = gcp[(gcp['date'] >= '2025-07-25') & (gcp['date'] <= '2025-08-25')]
    ax.fill_between(window2['date'], 0, window2['cumsum'], alpha=0.15, color=MYSTERY_COLOR)
    ax.plot(window2['date'], window2['cumsum'], color=GCP_COLOR, linewidth=1, alpha=0.6)
    ax.plot(window2['date'], window2['cumsum_smooth'], color=GCP_COLOR, linewidth=2, alpha=0.9)

    aug10 = window2[window2['date'] == '2025-08-10']
    if not aug10.empty:
        ax.scatter(aug10['date'], aug10['cumsum'], color=MYSTERY_COLOR, s=400, zorder=5,
                   edgecolors='white', linewidth=2)
        ax.annotate('AUGUST 10\nHourly Z = 42.32\nMost extreme sustained\nhour of 2025\n\n'
                    'Nothing happened.\nOr nothing we measured.',
                     xy=(aug10['date'].iloc[0], aug10['cumsum'].iloc[0]),
                     xytext=(60, 80), textcoords='offset points',
                     fontsize=10, color=MYSTERY_COLOR, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=MYSTERY_COLOR, lw=2),
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#0a0015',
                               edgecolor=MYSTERY_COLOR, alpha=0.95))

    # Mark Aug 15
    ax.axvline(pd.to_datetime('2025-08-15'), color='#FFAA44', alpha=0.6, linewidth=1.5, linestyle='--')
    ax.annotate('AUG 15\nTrump-Putin\nsummit',
                xy=(pd.to_datetime('2025-08-15'), ax.get_ylim()[1] * 0.6),
                fontsize=8, color='#FFAA44', ha='center', va='bottom', rotation=90)

    if not aug10.empty:
        ax.annotate('', xy=(pd.to_datetime('2025-08-15'), aug10['cumsum'].iloc[0] * 0.3),
                    xytext=(aug10['date'].iloc[0], aug10['cumsum'].iloc[0] * 0.3),
                    arrowprops=dict(arrowstyle='<->', color='white', lw=1, alpha=0.5))
        mid_date = pd.to_datetime('2025-08-12') + pd.Timedelta(hours=12)
        ax.text(mid_date, aug10['cumsum'].iloc[0] * 0.33, '5 days',
                color='white', fontsize=9, ha='center', alpha=0.7)

    ax.set_title('THE UNEXPLAINED: AUGUST 10, 2025\n'
                 '42 standard deviations. Five days before two leaders decided a war.',
                 fontsize=13, fontweight='bold', color=MYSTERY_COLOR, pad=15)
    ax.set_ylabel('Daily Coherence Sum', fontsize=11, color=GCP_COLOR)
    ax.tick_params(axis='y', labelcolor=GCP_COLOR)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / '07_the_unexplained.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: 07_the_unexplained.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("REACTIVE DATA CAMPAIGNS — CHART GENERATOR v2")
    print("Full spectrum: Crisis + Celebration + Mystery")
    print("=" * 60)

    gcp = load_gcp2_daily()
    gold = fetch_gold_prices()
    coffee = fetch_coffee_prices()
    eggs = construct_egg_prices()

    chart1_full_timeline(gcp)
    chart2_gold(gcp, gold)
    chart3_coffee(gcp, coffee)
    chart4_eggs(gcp, eggs)
    chart5_celebrations(gcp)
    chart6_spectrum(gcp, gold, coffee, eggs)
    chart7_unexplained(gcp)

    print("\n" + "=" * 60)
    print("ALL 7 CHARTS GENERATED")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)
