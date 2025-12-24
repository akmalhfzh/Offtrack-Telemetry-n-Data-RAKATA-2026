"""
Telemetry Data Analysis
=======================

Analyzes raw telemetry data from Shell Eco-Marathon runs.
Generates insights and visualizations for paper.

Author: RAKATA Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "raw"
FIGURE_DIR = PROJECT_ROOT / "Paper" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Plot settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between GPS points in meters."""
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def load_telemetry(filepath: Path) -> pd.DataFrame:
    """Load and preprocess telemetry data."""
    df = pd.read_csv(filepath)
    
    # Convert time to seconds
    df['time_s'] = (df['millis'] - df['millis'].iloc[0]) / 1000
    
    # Convert speed to m/s
    df['speed_mps'] = df['kecepatan'] / 3.6
    
    # Calculate distance from GPS
    distances = [0]
    for i in range(1, len(df)):
        if df['lat'].iloc[i] != 0 and df['lng'].iloc[i] != 0:
            d = haversine_distance(
                df['lat'].iloc[i-1], df['lng'].iloc[i-1],
                df['lat'].iloc[i], df['lng'].iloc[i]
            )
            distances.append(distances[-1] + min(d, 50))  # Cap single step at 50m
        else:
            distances.append(distances[-1])
    df['distance_m'] = distances
    
    # Calculate power (P = V * I)
    V_nominal = 48.0
    df['power_W'] = V_nominal * df['arus']
    
    # Calculate energy
    dt = np.diff(df['time_s'], prepend=0)
    df['energy_Wh'] = np.cumsum(df['power_W'] * dt / 3600)
    
    return df


def load_logbook(filepath: Path) -> pd.DataFrame:
    """Load logbook with lap data."""
    df = pd.read_csv(filepath)
    return df


def analyze_laps(telemetry: pd.DataFrame, logbook: pd.DataFrame) -> pd.DataFrame:
    """Analyze each lap from logbook."""
    lap_stats = []
    
    for _, row in logbook.iterrows():
        try:
            start_idx = int(row['Data_awal'])
            end_idx = int(row['Data_akhir'])
            
            if start_idx >= len(telemetry) or end_idx >= len(telemetry):
                continue
            
            lap_data = telemetry.iloc[start_idx:end_idx]
            
            if len(lap_data) < 10:
                continue
            
            # Compute stats
            duration = lap_data['time_s'].iloc[-1] - lap_data['time_s'].iloc[0]
            distance = lap_data['distance_m'].iloc[-1] - lap_data['distance_m'].iloc[0]
            energy = lap_data['energy_Wh'].iloc[-1] - lap_data['energy_Wh'].iloc[0]
            avg_speed = lap_data['kecepatan'].mean()
            max_speed = lap_data['kecepatan'].max()
            avg_power = lap_data['power_W'].mean()
            
            # Energy efficiency
            if energy > 0:
                km_per_kWh = (distance / 1000) / (energy / 1000)
            else:
                km_per_kWh = 0
            
            # Throttle analysis
            throttle_time = (lap_data['throttle'] > 0.5).sum() / len(lap_data) * 100
            
            lap_stats.append({
                'data_id': row.get('data', 'unknown'),
                'lap': row.get('Lap', 1),
                'duration_s': duration,
                'distance_m': distance,
                'energy_Wh': energy,
                'avg_speed_kmh': avg_speed,
                'max_speed_kmh': max_speed,
                'avg_power_W': avg_power,
                'km_per_kWh': km_per_kWh,
                'throttle_pct': throttle_time,
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(lap_stats)


def plot_telemetry_overview(df: pd.DataFrame, save_path: Path = None):
    """Plot overview of telemetry data."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Time array
    t = df['time_s'].values
    
    # 1. Speed profile
    ax = axes[0, 0]
    ax.plot(t, df['kecepatan'], 'b-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Speed Profile')
    ax.grid(True, alpha=0.3)
    
    # 2. Throttle state
    ax = axes[0, 1]
    ax.fill_between(t, 0, df['throttle'], alpha=0.7, color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Throttle')
    ax.set_title('Throttle State')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # 3. Current
    ax = axes[1, 0]
    ax.plot(t, df['arus'], 'r-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Current (A)')
    ax.set_title('Motor Current')
    ax.grid(True, alpha=0.3)
    
    # 4. Power
    ax = axes[1, 1]
    ax.plot(t, df['power_W'], 'orange', linewidth=0.5)
    ax.axhline(y=df['power_W'].mean(), color='red', linestyle='--', 
               label=f'Avg: {df["power_W"].mean():.1f} W')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    ax.set_title('Electrical Power')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Cumulative energy
    ax = axes[2, 0]
    ax.plot(t, df['energy_Wh'], 'purple', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Energy (Wh)')
    ax.set_title(f'Energy Consumption: {df["energy_Wh"].iloc[-1]:.1f} Wh total')
    ax.grid(True, alpha=0.3)
    
    # 6. GPS Track
    ax = axes[2, 1]
    valid_gps = (df['lat'] != 0) & (df['lng'] != 0)
    if valid_gps.sum() > 10:
        scatter = ax.scatter(df.loc[valid_gps, 'lng'], df.loc[valid_gps, 'lat'],
                           c=df.loc[valid_gps, 'kecepatan'], cmap='RdYlGn',
                           s=1, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Speed (km/h)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('GPS Track (colored by speed)')
        ax.set_aspect('equal')
    else:
        ax.text(0.5, 0.5, 'No valid GPS data', transform=ax.transAxes,
               ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_lap_comparison(lap_stats: pd.DataFrame, save_path: Path = None):
    """Compare lap performance metrics."""
    if len(lap_stats) < 2:
        print("Not enough laps for comparison")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    x = range(len(lap_stats))
    labels = [f"{row['data_id']}\nL{row['lap']}" for _, row in lap_stats.iterrows()]
    
    # 1. Energy efficiency
    ax = axes[0, 0]
    bars = ax.bar(x, lap_stats['km_per_kWh'], color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('km/kWh')
    ax.set_title('Energy Efficiency per Lap')
    ax.axhline(y=lap_stats['km_per_kWh'].mean(), color='red', linestyle='--',
              label=f'Avg: {lap_stats["km_per_kWh"].mean():.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight best
    best_idx = lap_stats['km_per_kWh'].idxmax()
    bars[best_idx].set_color('darkgreen')
    
    # 2. Lap time
    ax = axes[0, 1]
    ax.bar(x, lap_stats['duration_s'], color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Time (s)')
    ax.set_title('Lap Duration')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Average speed vs efficiency scatter
    ax = axes[1, 0]
    scatter = ax.scatter(lap_stats['avg_speed_kmh'], lap_stats['km_per_kWh'],
                        c=lap_stats['throttle_pct'], cmap='RdYlGn_r',
                        s=150, edgecolors='black')
    ax.set_xlabel('Average Speed (km/h)')
    ax.set_ylabel('Efficiency (km/kWh)')
    ax.set_title('Speed vs Efficiency Trade-off')
    plt.colorbar(scatter, ax=ax, label='Throttle %')
    ax.grid(True, alpha=0.3)
    
    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = f"""
    LAP ANALYSIS SUMMARY
    ═══════════════════════════════════
    
    Total Laps Analyzed: {len(lap_stats)}
    
    ───────────────────────────────────
    EFFICIENCY (km/kWh)
    ───────────────────────────────────
    Best:     {lap_stats['km_per_kWh'].max():.1f}
    Worst:    {lap_stats['km_per_kWh'].min():.1f}
    Average:  {lap_stats['km_per_kWh'].mean():.1f}
    
    ───────────────────────────────────
    LAP TIME (seconds)
    ───────────────────────────────────
    Fastest:  {lap_stats['duration_s'].min():.1f}
    Slowest:  {lap_stats['duration_s'].max():.1f}
    Average:  {lap_stats['duration_s'].mean():.1f}
    
    ───────────────────────────────────
    AVERAGE SPEED (km/h)
    ───────────────────────────────────
    Maximum:  {lap_stats['avg_speed_kmh'].max():.1f}
    Minimum:  {lap_stats['avg_speed_kmh'].min():.1f}
    Average:  {lap_stats['avg_speed_kmh'].mean():.1f}
    """
    
    ax.text(0.1, 0.95, summary, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def generate_latex_table(lap_stats: pd.DataFrame) -> str:
    """Generate LaTeX table for paper."""
    
    latex = r"""\begin{table}[h]
\centering
\caption{Lap Performance Analysis from Telemetry Data}
\label{tab:lap_analysis}
\begin{tabular}{lccccc}
\hline
\textbf{Lap ID} & \textbf{Time (s)} & \textbf{Distance (m)} & \textbf{Energy (Wh)} & \textbf{km/kWh} & \textbf{Avg Speed} \\
\hline
"""
    
    for _, row in lap_stats.iterrows():
        latex += f"{row['data_id']}-L{row['lap']} & {row['duration_s']:.1f} & {row['distance_m']:.0f} & {row['energy_Wh']:.1f} & {row['km_per_kWh']:.1f} & {row['avg_speed_kmh']:.1f} \\\\\n"
    
    latex += r"""\hline
\textbf{Average} & """ + f"{lap_stats['duration_s'].mean():.1f} & {lap_stats['distance_m'].mean():.0f} & {lap_stats['energy_Wh'].mean():.1f} & {lap_stats['km_per_kWh'].mean():.1f} & {lap_stats['avg_speed_kmh'].mean():.1f}" + r""" \\
\hline
\end{tabular}
\end{table}
"""
    return latex


def main():
    print("=" * 60)
    print("SEM Telemetry Data Analysis")
    print("=" * 60)
    
    # Load data
    telemetry_file = DATA_DIR / "26novfiltered.csv"
    logbook_file = DATA_DIR / "Logbook_fixed.csv"
    
    print(f"\n[1] Loading telemetry: {telemetry_file.name}")
    telemetry = load_telemetry(telemetry_file)
    print(f"    Loaded {len(telemetry)} samples")
    print(f"    Duration: {telemetry['time_s'].iloc[-1]:.1f} seconds")
    print(f"    Distance: {telemetry['distance_m'].iloc[-1]:.0f} meters")
    
    print(f"\n[2] Loading logbook: {logbook_file.name}")
    logbook = load_logbook(logbook_file)
    print(f"    Loaded {len(logbook)} lap entries")
    
    # Analyze laps
    print("\n[3] Analyzing laps...")
    lap_stats = analyze_laps(telemetry, logbook)
    print(f"    Analyzed {len(lap_stats)} valid laps")
    
    if len(lap_stats) > 0:
        print("\nLap Statistics:")
        print(lap_stats.to_string(index=False))
    
    # Generate plots
    print("\n[4] Generating visualizations...")
    
    fig1 = plot_telemetry_overview(telemetry, FIGURE_DIR / "telemetry_overview.png")
    
    if len(lap_stats) > 0:
        fig2 = plot_lap_comparison(lap_stats, FIGURE_DIR / "lap_comparison.png")
        
        # Save LaTeX table
        latex = generate_latex_table(lap_stats)
        table_path = FIGURE_DIR / "lap_table.tex"
        with open(table_path, 'w') as f:
            f.write(latex)
        print(f"\nSaved LaTeX table: {table_path}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
