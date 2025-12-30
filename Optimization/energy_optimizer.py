"""
Energy Optimization using Trained NARX Model
=============================================

Enhanced version with "Golden Standard" visualizations:
- GPS Track with Throttle Probability overlay
- Recommended Gas Start/Stop markers
- Speed Profile with Target Zone
- Lap-by-lap comparison charts

Supports both NARX V2 (NARX_SEM_Model_Final.mat) and V3 (NARX_V3_Model.mat)

Paper: Telemetry-Driven Digital Twin Modeling with ML-Enhanced 
       Energy Optimization for Shell Eco-Marathon

Author: RAKATA Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from scipy.io import loadmat
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "raw"
MODEL_DIR = PROJECT_ROOT / "NN"
FIGURE_DIR = PROJECT_ROOT / "Paper" / "figures"

# Create separate output directories for V2 and V3
FIGURE_DIR_V2 = FIGURE_DIR / "v2_output"
FIGURE_DIR_V3 = FIGURE_DIR / "v3_output"

# Publication settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


@dataclass
class VehicleParams:
    """Shell Eco-Marathon Urban Concept vehicle parameters."""
    mass: float = 90.0           # kg (vehicle + driver)
    Cd: float = 0.15             # Drag coefficient
    A_frontal: float = 0.4       # m² frontal area
    Crr: float = 0.005           # Rolling resistance
    wheel_radius: float = 0.28   # m
    motor_efficiency: float = 0.85
    V_battery: float = 48.0      # V nominal
    rho_air: float = 1.225       # kg/m³
    g: float = 9.81              # m/s²


@dataclass
class LapData:
    """Processed lap data from NARX model."""
    lap_number: int
    source: str
    lat: np.ndarray
    lng: np.ndarray
    speed: np.ndarray
    distance: np.ndarray
    throttle_state: np.ndarray
    speed_upper: np.ndarray
    speed_lower: np.ndarray
    total_distance: float
    lap_time: float
    lap_energy: float  # Wh
    has_gps: bool = True
    throttle_distances: Optional[np.ndarray] = None
    gliding_distances: Optional[np.ndarray] = None


@dataclass
class OptimizationResult:
    """Results from energy optimization."""
    total_energy_Wh: float
    total_time_s: float
    km_per_kWh: float
    speed_profile: np.ndarray
    throttle_profile: np.ndarray
    distance: np.ndarray
    gas_start_positions: List[float]
    gas_stop_positions: List[float]


class GoldenStandardOptimizer:
    """
    Energy optimizer with Golden Standard visualizations.
    
    Produces outputs matching the reference image:
    - GPS Track with throttle probability colormap
    - Recommended Speed Profile with target zone
    - Gas Start/Stop markers
    - Per-lap statistics comparison
    """
    
    def __init__(self, vehicle: VehicleParams = None):
        self.vehicle = vehicle or VehicleParams()
        self.laps: List[LapData] = []
        self.model_version = "unknown"
        
    def load_narx_model(self, mat_path: Path) -> bool:
        """Load NARX model predictions from .mat file."""
        try:
            mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
            
            # Detect model version
            if 'V3' in mat_path.name:
                self.model_version = "V3"
            else:
                self.model_version = "V2"
            
            print(f"  Loading {self.model_version} model from {mat_path.name}")
            
            # Extract lap_info structure
            if 'lap_info' in mat:
                lap_info = mat['lap_info']
                if not hasattr(lap_info, '__len__'):
                    lap_info = [lap_info]
                
                self.laps = []
                for idx, lap in enumerate(lap_info):
                    try:
                        lap_data = self._extract_lap_data(lap, idx)
                        if lap_data is not None:
                            self.laps.append(lap_data)
                    except Exception as e:
                        print(f"    Warning: Could not load lap {idx}: {e}")
                
                print(f"  Loaded {len(self.laps)} laps from {self.model_version} model")
                return True
            
            print("  Warning: lap_info not found in model")
            return False
            
        except Exception as e:
            print(f"  Error loading NARX model: {e}")
            return False
    
    def _extract_lap_data(self, lap, idx: int) -> Optional[LapData]:
        """Extract lap data from MATLAB structure."""
        try:
            # Required fields
            distance = np.atleast_1d(np.array(lap.distance)).flatten()
            speed = np.atleast_1d(np.array(lap.speed)).flatten()
            
            # GPS coordinates (optional)
            lat = np.atleast_1d(getattr(lap, 'lat', np.zeros(len(speed)))).flatten()
            lng = np.atleast_1d(getattr(lap, 'lng', np.zeros(len(speed)))).flatten()
            
            # Check if GPS is valid
            has_gps = np.any((lat != 0) & (lng != 0))
            
            # Speed bounds
            speed_upper = np.atleast_1d(getattr(lap, 'speed_upper', speed * 1.2)).flatten()
            speed_lower = np.atleast_1d(getattr(lap, 'speed_lower', speed * 0.8)).flatten()
            
            # Throttle state
            throttle_state = np.atleast_1d(getattr(lap, 'throttle_state', np.zeros(len(speed)))).flatten()
            
            # Throttle/gliding event distances
            throttle_distances = np.atleast_1d(getattr(lap, 'throttle_distances', np.array([]))).flatten()
            gliding_distances = np.atleast_1d(getattr(lap, 'gliding_distances', np.array([]))).flatten()
            
            # Lap metrics
            total_distance = float(getattr(lap, 'total_distance', distance[-1] if len(distance) > 0 else 0))
            lap_time = float(getattr(lap, 'lap_time', 200.0))
            lap_energy = float(getattr(lap, 'lap_energy', 5.0))  # Wh
            
            # Source info
            lap_number = int(getattr(lap, 'lap_number', idx + 1))
            source = str(getattr(lap, 'source', 'unknown'))
            
            return LapData(
                lap_number=lap_number,
                source=source,
                lat=lat,
                lng=lng,
                speed=speed,
                distance=distance,
                throttle_state=throttle_state,
                speed_upper=speed_upper,
                speed_lower=speed_lower,
                total_distance=total_distance,
                lap_time=lap_time,
                lap_energy=lap_energy,
                has_gps=has_gps,
                throttle_distances=throttle_distances,
                gliding_distances=gliding_distances,
            )
        except Exception as e:
            print(f"    Error extracting lap {idx}: {e}")
            return None
    
    def compute_throttle_probability(self, lap: LapData) -> np.ndarray:
        """
        Compute throttle probability based on speed bounds and actual speed.
        
        Higher probability when:
        - Speed is near lower bound (need to accelerate)
        - Near throttle event locations
        """
        n = len(lap.speed)
        prob = np.zeros(n)
        
        # Based on position relative to speed bounds
        speed_range = lap.speed_upper - lap.speed_lower
        speed_range[speed_range < 1] = 1  # Avoid division by zero
        
        # Probability increases when speed is low in the window
        relative_pos = (lap.speed - lap.speed_lower) / speed_range
        prob = 1.0 - np.clip(relative_pos, 0, 1)
        
        # Boost probability near throttle event locations
        if lap.throttle_distances is not None and len(lap.throttle_distances) > 0:
            for td in lap.throttle_distances:
                # Find closest point
                dist_to_event = np.abs(lap.distance - td)
                influence = np.exp(-dist_to_event / 50)  # 50m influence radius
                prob = np.maximum(prob, influence * 0.9)
        
        # Use actual throttle state if available
        if np.any(lap.throttle_state > 0):
            prob = 0.5 * prob + 0.5 * lap.throttle_state
        
        # Smooth the probability
        prob = gaussian_filter1d(prob, sigma=3)
        
        return np.clip(prob, 0, 1)
    
    def identify_gas_events(self, lap: LapData) -> Tuple[List[float], List[float]]:
        """
        Identify recommended gas start and stop positions.
        
        Returns: (gas_start_positions, gas_stop_positions)
        """
        if lap.throttle_distances is not None and len(lap.throttle_distances) > 0:
            gas_starts = list(lap.throttle_distances)
        else:
            # Estimate from throttle state changes
            gas_starts = []
            throttle_state = lap.throttle_state
            for i in range(1, len(throttle_state)):
                if throttle_state[i] > 0.5 and throttle_state[i-1] <= 0.5:
                    gas_starts.append(lap.distance[i])
        
        if lap.gliding_distances is not None and len(lap.gliding_distances) > 0:
            gas_stops = list(lap.gliding_distances)
        else:
            # Estimate from throttle state changes
            gas_stops = []
            throttle_state = lap.throttle_state
            for i in range(1, len(throttle_state)):
                if throttle_state[i] <= 0.5 and throttle_state[i-1] > 0.5:
                    gas_stops.append(lap.distance[i])
        
        return gas_starts, gas_stops
    
    def optimize_lap(self, lap: LapData, time_weight: float = 0.5) -> OptimizationResult:
        """Optimize a single lap for minimum energy."""
        n = len(lap.distance)
        
        # Optimal speed within bounds
        speed_opt = np.zeros(n)
        for i in range(n):
            v_min = lap.speed_lower[i] / 3.6
            v_max = lap.speed_upper[i] / 3.6
            
            # Find optimal speed balancing energy and time
            v_opt = v_min + (v_max - v_min) * (1 - time_weight)
            speed_opt[i] = v_opt * 3.6
        
        # Compute throttle profile
        throttle_prob = self.compute_throttle_probability(lap)
        
        # Identify gas events
        gas_starts, gas_stops = self.identify_gas_events(lap)
        
        # Compute total energy and time
        dt = np.diff(lap.distance) / (speed_opt[:-1] / 3.6 + 1e-6)
        total_time = np.sum(dt)
        
        # Energy computation
        v_mps = speed_opt / 3.6
        P_drag = 0.5 * self.vehicle.rho_air * self.vehicle.Cd * self.vehicle.A_frontal * v_mps**3
        P_roll = self.vehicle.Crr * self.vehicle.mass * self.vehicle.g * v_mps
        P_total = (P_drag + P_roll) / self.vehicle.motor_efficiency
        
        # Only count energy when throttle is on
        P_actual = P_total * throttle_prob
        energy_j = np.sum(P_actual[:-1] * dt)
        energy_wh = energy_j / 3600
        
        # Compute efficiency
        dist_km = lap.total_distance / 1000
        if energy_wh > 0:
            km_per_kwh = dist_km / (energy_wh / 1000)
        else:
            km_per_kwh = float('inf')
        
        return OptimizationResult(
            total_energy_Wh=energy_wh,
            total_time_s=total_time,
            km_per_kWh=km_per_kwh,
            speed_profile=speed_opt,
            throttle_profile=throttle_prob,
            distance=lap.distance,
            gas_start_positions=gas_starts,
            gas_stop_positions=gas_stops,
        )


def plot_golden_standard(
    optimizer: GoldenStandardOptimizer,
    lap_idx: int = 0,
    save_dir: Path = None,
    target_time_s: float = 211.1
) -> plt.Figure:
    """
    Generate Golden Standard visualization matching the reference image.
    
    Layout:
    - Top-left: Recommended Speed Profile with Target Zone
    - Top-right: Lap Analysis (track, speed strategy, speed vs distance by throttle)
    - Bottom-left: GPS Track with Throttle Probability and Gas Start/Stop
    - Bottom-right: Lap statistics comparison (time, energy, distance)
    """
    if len(optimizer.laps) == 0:
        print("  No laps loaded!")
        return None
    
    lap = optimizer.laps[min(lap_idx, len(optimizer.laps) - 1)]
    opt_result = optimizer.optimize_lap(lap)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Custom colormap for throttle probability (blue -> yellow -> red)
    cmap_throttle = LinearSegmentedColormap.from_list(
        'throttle_prob', 
        [(0.0, 0.0, 0.8), (0.3, 0.3, 1.0), (1.0, 1.0, 0.3), (1.0, 0.6, 0.0), (1.0, 0.0, 0.0)]
    )
    
    # ========== TOP LEFT: Recommended Speed Profile with Target Zone ==========
    ax1 = fig.add_subplot(2, 2, 1)
    
    dist_m = lap.distance
    
    # Fill target zone
    ax1.fill_between(dist_m, lap.speed_lower, lap.speed_upper, 
                     alpha=0.4, color='cyan', label='Target Zone')
    ax1.plot(dist_m, lap.speed_upper, 'g-', linewidth=2, label='Upper Limit')
    ax1.plot(dist_m, lap.speed_lower, 'b-', linewidth=2, label='Lower Limit')
    ax1.plot(dist_m, lap.speed, 'k-', linewidth=1.5, alpha=0.7, label='Actual Speed')
    
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title(f'Recommended Speed Profile (Target: {target_time_s:.1f}s)')
    ax1.legend(loc='upper right')
    ax1.set_xlim([0, lap.distance[-1]])
    
    # ========== TOP RIGHT: Lap Analysis Panel (3 subplots) ==========
    
    # Track map
    ax2a = fig.add_subplot(2, 4, 3)
    if lap.has_gps and np.any(lap.lat != 0):
        throttle_prob = optimizer.compute_throttle_probability(lap)
        scatter = ax2a.scatter(lap.lng, lap.lat, c=throttle_prob, cmap=cmap_throttle, 
                              s=15, alpha=0.8)
        
        # Add gas start/stop markers
        gas_starts, gas_stops = optimizer.identify_gas_events(lap)
        for gs in gas_starts:
            idx = np.argmin(np.abs(lap.distance - gs))
            if idx < len(lap.lng):
                ax2a.plot(lap.lng[idx], lap.lat[idx], 'o', color='lime', markersize=10, 
                         markeredgecolor='darkgreen', markeredgewidth=2)
        for gstop in gas_stops:
            idx = np.argmin(np.abs(lap.distance - gstop))
            if idx < len(lap.lng):
                ax2a.plot(lap.lng[idx], lap.lat[idx], 's', color='red', markersize=8,
                         markeredgecolor='darkred', markeredgewidth=2)
        
        ax2a.set_xlabel('Longitude')
        ax2a.set_ylabel('Latitude')
        ax2a.set_aspect('equal', 'box')
    else:
        ax2a.text(0.5, 0.5, 'No GPS Data\n(Speed Integration Used)', 
                 ha='center', va='center', transform=ax2a.transAxes, fontsize=10)
    
    ax2a.set_title(f'Lap {lap.lap_number}: Track & Strategy')
    
    # Speed strategy window
    ax2b = fig.add_subplot(2, 4, 4)
    dist_km = lap.distance / 1000
    ax2b.fill_between(dist_km, lap.speed_lower, lap.speed_upper, alpha=0.3, color='lightgreen')
    ax2b.plot(dist_km, lap.speed_upper, 'g-', linewidth=1.5)
    ax2b.plot(dist_km, lap.speed_lower, 'g-', linewidth=1.5)
    ax2b.plot(dist_km, lap.speed, 'k-', linewidth=1.5)
    ax2b.set_xlabel('Distance (km)')
    ax2b.set_ylabel('Speed (km/h)')
    ax2b.set_title('Speed Strategy Window')
    ax2b.set_xlim([0, max(dist_km)])
    
    # Speed vs distance colored by throttle
    ax2c = fig.add_subplot(2, 4, 8)
    throttle_prob = optimizer.compute_throttle_probability(lap)
    scatter2 = ax2c.scatter(dist_km, lap.speed, c=throttle_prob, cmap=cmap_throttle, 
                           s=20, alpha=0.8)
    ax2c.set_xlabel('Distance (km)')
    ax2c.set_ylabel('Speed (km/h)')
    ax2c.set_title('Speed vs Distance (colored by Throttle)')
    cbar = plt.colorbar(scatter2, ax=ax2c, shrink=0.8)
    cbar.set_label('Throttle Prob.')
    
    # ========== BOTTOM LEFT: GPS Track with Throttle Probability ==========
    ax3 = fig.add_subplot(2, 2, 3)
    
    if lap.has_gps and np.any(lap.lat != 0):
        throttle_prob = optimizer.compute_throttle_probability(lap)
        scatter3 = ax3.scatter(lap.lng, lap.lat, c=throttle_prob, cmap=cmap_throttle,
                              s=30, alpha=0.8)
        
        # Add gas start markers (green circles)
        gas_starts, gas_stops = optimizer.identify_gas_events(lap)
        
        start_plotted = False
        stop_plotted = False
        
        for gs in gas_starts:
            idx = np.argmin(np.abs(lap.distance - gs))
            if idx < len(lap.lng):
                label = 'Rec. Gas Start' if not start_plotted else None
                ax3.plot(lap.lng[idx], lap.lat[idx], 'o', color='lime', markersize=14, 
                        markeredgecolor='darkgreen', markeredgewidth=3, label=label)
                start_plotted = True
        
        # Add gas stop markers (red X)
        for gstop in gas_stops:
            idx = np.argmin(np.abs(lap.distance - gstop))
            if idx < len(lap.lng):
                label = 'Rec. Gas Stop' if not stop_plotted else None
                ax3.plot(lap.lng[idx], lap.lat[idx], 'X', color='red', markersize=12,
                        markeredgecolor='darkred', markeredgewidth=2, label=label)
                stop_plotted = True
        
        # Colorbar
        cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
        cbar3.set_label('Throttle Probability')
        
        # Legend
        ax3.legend(loc='upper right')
        
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.set_aspect('equal', 'box')
    else:
        # Fallback: distance-based visualization
        throttle_prob = optimizer.compute_throttle_probability(lap)
        scatter3 = ax3.scatter(lap.distance, lap.speed, c=throttle_prob, 
                              cmap=cmap_throttle, s=30, alpha=0.8)
        
        # Add gas markers on distance axis
        gas_starts, gas_stops = optimizer.identify_gas_events(lap)
        for gs in gas_starts:
            idx = np.argmin(np.abs(lap.distance - gs))
            if idx < len(lap.speed):
                ax3.plot(lap.distance[idx], lap.speed[idx], 'o', color='lime', 
                        markersize=12, markeredgecolor='darkgreen', markeredgewidth=2)
        for gstop in gas_stops:
            idx = np.argmin(np.abs(lap.distance - gstop))
            if idx < len(lap.speed):
                ax3.plot(lap.distance[idx], lap.speed[idx], 'X', color='red', 
                        markersize=10, markeredgecolor='darkred', markeredgewidth=2)
        
        cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
        cbar3.set_label('Throttle Probability')
        ax3.set_xlabel('Distance (m)')
        ax3.set_ylabel('Speed (km/h)')
    
    ax3.set_title(f'Recommended Throttle Strategy on Track (Based on Lap {lap.lap_number} Geometry)')
    
    # ========== BOTTOM RIGHT: Lap Statistics Comparison ==========
    ax4a = fig.add_subplot(2, 4, 7)
    ax4b = fig.add_subplot(2, 6, 11)
    ax4c = fig.add_subplot(2, 6, 12)
    
    # Prepare data for all laps
    if len(optimizer.laps) > 1:
        lap_numbers = [l.lap_number for l in optimizer.laps]
        lap_times = [l.lap_time for l in optimizer.laps]
        lap_energies = [l.lap_energy for l in optimizer.laps]  # Wh
        lap_distances = [l.total_distance / 1000 for l in optimizer.laps]
        colors = ['#4a90d9'] * len(lap_numbers)
        # Highlight current lap in red
        if lap_idx < len(colors):
            colors[lap_idx] = 'red'
    else:
        # Single lap comparison (actual vs optimized)
        lap_numbers = [lap.lap_number, 'Opt']
        lap_times = [lap.lap_time, opt_result.total_time_s]
        lap_energies = [lap.lap_energy, opt_result.total_energy_Wh]
        lap_distances = [lap.total_distance / 1000, lap.total_distance / 1000]
        colors = ['#4a90d9', 'red']
    
    # Lap time comparison
    bars1 = ax4a.bar(range(len(lap_times)), lap_times, color=colors)
    ax4a.set_xlabel('Lap')
    ax4a.set_ylabel('Time (s)')
    ax4a.set_title('Lap Time Comparison')
    ax4a.set_xticks(range(len(lap_numbers)))
    ax4a.set_xticklabels([str(x) for x in lap_numbers], fontsize=8)
    
    # Energy comparison (in kJ for visibility)
    energy_kj = [e * 3.6 for e in lap_energies]  # Convert Wh to kJ
    bars2 = ax4b.bar(range(len(energy_kj)), energy_kj, color=colors)
    ax4b.set_ylabel('Energy (kJ)')
    ax4b.set_title('Energy (kJ)')
    ax4b.set_xticks(range(len(lap_numbers)))
    ax4b.set_xticklabels([str(x) for x in lap_numbers], fontsize=8)
    
    # Distance comparison
    bars3 = ax4c.bar(range(len(lap_distances)), lap_distances, color=colors)
    ax4c.set_ylabel('Distance (km)')
    ax4c.set_title('Distance (km)')
    ax4c.set_xticks(range(len(lap_numbers)))
    ax4c.set_xticklabels([str(x) for x in lap_numbers], fontsize=8)
    
    # Add super title
    energy_wh = lap.lap_energy
    energy_kj_total = energy_wh * 3.6
    fig.suptitle(
        f'Lap {lap.lap_number} Analysis | Time: {lap.lap_time:.1f}s | Energy: {energy_kj_total:.1f}kJ',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f'golden_standard_lap{lap.lap_number}.png'
        fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_dir / filename}")
    
    return fig


def plot_multi_lap_comparison(
    optimizer: GoldenStandardOptimizer,
    save_dir: Path = None
) -> plt.Figure:
    """Generate multi-lap comparison chart."""
    if len(optimizer.laps) == 0:
        print("  No laps loaded!")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    
    lap_numbers = [l.lap_number for l in optimizer.laps]
    lap_times = [l.lap_time for l in optimizer.laps]
    lap_energies = [l.lap_energy for l in optimizer.laps]  # Wh
    lap_distances = [l.total_distance for l in optimizer.laps]
    
    # Compute efficiencies
    efficiencies = []
    for l in optimizer.laps:
        if l.lap_energy > 0:
            eff = (l.total_distance / 1000) / (l.lap_energy / 1000)  # km/kWh
        else:
            eff = 0
        efficiencies.append(eff)
    
    # Sources for color coding
    sources = [l.source for l in optimizer.laps]
    unique_sources = list(set(sources))
    colors_list = plt.cm.tab10(np.linspace(0, 1, max(len(unique_sources), 1)))
    color_map = {s: colors_list[i] for i, s in enumerate(unique_sources)}
    bar_colors = [color_map[s] for s in sources]
    
    # Lap times
    ax = axes[0, 0]
    ax.bar(range(len(lap_times)), lap_times, color=bar_colors)
    ax.set_xlabel('Lap')
    ax.set_ylabel('Time (s)')
    ax.set_title('Lap Times')
    ax.set_xticks(range(len(lap_numbers)))
    ax.set_xticklabels(lap_numbers)
    
    # Energy consumption
    ax = axes[0, 1]
    ax.bar(range(len(lap_energies)), lap_energies, color=bar_colors)
    ax.set_xlabel('Lap')
    ax.set_ylabel('Energy (Wh)')
    ax.set_title('Energy Consumption')
    ax.set_xticks(range(len(lap_numbers)))
    ax.set_xticklabels(lap_numbers)
    
    # Distances
    ax = axes[0, 2]
    ax.bar(range(len(lap_distances)), [d/1000 for d in lap_distances], color=bar_colors)
    ax.set_xlabel('Lap')
    ax.set_ylabel('Distance (km)')
    ax.set_title('Lap Distances')
    ax.set_xticks(range(len(lap_numbers)))
    ax.set_xticklabels(lap_numbers)
    
    # Efficiency (km/kWh)
    ax = axes[1, 0]
    ax.bar(range(len(efficiencies)), efficiencies, color=bar_colors)
    ax.set_xlabel('Lap')
    ax.set_ylabel('km/kWh')
    ax.set_title('Energy Efficiency')
    ax.set_xticks(range(len(lap_numbers)))
    ax.set_xticklabels(lap_numbers)
    
    # Speed distribution
    ax = axes[1, 1]
    for i, lap in enumerate(optimizer.laps):
        ax.plot(lap.distance/1000, lap.speed, label=f'Lap {lap.lap_number}', alpha=0.7)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Speed Profiles')
    if len(optimizer.laps) <= 6:
        ax.legend(fontsize=8)
    
    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = f"""
    MODEL: NARX {optimizer.model_version}
    {'='*35}
    
    Total Laps:     {len(optimizer.laps)}
    Total Distance: {sum(lap_distances)/1000:.2f} km
    
    Lap Time:
      Min: {min(lap_times):.1f}s  Max: {max(lap_times):.1f}s
      Avg: {np.mean(lap_times):.1f}s
    
    Energy:
      Min: {min(lap_energies):.1f} Wh  Max: {max(lap_energies):.1f} Wh
      Avg: {np.mean(lap_energies):.1f} Wh
    
    Efficiency:
      Min: {min(efficiencies):.0f} km/kWh
      Max: {max(efficiencies):.0f} km/kWh
      Avg: {np.mean(efficiencies):.0f} km/kWh
    
    Data Sources: {', '.join(unique_sources)}
    """
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Legend for sources
    if len(unique_sources) > 1:
        legend_elements = [plt.Rectangle((0,0), 1, 1, color=color_map[s], label=s) 
                          for s in unique_sources]
        fig.legend(handles=legend_elements, loc='upper center', ncol=len(unique_sources),
                  bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle(f'NARX {optimizer.model_version} - Multi-Lap Comparison', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f'multi_lap_comparison_{optimizer.model_version}.png'
        fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_dir / filename}")
    
    return fig


def main():
    print("=" * 70)
    print("  NARX-Enhanced Energy Optimization - Golden Standard Output")
    print("  Supports NARX V2 (SEM_Model_Final) and V3 models")
    print("=" * 70)
    
    # Create output directories
    FIGURE_DIR_V2.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR_V3.mkdir(parents=True, exist_ok=True)
    
    # ======================== PROCESS V2 MODEL ========================
    print("\n" + "="*50)
    print("  Processing NARX V2 (NARX_SEM_Model_Final.mat)")
    print("="*50)
    
    optimizer_v2 = GoldenStandardOptimizer()
    model_v2_path = MODEL_DIR / "NARX_SEM_Model_Final.mat"
    
    if model_v2_path.exists():
        if optimizer_v2.load_narx_model(model_v2_path):
            # Generate golden standard for each lap (first 3 laps max)
            num_laps = min(3, len(optimizer_v2.laps))
            for i in range(num_laps):
                plot_golden_standard(optimizer_v2, lap_idx=i, save_dir=FIGURE_DIR_V2)
            
            # Multi-lap comparison
            plot_multi_lap_comparison(optimizer_v2, save_dir=FIGURE_DIR_V2)
            
            print(f"\n  V2 figures saved to: {FIGURE_DIR_V2}")
        else:
            print("  Failed to load V2 model")
    else:
        print(f"  Warning: V2 model not found at {model_v2_path}")
    
    # ======================== PROCESS V3 MODEL ========================
    print("\n" + "="*50)
    print("  Processing NARX V3 (NARX_V3_Model.mat)")
    print("="*50)
    
    optimizer_v3 = GoldenStandardOptimizer()
    model_v3_path = MODEL_DIR / "NARX_V3_Model.mat"
    
    if model_v3_path.exists():
        if optimizer_v3.load_narx_model(model_v3_path):
            # Generate golden standard for each lap (first 3 laps max)
            num_laps = min(3, len(optimizer_v3.laps))
            for i in range(num_laps):
                plot_golden_standard(optimizer_v3, lap_idx=i, save_dir=FIGURE_DIR_V3)
            
            # Multi-lap comparison
            plot_multi_lap_comparison(optimizer_v3, save_dir=FIGURE_DIR_V3)
            
            print(f"\n  V3 figures saved to: {FIGURE_DIR_V3}")
        else:
            print("  Failed to load V3 model")
    else:
        print(f"  Warning: V3 model not found at {model_v3_path}")
    
    # ======================== SUMMARY ========================
    print("\n" + "="*70)
    print("  OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\n  Output directories:")
    print(f"    V2: {FIGURE_DIR_V2}")
    print(f"    V3: {FIGURE_DIR_V3}")
    print("\n  Generated files:")
    print("    - golden_standard_lapX.png  (per-lap analysis)")
    print("    - multi_lap_comparison_VX.png  (summary)")
    print("="*70)
    
    plt.show()


if __name__ == "__main__":
    main()
