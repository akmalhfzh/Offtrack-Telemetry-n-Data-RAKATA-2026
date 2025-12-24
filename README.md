# Telemetry-Driven Digital Twin Modeling with ML-Enhanced Energy Optimization

**Shell Eco-Marathon Off-Track Competition**

## Overview

This repository contains the complete pipeline for optimizing energy efficiency in Shell Eco-Marathon (SEM) vehicles using machine learning-based telemetry analysis.

### Key Components

1. **NARX Neural Network** - Learns optimal driving strategy from successful laps
2. **Energy Optimization** - Uses NARX predictions for energy-efficient driving
3. **Telemetry Analysis** - Extracts insights from vehicle sensor data
4. **Validation Tools** - Generates publication-quality figures and metrics

## Repository Structure

```
├── Data/
│   └── raw/
│       ├── 26novfiltered.csv      # Telemetry data (GPS, speed, current, throttle)
│       ├── Logbook_fixed.csv      # Lap boundaries and manual annotations
│       └── Logbook_TD_26_10.csv   # Original logbook
│
├── NN/
│   ├── NARX_V2.m                  # Main NARX training script (MATLAB)
│   ├── NARX_SEM_Model_Final.mat   # Trained model
│   └── training_output/           # Training results and figures
│
├── Analysis/
│   └── telemetry_analysis.py      # Telemetry visualization and statistics
│
├── Optimization/
│   └── energy_optimizer.py        # NARX-based energy optimization
│
├── Validation/
│   └── validate_narx.py           # Model validation and paper figures
│
└── Paper/
    └── figures/                   # Publication-ready figures
```

## NARX Model Architecture

### Input Features (9 per segment)
| Feature | Description |
|---------|-------------|
| `distance` | Cumulative distance in segment (m) |
| `slope` | Track slope angle (degrees) |
| `curvature` | Track curvature (1/m) |
| `lap_time` | Target lap time (s) |
| `lap_energy` | Target energy budget (Wh) |
| `max_speed` | Maximum achievable speed (km/h) |
| `aggressiveness` | Driving style factor (0-1) |
| `prev_throttle` | Previous segment throttle state |
| `prev_gliding` | Previous segment gliding state |

### Output (3 per segment)
| Output | Description |
|--------|-------------|
| `speed_upper` | Maximum speed bound (km/h) |
| `speed_lower` | Minimum speed bound (km/h) |
| `throttle_ratio` | Fraction of segment with throttle active |

### Training Results
- **Best MSE**: 0.018854 (epoch 3)
- **Training epochs**: 18
- **Hidden neurons**: 15
- **Delays**: 1:3 (input and feedback)

## Quick Start

### 1. Train NARX Model (MATLAB)
```matlab
cd NN
NARX_V2  % Generates NARX_SEM_Model_Final.mat
```

### 2. Analyze Telemetry (Python)
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run analysis
python Analysis/telemetry_analysis.py
```

### 3. Validate Model
```bash
python Validation/validate_narx.py
```

### 4. Run Optimization
```bash
python Optimization/energy_optimizer.py
```

## Performance Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Best km/kWh | 150 | 300+ |
| Lap Time | 3:20 | 3:00 |
| Energy/Lap | 8-23 Wh | <8 Wh |

## How NARX Improves Efficiency

1. **Speed Bounds** - Predicts optimal speed range for each track segment
2. **Throttle Timing** - Learns where to accelerate vs glide
3. **Energy Prediction** - Estimates energy consumption for strategy planning

## Next Steps for Competition

1. ✅ Train NARX on existing data
2. ⏳ Collect more laps following NARX strategy
3. ⏳ Retrain with expanded dataset
4. ⏳ Hardware optimization (tires, aero, weight)
5. ⏳ Competition practice runs

## Requirements

### MATLAB
- MATLAB R2020a or later
- Deep Learning Toolbox
- Signal Processing Toolbox

### Python
- Python 3.8+
- See `requirements.txt`

## Authors

RAKATA Team - Shell Eco-Marathon 2025

## License

MIT License
