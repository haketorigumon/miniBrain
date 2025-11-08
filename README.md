# miniBrain

A Python simulation and analysis toolkit for exploring bistable neural dynamics, global workspace theory, and reflective hierarchy models. Includes real-time visualization, complexity diagnostics, meta-autotuning, and machine learning integration for advanced metrics and prediction.

## Features

- **Bistable Layer Simulation**: Models neural units with bistable dynamics.
- **Global Workspace & Reflective Hierarchy**: Compare two cognitive architectures.
- **Real-Time Visualization**: Animated heatmaps and phase coherence charts.
- **Parameter Sweep & Auto-Tuning**: Explore and optimize parameters for high complexity and phase-locking.
- **Complexity Metrics**: Shannon entropy, Lyapunov proxy, Lempel-Ziv complexity, mutual information.
- **Perturbation Analysis**: Quantify system sensitivity and robustness.
- **Meta-Autotuner**: PyTorch neural network for adaptive parameter control.
- **Workspace Decoder & Forward Predictor**: Machine learning models (Ridge regression, neural net) for decoding and prediction.
- **Comprehensive Diagnostics**: Tables and charts for all metrics, including mutual information matrices and perturbation deltas.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BrandonRaeder/miniBrain.git
   cd miniBrain
   ```
2. (Recommended) Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main simulation and analysis script:
```bash
python live_phase_aurora.py
```

- The script will auto-tune parameters, run real-time visualizations, and print/plot all metrics and tables.
- Modify parameters in `live_phase_aurora.py` for custom experiments.

## Key Files

- `live_phase_aurora.py`: Main simulation, visualization, diagnostics, and ML integration.
- `requirements.txt`: Python dependencies (numpy, matplotlib, torch, scikit-learn, pandas).

## Example Output

- Animated heatmaps of neuron phases
- Phase coherence charts for both models
- Parameter sweep visualizations
- Complexity and mutual information tables
- Perturbation sensitivity analysis

## Extending

- Add new models or metrics by editing `live_phase_aurora.py`.
- Integrate additional ML models for prediction or decoding.
- Use the provided functions for custom analysis and visualization.

## Citation
If you use miniBrain in your research, please cite the repository and reference the original authors.

## License
MIT License

---

For questions, issues, or contributions, please open an issue or pull request on GitHub.
