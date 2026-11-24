# AI Data Center Tripping and Grid Frequency Risk

Code and data for the preprint "Rare-Event RoCoF Risk of Correlated AI Data Center Tripping:
A Surrogate-Assisted Subset Simulation Study" by Mir Ishfak Faisal Rafi.

Archived version: https://doi.org/10.5281/zenodo.24489890
Preprint: https://doi.org/10.5281/zenodo.24491676

## Requirements

Python 3.11 or newer and the packages in requirements.txt (ANDES 2.0.0, numpy, pandas,
scipy, scikit-learn, matplotlib).

    pip install -r requirements.txt

## How to run

Run these in order from this folder:

    python ai_load_model.py
    python scenarios.py
    python plot_results.py
    python run_rare.py
    python analyze_rare.py
    python run_low_inertia.py
    python sensitivity.py
    python make_figs.py

The first three scripts cover the single-campus study, the next four the rare-event study,
and the last one makes the paper figures. The full run takes about 30-60 minutes.

## Notes

- The result files from the paper are included, so you can check the numbers without
  rerunning anything.
- The rare-event results are random, so a rerun gives slightly different probabilities.
- You can ignore scikit-learn convergence warnings, or hide them with `python -W ignore`.

## License

MIT
