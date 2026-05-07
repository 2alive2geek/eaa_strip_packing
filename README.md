# Two-Dimensional Strip Packing (2SP) Solver 

This repository contains a solver for the 2SP packing problem. Multiple solutions will be implemented in order to find the best performing algorithm (i.e. exact, heuristic approaches, metaheuristic approaches). 

If time will be on my side, I will try implementing a prediction model which shall estimate the algorithm that can find the best solution (time / optimality) for different benchmarking instances.

## Project Structure

| File | Description |
|---|---|
| `benchmarks.py` | Generates benchmark instances inspired by Bengtsson (1982), Berkey & Wang (1987), and Martello & Vigo (1998) |
| `solver_exact.py` | Exact solver using OR-Tools CP-SAT (constraint programming with NoOverlap2D) |
| `solver_heuristic.py` | Bottom-Left-Fill (BLF) heuristic with multiple sorting strategies (DH, DW, DA, DP) |
| `main.py` | Experiment runner — compares both solvers on all benchmarks and outputs results |

## Usage

```bash
# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Run the full experiment
cd eaa_strip_packing
python main.py

# Options
python main.py --exact-limit 20   # run exact solver on instances with n <= 20
python main.py --time-limit 120   # CP-SAT time limit in seconds
python main.py --seed 42          # random seed for benchmark generation
```

Results are printed as tables and saved to `results.csv`.