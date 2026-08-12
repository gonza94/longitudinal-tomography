# LongitudinalTomography.jl

A Julia prototype for differentiable longitudinal phase-space tomography from
turn-by-turn resistive-wall-current-monitor profiles.

This repository uses a hybrid layout:

- `src/LongitudinalTomography.jl` contains reusable physics and measurement
  operators.
- `notebooks/01_synthetic_forward_model.jl` is a VS Code-friendly experiment
  with `# %%` cells.
- `test/runtests.jl` verifies trapping, exact transport, charge conservation,
  and an end-to-end automatic-differentiation gradient.

## Phase 1 scope

The current forward model assumes:

- one stationary RF system;
- constant Recycler parameters;
- particles strictly inside the separatrix;
- exact nonlinear libration transport using Jacobi elliptic functions;
- a Gaussian differentiable soft histogram;
- normalized profiles with no RWCM transfer function or noise yet.

The synthetic example generates 4,096 particles, 51 profiles from turn 0 to
250, and 75 time bins spanning one RF bucket.

## Setup on macOS

From a terminal in this directory:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Then open `notebooks/01_synthetic_forward_model.jl` in VS Code and execute the
`# %%` cells with the Julia extension.

The first CairoMakie compilation can take a little while. Subsequent runs are
much faster.

## Key API

```julia
theta0, P0 = trapped_coordinates(u, v)

cascade = predict_rwcm_cascade(
    theta0,
    P0,
    turns,
    machine,
    time_bin_centers_s;
    sigma=soft_sigma_s,
)
```

`cascade.profiles` has shape `(n_turns, n_time_bins)` and each row sums to one.
The implementation avoids hard bin assignment and array mutation in the
differentiated forward path.

