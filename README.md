# LongitudinalTomography.jl

A Julia research prototype for longitudinal beam dynamics and phase-space
tomography from turn-by-turn resistive-wall-current-monitor (RWCM) profiles,
developed with the Fermilab Recycler and slip stacking in mind.

## Status

- **Single RF:** Analytical transport of particles trapped inside a stationary
  bucket, differentiable soft-histogram RWCM profiles, and grid tomography using
  a precomputed response matrix. Grid reconstruction fits nonnegative cell
  weights with optional total-variation and entropy penalties.
- **Slip stacking:** A noninteracting two-RF model transports RF1 and RF2 bunches
  in their own buckets and places them on one unwrapped RWCM time axis. The
  combined grid response supports independent distributions for multiple
  bunches and has a reconstruction path. Tests currently cover the combined
  forward response, charge conservation, and its gradient.

This is a synthetic-data research tool, not a calibrated experimental
reconstruction pipeline. The two RF families do not perturb one another; the
model does not include coupled two-RF dynamics, capture or loss, or an RWCM
transfer function. Analytical transport requires particles strictly inside a
stationary separatrix.

## Get started

Use Julia 1.10 or newer from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Load the local package with `julia --project=.` followed by
`using LongitudinalTomography`. The reusable API is in [`src/`](src/); tests are
in [`test/`](test/). For worked experiments, see the
[single-RF forward example](julia_notebooks/singleRF/01_synthetic_forward_model.jl),
[single-RF grid tomography notebooks](julia_notebooks/singleRF_grid_tomo/), and
[noninteracting two-RF notebooks](julia_notebooks/dualRF_grid_tomo_no_interaction/).
