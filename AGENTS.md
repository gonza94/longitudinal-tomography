# AGENTS.md

## Purpose

This repository develops differentiable longitudinal beam-dynamics and tomography tools in Julia, with the Fermilab Recycler and slip stacking as the main application.

The current implementation has three foundations:

1. Exact analytical transport inside a stationary single-RF separatrix.
2. Grid tomography using a precomputed differentiable soft-histogram response matrix.
3. A noninteracting two-RF approximation in which the RF1 and RF2 bucket families evolve independently and RF2 is translated into the RF1 frame.

The immediate development goal is to extend grid tomography to one or more slip-stacked bunches from both RF families. Preserve the validated single-RF behavior while doing this.

## Start Here

Before changing code, inspect:

- `Project.toml`
- `src/LongitudinalTomography.jl`
- `src/grid_tomography.jl`
- `src/two_rf_noninteracting.jl`
- `test/runtests.jl`
- `test/grid_tomography.jl`
- `test/two_rf_noninteracting.jl`
- the notebook closest to the requested task under `julia_notebooks/`

Do not infer an API from a notebook when the source defines something different. Treat the package source and passing tests as authoritative.

## Repository Organization

- Put reusable library code in `src/`.
- Keep `src/LongitudinalTomography.jl` as the top-level module, include new source files there, and export only intentional public APIs.
- Put automated tests in `test/` and include new test files from `test/runtests.jl`.
- Use `julia_notebooks/` for demonstrations, plots, parameter scans, and end-to-end experiments. Do not make package behavior depend on notebook state.
- Do not manually edit `Manifest.toml`. Use `Pkg` through the project environment.
- Avoid adding a dependency when a small, clear implementation using Base Julia or an existing dependency is sufficient.

Recommended naming for the next feature:

- source: `src/slip_stacked_grid_tomography.jl`
- tests: `test/slip_stacked_grid_tomography.jl`
- notebook: `julia_notebooks/dualRF_no_interaction/slip_stacked_grid_tomography.ipynb`

Names may change if the existing architecture suggests a cleaner dispatch-based extension.

## Julia Environment

Run commands from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

For an interactive Julia session:

```bash
julia --project=.
```

This package is local and unregistered. Never run:

```julia
Pkg.add("LongitudinalTomography")
```

Activate the repository first and then load it:

```julia
import Pkg
Pkg.activate(project_directory)
Pkg.instantiate()
using LongitudinalTomography
```

In notebooks, do not assume a fixed number of parent directories unless the notebook location is fixed. Prefer a small upward search for the nearest `Project.toml` whose `name` is `LongitudinalTomography`.

## Physics Conventions

### Single-RF normalized Hamiltonian

The stationary bucket uses

```math
H(\theta,P)=\frac{P^2}{2}-\cos\theta,
```

with separatrix energy `H = 1`. The analytical libration transport is valid only for trapped particles strictly inside the separatrix.

Use the elliptic modulus `k` in documentation, equations, variable names, and public APIs. `JacobiElliptic.jl` may require the parameter `m = k^2` internally, but do not expose that internal convention as the physical notation.

### BLonD and Recycler coordinates

The physical coordinates are arrival-time offset `delta_t`, energy offset `delta_E`, and turn number `N`.

For a `SingleRFMachine`,

```math
\theta=\omega_{\mathrm{rf}}(\Delta t-\Delta t_s),
```

```math
P=\frac{h\eta_0}{\beta_s^2E_s\nu_s}\Delta E,
```

and, to first order,

```math
\delta\equiv\frac{\Delta p}{p_0}
=\frac{\Delta E}{\beta_s^2E_s}.
```

Normalized momentum `P` is not `delta_p_over_p0`. Label and store them separately.

When longitudinal position is needed, use the explicit convention

```math
z=-\beta_s c\,\Delta t.
```

Keep units visible in names for dimensional quantities, for example `_hz`, `_s`, `_eV`, and `_m`.

### Two-RF sign convention

The package defines

```math
f_{\mathrm{slip}}=f_{\mathrm{rf1}}-f_{\mathrm{rf2}},
```

so a lower-frequency RF2 has positive `f_slip_hz`. Then

```math
\nu_{\mathrm{slip}}=\frac{f_{\mathrm{slip}}}{f_{\mathrm{rev}}},
\qquad
\alpha_s=\frac{\nu_{\mathrm{slip}}}{\nu_{s1}},
```

```math
\frac{\Delta p_{\mathrm{sep}}}{p_0}
=\frac{\nu_{\mathrm{slip}}}{h\eta_0},
\qquad
\Delta E_{\mathrm{sep}}
=\beta_s^2E_s\frac{\Delta p_{\mathrm{sep}}}{p_0}.
```

Below transition, a lower-frequency RF2 has a negative physical momentum and energy offset even though its Lee--Ng normalized RF1 momentum center `alpha_s` is positive. Tests must protect this sign distinction.

### Local, wrapped, and absolute phase

Never use a phase shift of `2pi` to identify a different RF bucket inside the single-bucket analytical solver. The solver is periodic and will erase that identity.

Represent a bunch using:

- a local phase `theta_local` within its own bucket;
- an integer `bucket_index`;
- an unwrapped absolute RF1 phase `theta_absolute`.

For RF1 bucket `b`,

```math
\Theta_1(N)=\theta_{1,\mathrm{local}}(N)+2\pi b.
```

For RF2 bucket `b`, expressed in RF1 coordinates,

```math
\Theta_2(N)=\theta_{2,\mathrm{local}}(N)
+2\pi b+\Delta\phi_0
+2\pi\nu_{\mathrm{slip}}(N-N_0).
```

The RF2 normalized momentum in the RF1 convention is

```math
P_{2,\mathrm{RF1}}
=\frac{\nu_{s2}}{\nu_{s1}}P_{2,\mathrm{local}}+\alpha_s.
```

Use unwrapped phase or absolute `delta_t`/`z` when studying bunch-train motion and crossings. Wrap only at an explicitly periodic measurement boundary. Do not sort or wrap unwrapped trajectories merely to make a plot look continuous.

## Model Scope and Limitations

The current `NonInteractingTwoRFMachine` is an approximation:

- RF1 acts only on family 1.
- RF2 acts only on family 2.
- Each family follows an integrable stationary single-RF Hamiltonian.
- RF2 is translated into the RF1 frame using slip phase and momentum separation.
- Cross-family perturbations, resonances, chaotic layers, capture, and loss are absent.

Do not describe results from this model as the solution of the full coupled two-RF Hamiltonian. Use names such as `noninteracting`, `uncoupled`, or `translated single-RF` in code and documentation.

The exact elliptic solution becomes numerically delicate near the separatrix. Keep active grid nodes inside it using a nonzero `energy_margin`, and test the chosen margin. Do not silently clip invalid particles into the bucket.

## Differentiability Rules

- Keep differentiated functions pure and type-stable where practical.
- Prefer broadcasting and array expressions over mutation in code traversed by Zygote.
- Do not use hard histogram bin assignment in a differentiable forward model. Use the existing normalized Gaussian soft assignments.
- Precompute fixed dynamics and measurement responses outside the reconstruction loop.
- For fixed machine parameters, observation turns, bins, and `sigma`, tomography should reduce to a matrix-vector product.
- Do not differentiate through plotting, grid construction, bucket-index bookkeeping, validation, or response metadata.
- If machine parameters become trainable later, rebuild or differentiate the response through a separate clearly named path; do not pretend a precomputed matrix remains valid while its machine parameters change.
- Add a finite-difference comparison whenever introducing a new differentiated scalar objective.

## Immediate Goal: Slip-Stacked Grid Tomography

Implement the simplest useful model first: one bunch in RF1 and one bunch in RF2, both noninteracting, observed by one common RWCM time axis. Then generalize to two bunches per family.

For fixed machine and bucket placement, the forward model should remain linear in the initial cell masses:

```math
\widehat{\mathbf y}
=\mathbf A\mathbf w.
```

Build the combined response conceptually as

```math
\mathbf A=
\begin{bmatrix}
\mathbf A_{1,b_1} & \mathbf A_{2,b_2} & \cdots
\end{bmatrix},
```

where each block transports one family/bucket grid into the same absolute RWCM bins.

### Required forward-model behavior

1. Construct active local phase-space grids inside each family’s separatrix.
2. Transport RF1 grid nodes with RF1 and RF2 grid nodes with RF2.
3. Translate each local trajectory using its family and integer bucket index.
4. Convert unwrapped absolute RF1 phase to absolute `delta_t`.
5. Deposit every response column into the same soft RWCM bin grid.
6. Flatten the response to `(n_turns*n_time_bins, n_unknown_cells)` only after confirming the tensor axis order.
7. Retain column metadata mapping every unknown to `(family, bucket_index, local_cell_index)`.

Each response column should deposit unit charge at every observation turn before bunch-charge weighting. If the final weights represent total beam fractions, normalize once across all families and buckets, not independently within every block.

### Independent versus shared bunch shapes

Support these as distinct, explicit modes rather than conflating them:

- **Independent distributions:** every bunch has its own grid weights.
- **Shared template:** selected bunches reuse one local grid distribution with separate or fixed bunch charges.

Start with independent distributions because the linear operator and unknown vector are unambiguous. Add shared-template parameter tying only after the independent case passes.

### Measurement window

Choose absolute time bins wide enough to contain all selected buckets over the requested turns. A one-bucket interval such as `[-T_rf/2, T_rf/2]` cannot display several unwrapped buckets crossing.

If modeling a periodic oscilloscope/RWCM window, implement that as an explicit circular measurement operator. Do not apply `wrap_to_pi` implicitly inside transport.

### Regularization

Total variation adjacency belongs within each local phase-space grid. Do not connect the last cell of one bucket block to the first cell of another. Cross-bunch or cross-family similarity should be a separate optional penalty.

The initial baseline loss may remain profile MSE plus existing TV and entropy penalties. Preserve the data term and regularization terms separately in diagnostics. Wasserstein or instrument-noise-aware losses can be evaluated after the forward operator is validated.

## Suggested API Shape

Use the existing style and dispatch where possible. A reasonable starting point is:

```julia
struct BucketSpec
    family::Symbol       # :rf1 or :rf2
    bucket_index::Int
    charge_fraction::Float64
end

struct SlipStackedGridResponse
    matrix
    turns
    time_bin_centers_s
    sigma_s
    column_metadata
    # grid and machine/bucket metadata as needed
end
```

Potential public functions:

```julia
build_slip_stacked_response(...)
profiles_from_response(...)
slip_stacked_tomography_loss(...)
reconstruct_slip_stacked_grid(...)
```

This is guidance, not a mandate. Prefer extending `GridResponse` by clean dispatch if that produces a smaller and clearer API. Avoid broad `Any`-typed containers in performance-critical paths.

## Implementation Order

1. Write the expected one-RF1/one-RF2 behavior as tests.
2. Add typed bucket and response metadata.
3. Build the two-family forward response without inversion.
4. Compare the response-matrix cascade against a direct transported soft histogram for identical nodes and weights.
5. Test one bunch per family, then two bunches per family.
6. Create a synthetic target with known family/bucket weights.
7. Reconstruct it using the existing softmax-weight baseline.
8. Add a notebook that plots initial phase space, bucket separatrices, the target cascade, reconstructed cascade, residual, and recovered distributions.
9. Only after this baseline passes, consider machine-parameter inference, interacting two-RF dynamics, generative models, or experimental RWCM transfer functions.

## Required Tests for Slip-Stacked Tomography

At minimum, test:

- machine sign conventions for `nu_slip`, `alpha_s`, `delta_p_over_p0`, and `delta_E`;
- a fixed-point particle in RF2 follows the analytical moving bucket center;
- bucket indices shift absolute phase by exactly `2pi` and time by exactly one RF period;
- local coordinates are unchanged by bucket placement;
- response dimensions and column metadata are consistent;
- every response column conserves unit charge at every turn;
- combined response profiles equal direct soft-histogram profiles;
- total predicted charge equals the sum of supplied bunch charges;
- RF1 and RF2 blocks do not share accidental TV edges;
- one-bunch and two-bunch cases both work;
- wrapped and unwrapped measurement modes, if both exist, are tested independently;
- gradients are finite and agree with a directional finite difference;
- a small synthetic reconstruction decreases its loss and returns nonnegative normalized weights;
- all existing single-RF and grid-tomography tests remain unchanged and passing.

Use deterministic random seeds in tests and notebooks.

## Coding Style

- Favor small functions with explicit physical meaning.
- Add docstrings to public types and functions, including units and sign conventions.
- Use descriptive ASCII identifiers in APIs, with mathematical symbols in documentation when helpful.
- Use `2pi` consistently with existing Julia code.
- Validate dimensions, positive scales, nonempty arrays, and compatible machines at API boundaries.
- Return named tuples or typed structs with stable, documented fields.
- Avoid changing existing field names without updating all callers and tests.
- Avoid hidden global state.
- Keep plotting code outside numerical kernels.
- Comments should explain physics choices and non-obvious conventions, not restate syntax.

## Plotting and Diagnostics

Use CairoMakie in notebooks for static, reproducible plots. In RF1-frame phase-space plots:

- RF1 separatrix centers are `2pi*b`.
- RF2 separatrix centers are `2pi*b + slip_phase[N]`.
- RF2 momentum centers are `alpha_s` in normalized RF1 momentum.
- For unequal voltages, scale the RF2 local separatrix height by `nu_s2/nu_s1`.

Label normalized momentum `P` as `P`. Label physical momentum as `Delta p / p0` or `delta`; never label one as the other.

For tomography notebooks, always show:

- the true and reconstructed initial distributions;
- the measured/target and predicted cascades on common color limits;
- a residual cascade;
- loss history;
- training and held-out turn errors when validation turns are used;
- total-charge checks.

## Working Agreement for Codex

- Make the smallest coherent change that advances the requested milestone.
- Preserve unrelated user edits and notebook work.
- Before editing, state the intended files and physics assumption.
- Do not silently change sign, phase, unit, normalization, or wrapping conventions.
- When a requirement is ambiguous, state the chosen convention and make it explicit in the API.
- Run focused tests during development and the full test suite before handoff.
- If Julia or a required dependency is unavailable, say exactly which verification was not run; do not claim tests passed.
- Report changed files, the implemented behavior, test commands/results, and remaining model limitations.
- Do not implement the full interacting two-RF Hamiltonian unless explicitly requested. The next milestone is grid tomography on the validated noninteracting slip-stacking model.
