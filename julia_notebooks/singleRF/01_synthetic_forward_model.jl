# %% [markdown]
# # Phase 1: differentiable synthetic RWCM forward model
#
# This VS Code-friendly Julia notebook generates a trapped initial distribution,
# transports it analytically through a stationary Recycler-like RF bucket, and
# produces a turn-by-turn cascade of differentiable soft histograms.

# %%
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

# %%
using LongitudinalTomography
using CairoMakie
using Random
using Zygote
using Printf

CairoMakie.activate!()

# %% [markdown]
# ## Recycler-like stationary RF parameters

# %%
const PROTON_MASS_EV = 938.2720813e6

kinetic_energy_eV = 8.0e9
total_energy_eV = kinetic_energy_eV + PROTON_MASS_EV
gamma_s = total_energy_eV / PROTON_MASS_EV
beta_s = sqrt(1 - 1 / gamma_s^2)

machine = SingleRFMachine(
    harmonic=588,
    f_rev_hz=89.8e3,
    eta0=-8.8e-3,
    beta_s=beta_s,
    E_s_eV=total_energy_eV,
    rf_kick_eV=60.0e3,
    phi_s=0.0,
    delta_t_s=0.0,
)

@printf("Q_s                    = %.7f\n", synchrotron_tune(machine))
@printf("synchrotron period     = %.2f turns\n", synchrotron_period_turns(machine))
@printf("bucket half-width      = %.3f ns\n", bucket_half_width_s(machine) * 1e9)
@printf("bucket half-height     = %.3f MeV\n", bucket_half_height_eV(machine) * 1e-6)

# %% [markdown]
# ## Smooth synthetic initial distribution inside the separatrix
#
# The latent variables are Gaussian, but the final transformation is smooth and
# bounded:
#
#     theta = pi*tanh(u)
#     P = 2*rho*cos(theta/2)*tanh(v),  rho < 1.
#
# This guarantees every particle is inside the stationary separatrix.

# %%
rng = MersenneTwister(42)
n_particles = 4096
correlation = 0.55

z1 = randn(rng, n_particles)
z2 = randn(rng, n_particles)

u = 0.24 .* z1
v = 0.32 .* (
    correlation .* z1 .+
    sqrt(1 - correlation^2) .* z2
)

theta0, P0 = trapped_coordinates(u, v; momentum_fraction=0.92)
@assert all(inside_separatrix(theta0, P0; margin=1e-12))

delta_t0_s = delta_t_from_theta(theta0, machine)
delta_E0_eV = delta_energy_from_P(P0, machine)

# %% [markdown]
# ## Differentiable forward projection
#
# We use 51 profiles from turn 0 through turn 250. The 75 time-bin centers span
# one RF bucket, approximately 18.9 ns. `soft_sigma_s` controls the Gaussian
# soft-assignment width and can later be matched to the effective RWCM time
# resolution.

# %%
turns = collect(0.0:5.0:250.0)

half_width_s = bucket_half_width_s(machine)
n_time_bins = 75
time_bin_centers_s = collect(range(
    machine.delta_t_s - half_width_s,
    machine.delta_t_s + half_width_s;
    length=n_time_bins,
))
time_bin_spacing_s = time_bin_centers_s[2] - time_bin_centers_s[1]
soft_sigma_s = 0.75 * time_bin_spacing_s

cascade = predict_rwcm_cascade(
    theta0,
    P0,
    turns,
    machine,
    time_bin_centers_s;
    sigma=soft_sigma_s,
)

@printf("particles               = %d\n", n_particles)
@printf("profiles                = %d\n", length(turns))
@printf("time bins               = %d\n", n_time_bins)
@printf("time-bin spacing        = %.3f ns\n", time_bin_spacing_s * 1e9)
@printf("soft histogram sigma    = %.3f ns\n", soft_sigma_s * 1e9)
@printf(
    "largest normalization error = %.3e\n",
    maximum(abs.(sum(cascade.profiles; dims=2) .- 1)),
)

# %% [markdown]
# ## Initial phase space and synthetic RWCM cascade

# %%
figure = Figure(size=(1300, 500))

axis_initial = Axis(
    figure[1, 1];
    xlabel="Delta t [ns]",
    ylabel="Delta E [MeV]",
    title="Synthetic initial longitudinal phase space",
)

theta_sep = range(-pi, pi; length=1200)
P_sep = 2 .* cos.(theta_sep ./ 2)
delta_t_sep_ns = delta_t_from_theta(theta_sep, machine) .* 1e9
delta_E_sep_1_MeV = delta_energy_from_P(P_sep, machine) .* 1e-6
delta_E_sep_2_MeV = delta_energy_from_P(-P_sep, machine) .* 1e-6

lines!(
    axis_initial,
    delta_t_sep_ns,
    delta_E_sep_1_MeV;
    color=:black,
    linestyle=:dash,
    linewidth=1.5,
)
lines!(
    axis_initial,
    delta_t_sep_ns,
    delta_E_sep_2_MeV;
    color=:black,
    linestyle=:dash,
    linewidth=1.5,
)
scatter!(
    axis_initial,
    delta_t0_s .* 1e9,
    delta_E0_eV .* 1e-6;
    markersize=2.5,
    color=(:navy, 0.3),
)

axis_cascade = Axis(
    figure[1, 2];
    xlabel="Delta t [ns]",
    ylabel="turn N",
    title="Synthetic differentiable RWCM cascade",
)

heatmap_object = heatmap!(
    axis_cascade,
    time_bin_centers_s .* 1e9,
    turns,
    permutedims(cascade.profiles);
    colormap=:viridis,
)
Colorbar(figure[1, 3], heatmap_object; label="normalized line density")

display(figure)

# Uncomment to save the figure beside this notebook.
# save(joinpath(@__DIR__, "phase1_synthetic_forward_model.png"), figure)

# %% [markdown]
# ## End-to-end differentiability demonstration
#
# The scalar below shifts the unconstrained latent coordinate. The gradient
# propagates through the trapped-particle map, Jacobi elliptic solution, BLonD
# coordinate conversion, and soft histogram. A smaller subset is used here to
# keep this diagnostic quick and memory-friendly on a laptop.

# %%
gradient_particle_indices = 1:256
gradient_turns = turns[1:5:end]
u_gradient = u[gradient_particle_indices]
v_gradient = v[gradient_particle_indices]

target_theta, target_P = trapped_coordinates(
    u_gradient .+ 0.04,
    v_gradient;
    momentum_fraction=0.92,
)
target_profiles = predict_rwcm_cascade(
    target_theta,
    target_P,
    gradient_turns,
    machine,
    time_bin_centers_s;
    sigma=soft_sigma_s,
    check=false,
).profiles

function profile_loss(latent_shift)
    theta, P = trapped_coordinates(
        u_gradient .+ latent_shift,
        v_gradient;
        momentum_fraction=0.92,
    )
    prediction = predict_rwcm_cascade(
        theta,
        P,
        gradient_turns,
        machine,
        time_bin_centers_s;
        sigma=soft_sigma_s,
        check=false,
    ).profiles
    return sum(abs2, prediction .- target_profiles)
end

loss_at_zero = profile_loss(0.0)
gradient_at_zero = only(Zygote.gradient(profile_loss, 0.0))

@printf("demonstration loss      = %.6e\n", loss_at_zero)
@printf("d(loss)/d(latent shift) = %.6e\n", gradient_at_zero)
