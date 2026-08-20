module LongitudinalTomography

using JacobiElliptic
using Zygote

export SingleRFMachine
export revolution_period, omega0, omega_rf, synchrotron_tune
export synchrotron_period_turns, bucket_half_width_s, bucket_half_height_eV
export tau_from_turn, turn_from_tau
export theta_from_delta_t, delta_t_from_theta
export P_from_delta_energy, delta_energy_from_P, delta_p_over_p0_from_delta_E
export pendulum_energy, inside_separatrix, trapped_coordinates
export libration_transport
export soft_assignments, soft_histogram, predict_rwcm_cascade
export NonInteractingTwoRFMachine
export slip_tune, nu_slip, slip_stacking_parameter, alpha_s
export fractional_momentum_separation, energy_separation_eV
export rf2_center_phase, rf2_center_P, wrap_to_pi
export noninteracting_two_rf_transport
export place_rf1_bunches, place_rf2_bunches
export PhaseSpaceGrid, GridResponse
export separatrix_grid, grid_weight_matrix, build_response_matrix
export grid_weights_from_logits, profiles_from_response
export profile_mse, grid_total_variation, grid_entropy_penalty
export tomography_loss, tomography_diagnostics, reconstruct_grid
export BucketSpec, SlipStackedColumnMetadata, SlipStackedGridResponse
export transport_slip_stacked_grid, build_slip_stacked_response
export direct_slip_stacked_profiles, uniform_bucket_weights
export slip_stacked_grid_total_variation

"""
Constant machine parameters for a stationary, single-harmonic RF bucket.

`rf_kick_eV` is the energy kick qV in eV. For a proton, its numerical value is
the RF voltage in volts. The exact transport implemented in this package assumes
`sin(phi_s) = 0` and `eta0 * rf_kick_eV * cos(phi_s) < 0`.
"""
struct SingleRFMachine
    harmonic::Int
    f_rev_hz::Float64
    eta0::Float64
    beta_s::Float64
    E_s_eV::Float64
    rf_kick_eV::Float64
    phi_s::Float64
    delta_t_s::Float64
end


function SingleRFMachine(;
    harmonic::Integer,
    f_rev_hz::Real,
    eta0::Real,
    beta_s::Real,
    E_s_eV::Real,
    rf_kick_eV::Real,
    phi_s::Real=0.0,
    delta_t_s::Real=0.0,
)
    machine = SingleRFMachine(
        Int(harmonic),
        Float64(f_rev_hz),
        Float64(eta0),
        Float64(beta_s),
        Float64(E_s_eV),
        Float64(rf_kick_eV),
        Float64(phi_s),
        Float64(delta_t_s),
    )

    machine.harmonic > 0 || throw(ArgumentError("harmonic must be positive"))
    machine.f_rev_hz > 0 || throw(ArgumentError("f_rev_hz must be positive"))
    0 < machine.beta_s <= 1 || throw(ArgumentError("beta_s must lie in (0,1]"))
    machine.E_s_eV > 0 || throw(ArgumentError("E_s_eV must be positive"))
    machine.rf_kick_eV != 0 || throw(ArgumentError("rf_kick_eV must be nonzero"))
    abs(sin(machine.phi_s)) <= 1e-10 || throw(ArgumentError(
        "The analytical libration model requires a stationary bucket: sin(phi_s)=0.",
    ))

    stability = machine.eta0 * machine.rf_kick_eV * cos(machine.phi_s)
    stability < 0 || throw(ArgumentError(
        "Unstable synchronous phase: eta0*rf_kick_eV*cos(phi_s) must be negative.",
    ))

    return machine
end


revolution_period(machine::SingleRFMachine) = 1 / machine.f_rev_hz
omega0(machine::SingleRFMachine) = 2pi * machine.f_rev_hz
omega_rf(machine::SingleRFMachine) = machine.harmonic * omega0(machine)

function synchrotron_tune(machine::SingleRFMachine)
    tune_squared = -machine.harmonic * machine.eta0 * machine.rf_kick_eV *
        cos(machine.phi_s) / (2pi * machine.beta_s^2 * machine.E_s_eV)
    return sqrt(tune_squared)
end

synchrotron_period_turns(machine::SingleRFMachine) = 1 / synchrotron_tune(machine)

bucket_half_width_s(machine::SingleRFMachine) = pi / omega_rf(machine)

function bucket_half_height_eV(machine::SingleRFMachine)
    scale = machine.beta_s^2 * machine.E_s_eV * synchrotron_tune(machine) /
        (machine.harmonic * abs(machine.eta0))
    return 2scale
end


tau_from_turn(N, machine::SingleRFMachine; N0=0, tau0=0) =
    tau0 .+ 2pi * synchrotron_tune(machine) .* (N .- N0)

turn_from_tau(tau, machine::SingleRFMachine; N0=0, tau0=0) =
    N0 .+ (tau .- tau0) ./ (2pi * synchrotron_tune(machine))

theta_from_delta_t(delta_t, machine::SingleRFMachine) =
    omega_rf(machine) .* (delta_t .- machine.delta_t_s)

delta_t_from_theta(theta, machine::SingleRFMachine) =
    machine.delta_t_s .+ theta ./ omega_rf(machine)

P_from_delta_energy(delta_E, machine::SingleRFMachine) =
    machine.harmonic * machine.eta0 .* delta_E ./
    (machine.beta_s^2 * machine.E_s_eV * synchrotron_tune(machine))

delta_energy_from_P(P, machine::SingleRFMachine) =
    machine.beta_s^2 * machine.E_s_eV * synchrotron_tune(machine) .* P ./
    (machine.harmonic * machine.eta0)


pendulum_energy(theta, P) = 0.5 .* P.^2 .- cos.(theta)

inside_separatrix(theta, P; margin=0) =
    pendulum_energy(theta, P) .< (1 - margin)

"""
    trapped_coordinates(u, v; momentum_fraction=0.98)

Smoothly map unconstrained coordinates `(u,v)` into the interior of the
stationary RF separatrix:

    theta = pi*tanh(u)
    P = 2*momentum_fraction*cos(theta/2)*tanh(v)

For finite `u`, `v` and `momentum_fraction <= 1`, every returned particle has
normalized pendulum energy strictly below the separatrix energy E=1.
"""
function trapped_coordinates(u, v; momentum_fraction=0.98)
    0 < momentum_fraction <= 1 || throw(ArgumentError(
        "momentum_fraction must lie in (0,1].",
    ))
    size(u) == size(v) || throw(DimensionMismatch("u and v must have equal size"))

    theta = pi .* tanh.(u)
    P_limit = 2 .* cos.(theta ./ 2)
    P = momentum_fraction .* P_limit .* tanh.(v)
    return theta, P
end


# Public notation uses the elliptic modulus k. JacobiElliptic.jl expects m=k^2.
Fmod(phi, k) = JacobiElliptic.F(phi, k^2)
snmod(z, k) = JacobiElliptic.sn(z, k^2)
cnmod(z, k) = JacobiElliptic.cn(z, k^2)

"""
    libration_transport(theta0, P0, tau; check=true, margin=1e-10)

Pure, batched analytical transport for particles inside the stationary RF
separatrix. `theta0` and `P0` are particle vectors and `tau` is a vector of
normalized observation times.

Returns matrices with shape `(length(tau), length(theta0))`. The implementation
uses broadcasting rather than mutation so that it can be differentiated by
Zygote. Set `check=false` inside a repeatedly evaluated training loss after the
generator has already guaranteed trapped particles.
"""
function libration_transport(
    theta0::AbstractVector,
    P0::AbstractVector,
    tau::AbstractVector;
    check::Bool=true,
    margin::Real=1e-10,
)
    length(theta0) == length(P0) || throw(DimensionMismatch(
        "theta0 and P0 must contain the same number of particles.",
    ))
    isempty(theta0) && throw(ArgumentError("At least one particle is required."))
    isempty(tau) && throw(ArgumentError("At least one observation time is required."))

    energy = pendulum_energy(theta0, P0)
    if check && !all(energy .< (1 - margin))
        maximum_energy = maximum(energy)
        throw(DomainError(
            maximum_energy,
            "All particles must satisfy E < 1-margin for libration transport.",
        ))
    end

    # k is the elliptic modulus. The small floor only regularizes the exact
    # stable fixed point k=0, which otherwise gives the indeterminate ratio 0/0.
    k_squared_unregularized = (1 .+ energy) ./ 2
    fixed_point = k_squared_unregularized .<= 0
    k_squared = max.(k_squared_unregularized, eps(Float64))
    k = sqrt.(k_squared)

    sn0 = clamp.(sin.(theta0 ./ 2) ./ k, -1, 1)
    cn0 = clamp.(P0 ./ (2 .* k), -1, 1)
    amplitude0 = atan.(sn0, cn0)
    z0 = Fmod.(amplitude0, k)

    n_particles = length(theta0)
    n_times = length(tau)
    z = reshape(tau, n_times, 1) .+ reshape(z0, 1, n_particles)
    k_grid = reshape(k, 1, n_particles)

    sn_values = snmod.(z, k_grid)
    cn_values = cnmod.(z, k_grid)
    theta = 2 .* asin.(clamp.(k_grid .* sn_values, -1, 1))
    P = 2 .* k_grid .* cn_values

    # Preserve the exact stable fixed point. The elliptic-modulus floor above
    # otherwise turns (theta,P)=(0,0) into a spurious O(sqrt(eps)) orbit.
    fixed_point_grid = reshape(fixed_point, 1, n_particles)
    theta = ifelse.(fixed_point_grid, 0, theta)
    P = ifelse.(fixed_point_grid, 0, P)

    return (
        theta=theta,
        P=P,
        energy=energy,
        modulus=k,
    )
end


"""
    soft_assignments(samples, bin_centers; sigma)

Return differentiable Gaussian bin-assignment probabilities with shape
`(n_observations, n_bins, n_particles)`. Assignments are normalized over the
finite bin grid for every observation and particle, so each particle deposits
exactly unit charge at every observation.
"""
function soft_assignments(
    samples::AbstractMatrix,
    bin_centers::AbstractVector;
    sigma::Real,
)
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    n_observations, n_particles = size(samples)
    n_bins = length(bin_centers)
    n_bins > 1 || throw(ArgumentError("At least two bin centers are required."))

    sample_grid = reshape(samples, n_observations, 1, n_particles)
    bin_grid = reshape(bin_centers, 1, n_bins, 1)
    log_assignments = -0.5 .* ((sample_grid .- bin_grid) ./ sigma).^2

    # Stable softmax over bins for every observation and particle.
    shifted = log_assignments .- maximum(log_assignments; dims=2)
    unnormalized = exp.(shifted)
    return unnormalized ./ sum(unnormalized; dims=2)
end


"""
    soft_histogram(samples, bin_centers; sigma, particle_weights=nothing,
                   normalize=true)

Differentiable Gaussian soft binning for a matrix of samples with shape
`(n_observations, n_particles)`. Each particle's bin-assignment probabilities
are normalized across the finite bin grid, so charge is conserved exactly.

The result has shape `(n_observations, n_bins)`. When `normalize=true`, every
profile sums to one. No hard bin indexing or mutation is used.
"""
function soft_histogram(
    samples::AbstractMatrix,
    bin_centers::AbstractVector;
    sigma::Real,
    particle_weights=nothing,
    normalize::Bool=true,
)
    _, n_particles = size(samples)
    assignments = soft_assignments(samples, bin_centers; sigma=sigma)

    weighted_assignments = if isnothing(particle_weights)
        assignments
    else
        length(particle_weights) == n_particles || throw(DimensionMismatch(
            "particle_weights must contain one weight per particle.",
        ))
        assignments .* reshape(particle_weights, 1, 1, n_particles)
    end

    profiles = dropdims(sum(weighted_assignments; dims=3); dims=3)
    if normalize
        profiles = profiles ./ sum(profiles; dims=2)
    end
    return profiles
end


"""
    predict_rwcm_cascade(theta0, P0, turns, machine, time_bin_centers;
                         sigma, particle_weights=nothing, check=true)

Phase-1 synthetic forward model:

1. convert turns to normalized time;
2. analytically transport trapped particles;
3. convert phase to BLonD arrival-time offset;
4. project over energy using a differentiable soft histogram.

The returned `profiles` are normalized longitudinal line-density profiles. A
measured RWCM transfer function, gain, baseline, and noise model will be added
as a later measurement layer.
"""
function predict_rwcm_cascade(
    theta0::AbstractVector,
    P0::AbstractVector,
    turns::AbstractVector,
    machine::SingleRFMachine,
    time_bin_centers::AbstractVector;
    sigma::Real,
    particle_weights=nothing,
    check::Bool=true,
)
    tau = tau_from_turn(turns, machine)
    trajectory = libration_transport(theta0, P0, tau; check=check)
    delta_t = delta_t_from_theta(trajectory.theta, machine)
    profiles = soft_histogram(
        delta_t,
        time_bin_centers;
        sigma=sigma,
        particle_weights=particle_weights,
        normalize=true,
    )

    return (
        turns=turns,
        tau=tau,
        theta=trajectory.theta,
        P=trajectory.P,
        delta_t=delta_t,
        delta_E=delta_energy_from_P(trajectory.P, machine),
        profiles=profiles,
        energy=trajectory.energy,
        modulus=trajectory.modulus,
    )
end

include("two_rf_noninteracting.jl")
include("grid_tomography.jl")
include("slip_stacked_grid_tomography.jl")

end # module
