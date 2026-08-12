using Test
using Random
using Zygote
using LongitudinalTomography

const PROTON_MASS_EV = 938.2720813e6
const TOTAL_ENERGY_EV = 8.0e9 + PROTON_MASS_EV
const GAMMA_S = TOTAL_ENERGY_EV / PROTON_MASS_EV
const BETA_S = sqrt(1 - 1 / GAMMA_S^2)

machine = SingleRFMachine(
    harmonic=588,
    f_rev_hz=89.8e3,
    eta0=-8.8e-3,
    beta_s=BETA_S,
    E_s_eV=TOTAL_ENERGY_EV,
    rf_kick_eV=60e3,
)

@testset "Phase-1 synthetic forward model" begin
    rng = MersenneTwister(1234)
    n_particles = 64
    z1 = randn(rng, n_particles)
    z2 = randn(rng, n_particles)
    u = 0.24 .* z1
    v = 0.32 .* (0.55 .* z1 .+ sqrt(1 - 0.55^2) .* z2)
    theta0, P0 = trapped_coordinates(u, v; momentum_fraction=0.92)

    @test all(inside_separatrix(theta0, P0; margin=1e-12))

    turns = collect(0.0:25.0:250.0)
    tau = tau_from_turn(turns, machine)
    trajectory = libration_transport(theta0, P0, tau)

    @test size(trajectory.theta) == (length(turns), n_particles)
    @test size(trajectory.P) == size(trajectory.theta)
    @test maximum(abs.(trajectory.theta[1, :] .- theta0)) < 2e-9
    @test maximum(abs.(trajectory.P[1, :] .- P0)) < 2e-9

    energy_over_time = pendulum_energy(trajectory.theta, trajectory.P)
    energy_reference = reshape(trajectory.energy, 1, n_particles)
    @test maximum(abs.(energy_over_time .- energy_reference)) < 2e-9

    half_width = bucket_half_width_s(machine)
    time_bins = collect(range(-half_width, half_width; length=41))
    sigma = 0.65 * (time_bins[2] - time_bins[1])
    cascade = predict_rwcm_cascade(
        theta0,
        P0,
        turns,
        machine,
        time_bins;
        sigma=sigma,
    )

    @test size(cascade.profiles) == (length(turns), length(time_bins))
    @test maximum(abs.(sum(cascade.profiles; dims=2) .- 1)) < 1e-12
    @test all(isfinite, cascade.profiles)

    # End-to-end gradient check: unconstrained latent shift -> trapped particles
    # -> elliptic transport -> soft histogram -> scalar profile loss.
    target_theta, target_P = trapped_coordinates(u .+ 0.035, v; momentum_fraction=0.92)
    target = predict_rwcm_cascade(
        target_theta,
        target_P,
        turns,
        machine,
        time_bins;
        sigma=sigma,
        check=false,
    ).profiles

    function loss(latent_shift)
        theta, P = trapped_coordinates(u .+ latent_shift, v; momentum_fraction=0.92)
        prediction = predict_rwcm_cascade(
            theta,
            P,
            turns,
            machine,
            time_bins;
            sigma=sigma,
            check=false,
        ).profiles
        return sum(abs2, prediction .- target)
    end

    gradient_ad = only(Zygote.gradient(loss, 0.0))
    step = 1e-5
    gradient_fd = (loss(step) - loss(-step)) / (2step)

    @test isfinite(gradient_ad)
    @test isapprox(gradient_ad, gradient_fd; rtol=5e-3, atol=1e-8)
end

