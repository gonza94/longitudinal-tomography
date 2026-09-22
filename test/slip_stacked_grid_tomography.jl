@testset "Slip-stacked grid response" begin
    proton_mass_eV = 938.2720813e6
    total_energy_eV = 8.0e9 + proton_mass_eV
    gamma_s = total_energy_eV / proton_mass_eV
    beta_s = sqrt(1 - 1 / gamma_s^2)

    rf1 = SingleRFMachine(
        harmonic=588,
        f_rev_hz=89.8e3,
        eta0=-8.8e-3,
        beta_s=beta_s,
        E_s_eV=total_energy_eV,
        rf_kick_eV=60e3,
    )

    two_rf = NonInteractingTwoRFMachine(
        rf1;
        f_slip_hz=84 * 15.0,
        relative_phase0=0.0,
    )

    @test slip_tune(two_rf) > 0
    @test slip_stacking_parameter(two_rf) > 0
    @test fractional_momentum_separation(two_rf) < 0
    @test energy_separation_eV(two_rf) < 0

    grid = separatrix_grid(
        n_theta=9,
        n_P=9,
        energy_margin=0.08,
    )

    turns = collect(0.0:5.0:20.0)
    time_bin_centers_s = delta_t_from_theta(
        collect(range(-4pi, 3pi; length=85)),
        rf1,
    )
    sigma_s = 0.7 * (time_bin_centers_s[2] - time_bin_centers_s[1])

    bucket_specs = [
        BucketSpec(:rf1, 0, 0.4),
        BucketSpec(:rf2, -1, 0.6),
    ]

    response = build_slip_stacked_response(
        grid,
        bucket_specs,
        turns,
        two_rf,
        time_bin_centers_s;
        sigma=sigma_s,
    )

    n_cells = length(grid)
    n_rows = length(turns) * length(time_bin_centers_s)

    @test :slip_stacked_weights_from_logits in names(LongitudinalTomography)
    @test size(response.matrix) == (n_rows, 2n_cells)
    @test response.block_ranges == [1:n_cells, (n_cells + 1):(2n_cells)]
    @test length(response.column_metadata) == 2n_cells
    @test response.column_metadata[1] ==
        SlipStackedColumnMetadata(:rf1, 0, 1)
    @test response.column_metadata[n_cells + 1] ==
        SlipStackedColumnMetadata(:rf2, -1, 1)

    response_tensor = reshape(
        response.matrix,
        length(turns),
        length(time_bin_centers_s),
        size(response.matrix, 2),
    )
    column_charge_by_turn =
        dropdims(sum(response_tensor; dims=2); dims=2)
    @test maximum(abs.(column_charge_by_turn .- 1)) < 2e-14

    smoke_weights = slip_stacked_weights_from_logits(
        zeros(size(response.matrix, 2)),
        response,
    )
    @test length(smoke_weights) == size(response.matrix, 2)
    @test sum(smoke_weights) ≈ 1.0

    rf1_bucket0 = transport_slip_stacked_grid(
        grid,
        BucketSpec(:rf1, 0),
        turns,
        two_rf,
    )
    rf1_bucket1 = transport_slip_stacked_grid(
        grid,
        BucketSpec(:rf1, 1),
        turns,
        two_rf,
    )
    rf_period_s = 2pi / omega_rf(rf1)

    @test maximum(abs.(
        rf1_bucket1.theta_local .- rf1_bucket0.theta_local,
    )) < 2e-14
    @test maximum(abs.(
        rf1_bucket1.theta_absolute .- rf1_bucket0.theta_absolute .- 2pi,
    )) < 2e-14
    @test maximum(abs.(
        rf1_bucket1.delta_t .- rf1_bucket0.delta_t .- rf_period_s,
    )) < 2e-22

    center_grid = separatrix_grid(
        n_theta=3,
        n_P=3,
        energy_margin=0.0,
    )
    fixed_point_index = findfirst(
        (abs.(center_grid.theta_nodes) .< 1e-14) .&
            (abs.(center_grid.P_nodes) .< 1e-14),
    )
    @test !isnothing(fixed_point_index)

    rf2_center = transport_slip_stacked_grid(
        center_grid,
        BucketSpec(:rf2, -1),
        turns,
        two_rf,
    )
    expected_rf2_center_phase =
        -2pi .+
        rf2_center_phase(turns, two_rf; N0=first(turns))

    @test maximum(abs.(
        rf2_center.theta_absolute[:, fixed_point_index] .-
            expected_rf2_center_phase,
    )) < 2e-14

    rng = MersenneTwister(4321)
    weights = rand(rng, size(response.matrix, 2))
    weights ./= sum(weights)

    matrix_profiles = profiles_from_response(
        response,
        weights;
        normalize=false,
    )
    direct_profiles = direct_slip_stacked_profiles(
        response,
        weights,
        two_rf;
        normalize=false,
    )

    @test maximum(abs.(matrix_profiles .- direct_profiles)) < 2e-14

    charge_weights = uniform_bucket_weights(response; normalize=false)
    charge_profiles = profiles_from_response(
        response,
        charge_weights;
        normalize=false,
    )
    expected_total_charge = sum(spec.charge_fraction for spec in bucket_specs)
    charge_by_turn = vec(sum(charge_profiles; dims=2))

    @test maximum(abs.(charge_by_turn .- expected_total_charge)) < 2e-14

    block1_weights = rand(rng, n_cells)
    block2_weights = rand(rng, n_cells)
    stacked_weights = vcat(block1_weights, block2_weights)
    expected_tv =
        grid_total_variation(block1_weights, grid) +
        grid_total_variation(block2_weights, grid)

    @test slip_stacked_grid_total_variation(stacked_weights, response) ≈
        expected_tv

    target_weights = rand(rng, size(response.matrix, 2))
    target_weights ./= sum(target_weights)
    target_profiles = profiles_from_response(
        response,
        target_weights;
        normalize=false,
    )
    initial_logits = randn(rng, size(response.matrix, 2))
    direction = randn(rng, size(response.matrix, 2))
    direction ./= sqrt(sum(abs2, direction))

    objective(logits) = profile_mse(
        profiles_from_response(
            response,
            grid_weights_from_logits(logits);
            normalize=false,
        ),
        target_profiles,
    )

    gradient_ad = only(Zygote.gradient(objective, initial_logits))
    step = 1e-5
    directional_ad = sum(gradient_ad .* direction)
    directional_fd =
        (objective(initial_logits .+ step .* direction) -
         objective(initial_logits .- step .* direction)) / (2step)

    @test all(isfinite, gradient_ad)
    @test isapprox(directional_ad, directional_fd; rtol=5e-3, atol=1e-8)
end
