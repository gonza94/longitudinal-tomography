"""
    BucketSpec(family, bucket_index, charge_fraction=1.0)

Identify one slip-stacked bunch in the noninteracting two-RF model.

`family` must be `:rf1` or `:rf2`. `bucket_index` is the integer bucket
placement in the unwrapped RF1 phase convention. `charge_fraction` is metadata
used by [`uniform_bucket_weights`](@ref); response columns themselves deposit
unit charge at every observation turn before any bunch-charge weighting.
"""
struct BucketSpec
    family::Symbol
    bucket_index::Int
    charge_fraction::Float64

    function BucketSpec(
        family::Symbol,
        bucket_index::Integer,
        charge_fraction::Real=1.0,
    )
        family in (:rf1, :rf2) || throw(ArgumentError(
            "family must be :rf1 or :rf2.",
        ))
        isfinite(charge_fraction) || throw(ArgumentError(
            "charge_fraction must be finite.",
        ))
        charge_fraction >= 0 || throw(ArgumentError(
            "charge_fraction must be nonnegative.",
        ))

        return new(family, Int(bucket_index), Float64(charge_fraction))
    end
end

BucketSpec(;
    family::Symbol,
    bucket_index::Integer,
    charge_fraction::Real=1.0,
) = BucketSpec(family, bucket_index, charge_fraction)


"""
    SlipStackedColumnMetadata

Metadata for one column of a [`SlipStackedGridResponse`](@ref), mapping a flat
unknown back to its RF family, integer bucket placement, and local grid cell.
"""
struct SlipStackedColumnMetadata
    family::Symbol
    bucket_index::Int
    local_cell_index::Int
end


"""
    SlipStackedGridResponse

Precomputed linear operator for independent slip-stacked grid tomography.

`matrix` has shape `(n_turns*n_time_bins, n_unknown_cells)`. A product
`matrix * weights` reshaped to `(n_turns, n_time_bins)` gives the common RWCM
cascade. RF1 and RF2 grid nodes are transported in their own local stationary
buckets, translated to unwrapped absolute RF1 phase with the supplied bucket
indices, converted to absolute `delta_t` [s], and deposited into the same RWCM
time bins.
"""
struct SlipStackedGridResponse{T<:AbstractFloat}
    matrix::Matrix{T}
    rf1_grid::PhaseSpaceGrid{T}
    rf2_grid::PhaseSpaceGrid{T}
    bucket_specs::Vector{BucketSpec}
    block_ranges::Vector{UnitRange{Int}}
    column_metadata::Vector{SlipStackedColumnMetadata}
    turns::Vector{T}
    reference_turn::T
    time_bin_centers_s::Vector{T}
    sigma_s::T
end


function _reference_turn(turns::AbstractVector, N0)
    isempty(turns) && throw(ArgumentError(
        "At least one observation turn is required.",
    ))
    reference_turn = isnothing(N0) ? first(turns) : N0
    reference_turn isa Real || throw(ArgumentError("N0 must be a real turn."))
    return Float64(reference_turn)
end


function _grid_for_family(
    family::Symbol,
    rf1_grid::PhaseSpaceGrid{Float64},
    rf2_grid::PhaseSpaceGrid{Float64},
)
    if family == :rf1
        return rf1_grid
    elseif family == :rf2
        return rf2_grid
    end
    throw(ArgumentError("family must be :rf1 or :rf2."))
end


function _local_machine_for_family(
    family::Symbol,
    machine::NonInteractingTwoRFMachine,
)
    if family == :rf1
        return machine.rf1
    elseif family == :rf2
        return machine.rf2
    end
    throw(ArgumentError("family must be :rf1 or :rf2."))
end


"""
    transport_slip_stacked_grid(grid, bucket_spec, turns, machine;
                                N0=nothing, check=true)

Transport every active local grid cell for one slip-stacked bunch and return
its local coordinates together with unwrapped absolute RF1-frame coordinates.

For RF1,

```math
Theta_1(N)=theta_\\mathrm{local}(N)+2\\pi b.
```

For RF2,

```math
Theta_2(N)=theta_\\mathrm{local}(N)+2\\pi b
    +\\Delta\\phi_0+2\\pi\\nu_\\mathrm{slip}(N-N_0).
```

The returned `delta_t` is `delta_t_from_theta(theta_absolute, machine.rf1)` in
seconds. RF2 momentum is translated into the RF1 normalized momentum convention.
"""
function transport_slip_stacked_grid(
    grid::PhaseSpaceGrid{Float64},
    bucket_spec::BucketSpec,
    turns::AbstractVector,
    machine::NonInteractingTwoRFMachine;
    N0=nothing,
    check::Bool=true,
)
    turn_vector = Float64.(turns)
    reference_turn = _reference_turn(turn_vector, N0)
    local_machine = _local_machine_for_family(bucket_spec.family, machine)

    tau = tau_from_turn(turn_vector, local_machine; N0=reference_turn)
    local_trajectory = libration_transport(
        grid.theta_nodes,
        grid.P_nodes,
        tau;
        check=check,
        margin=grid.energy_margin,
    )

    theta_offset = 2pi * bucket_spec.bucket_index

    theta_absolute = if bucket_spec.family == :rf1
        local_trajectory.theta .+ theta_offset
    else
        slip_phase = rf2_center_phase(
            turn_vector,
            machine;
            N0=reference_turn,
        )
        local_trajectory.theta .+
            reshape(slip_phase, length(turn_vector), 1) .+
            theta_offset
    end

    P_rf1 = if bucket_spec.family == :rf1
        local_trajectory.P
    else
        momentum_scale =
            synchrotron_tune(machine.rf2) / synchrotron_tune(machine.rf1)
        momentum_scale .* local_trajectory.P .+
            slip_stacking_parameter(machine)
    end

    return (
        turns=turn_vector,
        N0=reference_turn,
        family=bucket_spec.family,
        bucket_index=bucket_spec.bucket_index,
        theta_local=local_trajectory.theta,
        P_local=local_trajectory.P,
        theta_absolute=theta_absolute,
        P_rf1=P_rf1,
        delta_t=delta_t_from_theta(theta_absolute, machine.rf1),
        delta_E_rf1=delta_energy_from_P(P_rf1, machine.rf1),
        energy=local_trajectory.energy,
        modulus=local_trajectory.modulus,
    )
end


function _response_block(
    grid::PhaseSpaceGrid{Float64},
    bucket_spec::BucketSpec,
    turns::Vector{Float64},
    reference_turn::Float64,
    machine::NonInteractingTwoRFMachine,
    time_bin_centers_s::Vector{Float64};
    sigma::Real,
    check::Bool,
)
    trajectory = transport_slip_stacked_grid(
        grid,
        bucket_spec,
        turns,
        machine;
        N0=reference_turn,
        check=check,
    )
    assignments = soft_assignments(
        trajectory.delta_t,
        time_bin_centers_s;
        sigma=sigma,
    )

    n_rows = length(turns) * length(time_bin_centers_s)
    return Matrix(reshape(assignments, n_rows, length(grid)))
end


"""
    build_slip_stacked_response(rf1_grid, rf2_grid, bucket_specs, turns,
                                machine, time_bin_centers_s; sigma,
                                N0=nothing, check=true)

Build a combined linear response for independent slip-stacked bunch
distributions. Each block transports one RF family/bucket grid and deposits
its unit-charge columns into the same absolute RWCM time axis.
"""
function build_slip_stacked_response(
    rf1_grid::PhaseSpaceGrid{Float64},
    rf2_grid::PhaseSpaceGrid{Float64},
    bucket_specs::AbstractVector{BucketSpec},
    turns::AbstractVector,
    machine::NonInteractingTwoRFMachine,
    time_bin_centers_s::AbstractVector;
    sigma::Real,
    N0=nothing,
    check::Bool=true,
)
    isempty(bucket_specs) && throw(ArgumentError(
        "At least one bucket specification is required.",
    ))
    length(time_bin_centers_s) > 1 || throw(ArgumentError(
        "At least two time-bin centers are required.",
    ))
    sigma > 0 || throw(ArgumentError("sigma must be positive"))

    turn_vector = Float64.(turns)
    time_bins = Float64.(time_bin_centers_s)
    reference_turn = _reference_turn(turn_vector, N0)

    blocks = Matrix{Float64}[]
    block_ranges = UnitRange{Int}[]
    column_metadata = SlipStackedColumnMetadata[]
    next_column = 1

    for bucket_spec in bucket_specs
        grid = _grid_for_family(bucket_spec.family, rf1_grid, rf2_grid)
        block = _response_block(
            grid,
            bucket_spec,
            turn_vector,
            reference_turn,
            machine,
            time_bins;
            sigma=sigma,
            check=check,
        )
        push!(blocks, block)

        block_range = next_column:(next_column + length(grid) - 1)
        push!(block_ranges, block_range)
        next_column += length(grid)

        append!(
            column_metadata,
            [
                SlipStackedColumnMetadata(
                    bucket_spec.family,
                    bucket_spec.bucket_index,
                    local_cell_index,
                )
                for local_cell_index in 1:length(grid)
            ],
        )
    end

    return SlipStackedGridResponse(
        hcat(blocks...),
        rf1_grid,
        rf2_grid,
        collect(bucket_specs),
        block_ranges,
        column_metadata,
        turn_vector,
        reference_turn,
        time_bins,
        Float64(sigma),
    )
end


function build_slip_stacked_response(
    grid::PhaseSpaceGrid{Float64},
    bucket_specs::AbstractVector{BucketSpec},
    turns::AbstractVector,
    machine::NonInteractingTwoRFMachine,
    time_bin_centers_s::AbstractVector;
    kwargs...,
)
    return build_slip_stacked_response(
        grid,
        grid,
        bucket_specs,
        turns,
        machine,
        time_bin_centers_s;
        kwargs...,
    )
end


function build_slip_stacked_response(
    grids::NamedTuple{(:rf1, :rf2),Tuple{PhaseSpaceGrid{Float64},PhaseSpaceGrid{Float64}}},
    bucket_specs::AbstractVector{BucketSpec},
    turns::AbstractVector,
    machine::NonInteractingTwoRFMachine,
    time_bin_centers_s::AbstractVector;
    kwargs...,
)
    return build_slip_stacked_response(
        grids.rf1,
        grids.rf2,
        bucket_specs,
        turns,
        machine,
        time_bin_centers_s;
        kwargs...,
    )
end


"""
    profiles_from_response(response::SlipStackedGridResponse, weights;
                           normalize=true)

Apply a precomputed slip-stacked response and return an
`(n_turns, n_time_bins)` cascade. When `normalize=true`, the full weight vector
is normalized once across all RF families and bucket blocks.
"""
function profiles_from_response(
    response::SlipStackedGridResponse,
    weights::AbstractVector;
    normalize::Bool=true,
)
    length(weights) == size(response.matrix, 2) || throw(DimensionMismatch(
        "weights must contain one value per slip-stacked response column.",
    ))

    effective_weights = if normalize
        total_weight = sum(weights)
        total_weight > 0 || throw(ArgumentError("weights must have positive sum"))
        weights ./ total_weight
    else
        weights
    end

    n_turns = length(response.turns)
    n_bins = length(response.time_bin_centers_s)
    return reshape(response.matrix * effective_weights, n_turns, n_bins)
end


"""
    uniform_bucket_weights(response; normalize=false)

Return flat independent grid weights that are uniform within each bucket block
and sum within that block to `BucketSpec.charge_fraction`. When
`normalize=true`, the full vector is normalized once globally.
"""
function uniform_bucket_weights(
    response::SlipStackedGridResponse;
    normalize::Bool=false,
)
    weights = vcat([
        fill(
            response.bucket_specs[i].charge_fraction /
                length(response.block_ranges[i]),
            length(response.block_ranges[i]),
        )
        for i in eachindex(response.bucket_specs)
    ]...)

    if normalize
        total_weight = sum(weights)
        total_weight > 0 || throw(ArgumentError(
            "Bucket charge fractions must have positive sum.",
        ))
        weights = weights ./ total_weight
    end

    return weights
end


"""
    direct_slip_stacked_profiles(rf1_grid, rf2_grid, bucket_specs, weights,
                                 turns, machine, time_bin_centers_s; sigma,
                                 N0=nothing, normalize=true, check=true)

Direct validation path for the slip-stacked response matrix. It transports and
deposits the same local grid nodes without using the precomputed matrix.
"""
function direct_slip_stacked_profiles(
    rf1_grid::PhaseSpaceGrid{Float64},
    rf2_grid::PhaseSpaceGrid{Float64},
    bucket_specs::AbstractVector{BucketSpec},
    weights::AbstractVector,
    turns::AbstractVector,
    machine::NonInteractingTwoRFMachine,
    time_bin_centers_s::AbstractVector;
    sigma::Real,
    N0=nothing,
    normalize::Bool=true,
    check::Bool=true,
)
    isempty(bucket_specs) && throw(ArgumentError(
        "At least one bucket specification is required.",
    ))
    length(time_bin_centers_s) > 1 || throw(ArgumentError(
        "At least two time-bin centers are required.",
    ))
    sigma > 0 || throw(ArgumentError("sigma must be positive"))

    turn_vector = Float64.(turns)
    time_bins = Float64.(time_bin_centers_s)
    reference_turn = _reference_turn(turn_vector, N0)

    expected_length = sum(
        length(_grid_for_family(spec.family, rf1_grid, rf2_grid))
        for spec in bucket_specs
    )
    length(weights) == expected_length || throw(DimensionMismatch(
        "weights must contain one value per slip-stacked response column.",
    ))

    effective_weights = if normalize
        total_weight = sum(weights)
        total_weight > 0 || throw(ArgumentError("weights must have positive sum"))
        weights ./ total_weight
    else
        weights
    end

    delta_t_blocks = Matrix{Float64}[]
    weight_blocks = Vector{Float64}[]
    start_index = 1

    for bucket_spec in bucket_specs
        grid = _grid_for_family(bucket_spec.family, rf1_grid, rf2_grid)
        stop_index = start_index + length(grid) - 1

        trajectory = transport_slip_stacked_grid(
            grid,
            bucket_spec,
            turn_vector,
            machine;
            N0=reference_turn,
            check=check,
        )
        push!(delta_t_blocks, trajectory.delta_t)
        push!(weight_blocks, Float64.(effective_weights[start_index:stop_index]))

        start_index = stop_index + 1
    end

    return soft_histogram(
        hcat(delta_t_blocks...),
        time_bins;
        sigma=sigma,
        particle_weights=vcat(weight_blocks...),
        normalize=false,
    )
end


function direct_slip_stacked_profiles(
    response::SlipStackedGridResponse,
    weights::AbstractVector,
    machine::NonInteractingTwoRFMachine;
    normalize::Bool=true,
    check::Bool=true,
)
    return direct_slip_stacked_profiles(
        response.rf1_grid,
        response.rf2_grid,
        response.bucket_specs,
        weights,
        response.turns,
        machine,
        response.time_bin_centers_s;
        sigma=response.sigma_s,
        N0=response.reference_turn,
        normalize=normalize,
        check=check,
    )
end


"""
    slip_stacked_grid_total_variation(weights, response; epsilon=1e-12)

Sum the existing local-grid total variation penalty over bucket blocks. No edge
is introduced between distinct RF families or neighboring bucket blocks.
"""
function slip_stacked_grid_total_variation(
    weights::AbstractVector,
    response::SlipStackedGridResponse;
    epsilon::Real=1e-12,
)
    length(weights) == size(response.matrix, 2) || throw(DimensionMismatch(
        "weights must contain one value per slip-stacked response column.",
    ))

    return sum(eachindex(response.bucket_specs)) do block_index
        bucket_spec = response.bucket_specs[block_index]
        grid = _grid_for_family(
            bucket_spec.family,
            response.rf1_grid,
            response.rf2_grid,
        )
        grid_total_variation(
            weights[response.block_ranges[block_index]],
            grid;
            epsilon=epsilon,
        )
    end
end


function _slip_stacked_target_charge(
    response::SlipStackedGridResponse,
    target_profiles::AbstractMatrix,
    total_charge,
)
    expected_size = (
        length(response.turns),
        length(response.time_bin_centers_s),
    )
    size(target_profiles) == expected_size || throw(DimensionMismatch(
        "target_profiles must have size $expected_size.",
    ))
    all(isfinite, target_profiles) || throw(ArgumentError(
        "target_profiles must contain only finite values.",
    ))

    effective_charge = if isnothing(total_charge)
        # Each row is one measured profile. Averaging the row sums is more
        # tolerant of small numerical charge errors than selecting one turn.
        sum(target_profiles) / size(target_profiles, 1)
    else
        total_charge isa Real || throw(ArgumentError(
            "total_charge must be a real number or nothing.",
        ))
        total_charge
    end

    isfinite(effective_charge) || throw(ArgumentError(
        "The inferred total charge must be finite.",
    ))
    effective_charge > 0 || throw(ArgumentError(
        "The inferred total charge must be positive.",
    ))
    return effective_charge
end


"""
    slip_stacked_weights_from_logits(logits, response;
                                     total_charge=1,
                                     charge_mode=:learned)

Convert unconstrained logits into nonnegative slip-stacked grid masses.

With `charge_mode=:learned`, one global softmax is used, so tomography infers
how the specified total charge is shared by the RF families and bucket blocks.
This is the parameterization used by the original notebook demonstration.

With `charge_mode=:fixed`, each bucket block has its own softmax and its total
mass is fixed in proportion to `BucketSpec.charge_fraction`. The distribution
within every block is still reconstructed.
"""
function slip_stacked_weights_from_logits(
    logits::AbstractVector,
    response::SlipStackedGridResponse;
    total_charge::Real=1.0,
    charge_mode::Symbol=:learned,
)
    n_unknowns = size(response.matrix, 2)
    length(logits) == n_unknowns || throw(DimensionMismatch(
        "logits must contain one value per slip-stacked response column.",
    ))
    isfinite(total_charge) || throw(ArgumentError(
        "total_charge must be finite.",
    ))
    total_charge > 0 || throw(ArgumentError(
        "total_charge must be positive.",
    ))

    if charge_mode === :learned
        return total_charge .* grid_weights_from_logits(logits)
    elseif charge_mode === :fixed
        specified_charge = sum(
            spec.charge_fraction for spec in response.bucket_specs
        )
        specified_charge > 0 || throw(ArgumentError(
            "At least one BucketSpec charge_fraction must be positive " *
            "when charge_mode=:fixed.",
        ))
        charge_scale = total_charge / specified_charge

        # Construct blocks without mutation so this path remains compatible
        # with Zygote reverse-mode differentiation.
        weight_blocks = map(eachindex(response.bucket_specs)) do block_index
            spec = response.bucket_specs[block_index]
            block_range = response.block_ranges[block_index]
            block_shape = grid_weights_from_logits(logits[block_range])
            charge_scale .* spec.charge_fraction .* block_shape
        end
        return vcat(weight_blocks...)
    end

    throw(ArgumentError(
        "charge_mode must be :learned or :fixed.",
    ))
end


"""
    tomography_loss(logits, response::SlipStackedGridResponse,
                    target_profiles; train_turn_indices=nothing,
                    lambda_tv=0, lambda_entropy=0,
                    total_charge=nothing, charge_mode=:learned)

Slip-stacked profile MSE plus optional total-variation and maximum-entropy
regularization. The target charge is inferred from the mean profile integral
unless `total_charge` is supplied explicitly.

Regularization is evaluated on weights normalized to unit total charge, making
`lambda_tv` and `lambda_entropy` independent of the absolute RWCM intensity
scale.
"""
function tomography_loss(
    logits::AbstractVector,
    response::SlipStackedGridResponse,
    target_profiles::AbstractMatrix;
    train_turn_indices=nothing,
    lambda_tv::Real=0.0,
    lambda_entropy::Real=0.0,
    tv_epsilon::Real=1e-12,
    total_charge=nothing,
    charge_mode::Symbol=:learned,
)
    lambda_tv >= 0 || throw(ArgumentError("lambda_tv must be nonnegative"))
    lambda_entropy >= 0 || throw(ArgumentError(
        "lambda_entropy must be nonnegative",
    ))

    effective_charge = _slip_stacked_target_charge(
        response,
        target_profiles,
        total_charge,
    )
    weights = slip_stacked_weights_from_logits(
        logits,
        response;
        total_charge=effective_charge,
        charge_mode=charge_mode,
    )
    prediction = profiles_from_response(response, weights; normalize=false)
    data_loss = profile_mse(
        prediction,
        target_profiles;
        turn_indices=train_turn_indices,
    )

    normalized_weights = weights ./ effective_charge
    tv_loss = slip_stacked_grid_total_variation(
        normalized_weights,
        response;
        epsilon=tv_epsilon,
    )
    entropy_loss = grid_entropy_penalty(normalized_weights)

    return data_loss + lambda_tv * tv_loss + lambda_entropy * entropy_loss
end


"""
    tomography_diagnostics(logits, response::SlipStackedGridResponse,
                           target_profiles; kwargs...)

Return the slip-stacked loss components, reconstructed cell masses, predicted
profiles, and inferred charge in each bucket block.
"""
function tomography_diagnostics(
    logits::AbstractVector,
    response::SlipStackedGridResponse,
    target_profiles::AbstractMatrix;
    train_turn_indices=nothing,
    lambda_tv::Real=0.0,
    lambda_entropy::Real=0.0,
    tv_epsilon::Real=1e-12,
    total_charge=nothing,
    charge_mode::Symbol=:learned,
)
    lambda_tv >= 0 || throw(ArgumentError("lambda_tv must be nonnegative"))
    lambda_entropy >= 0 || throw(ArgumentError(
        "lambda_entropy must be nonnegative",
    ))

    effective_charge = _slip_stacked_target_charge(
        response,
        target_profiles,
        total_charge,
    )
    weights = slip_stacked_weights_from_logits(
        logits,
        response;
        total_charge=effective_charge,
        charge_mode=charge_mode,
    )
    prediction = profiles_from_response(response, weights; normalize=false)
    data_loss = profile_mse(
        prediction,
        target_profiles;
        turn_indices=train_turn_indices,
    )

    normalized_weights = weights ./ effective_charge
    tv_loss = slip_stacked_grid_total_variation(
        normalized_weights,
        response;
        epsilon=tv_epsilon,
    )
    entropy_loss = grid_entropy_penalty(normalized_weights)
    total_loss = data_loss + lambda_tv * tv_loss + lambda_entropy * entropy_loss
    block_charges = [
        sum(weights[block_range]) for block_range in response.block_ranges
    ]

    return (
        total=total_loss,
        data=data_loss,
        total_variation=tv_loss,
        entropy=entropy_loss,
        weights=weights,
        profiles=prediction,
        total_charge=effective_charge,
        block_charges=block_charges,
        charge_mode=charge_mode,
    )
end


"""
    reconstruct_grid(response::SlipStackedGridResponse, target_profiles;
                     iterations=500, learning_rate=0.05, ...)

Reconstruct all RF-family and bucket-block cell masses using Adam applied to
unconstrained logits. With the default `charge_mode=:learned`, the block charge
sharing is reconstructed along with the phase-space shapes. Set
`charge_mode=:fixed` to enforce the relative charges stored in the bucket
specifications.
"""
function reconstruct_grid(
    response::SlipStackedGridResponse,
    target_profiles::AbstractMatrix;
    iterations::Integer=500,
    learning_rate::Real=0.05,
    initial_logits=nothing,
    train_turn_indices=nothing,
    lambda_tv::Real=0.0,
    lambda_entropy::Real=0.0,
    tv_epsilon::Real=1e-12,
    total_charge=nothing,
    charge_mode::Symbol=:learned,
    beta1::Real=0.9,
    beta2::Real=0.999,
    adam_epsilon::Real=1e-8,
)
    iterations >= 1 || throw(ArgumentError("iterations must be positive"))
    learning_rate > 0 || throw(ArgumentError("learning_rate must be positive"))
    0 <= beta1 < 1 || throw(ArgumentError("beta1 must lie in [0,1)"))
    0 <= beta2 < 1 || throw(ArgumentError("beta2 must lie in [0,1)"))
    adam_epsilon > 0 || throw(ArgumentError("adam_epsilon must be positive"))

    effective_charge = _slip_stacked_target_charge(
        response,
        target_profiles,
        total_charge,
    )
    n_unknowns = size(response.matrix, 2)
    logits = if isnothing(initial_logits)
        zeros(Float64, n_unknowns)
    else
        length(initial_logits) == n_unknowns || throw(DimensionMismatch(
            "initial_logits must contain one value per slip-stacked " *
            "response column.",
        ))
        Float64.(initial_logits)
    end
    all(isfinite, logits) || throw(ArgumentError(
        "initial_logits must contain only finite values.",
    ))

    first_moment = zeros(Float64, n_unknowns)
    second_moment = zeros(Float64, n_unknowns)
    loss_history = Float64[]

    objective = current_logits -> tomography_loss(
        current_logits,
        response,
        target_profiles;
        train_turn_indices=train_turn_indices,
        lambda_tv=lambda_tv,
        lambda_entropy=lambda_entropy,
        tv_epsilon=tv_epsilon,
        total_charge=effective_charge,
        charge_mode=charge_mode,
    )

    for iteration in 1:Int(iterations)
        loss_value, pullback = Zygote.pullback(objective, logits)
        gradient = only(pullback(one(loss_value)))
        isfinite(loss_value) || throw(ErrorException(
            "Non-finite loss encountered at iteration $iteration.",
        ))
        all(isfinite, gradient) || throw(ErrorException(
            "Non-finite gradient encountered at iteration $iteration.",
        ))
        push!(loss_history, Float64(loss_value))

        first_moment = beta1 .* first_moment .+ (1 - beta1) .* gradient
        second_moment = beta2 .* second_moment .+ (1 - beta2) .* gradient.^2
        corrected_first = first_moment ./ (1 - beta1^iteration)
        corrected_second = second_moment ./ (1 - beta2^iteration)
        logits = logits .- learning_rate .* corrected_first ./
            (sqrt.(corrected_second) .+ adam_epsilon)
    end

    diagnostics = tomography_diagnostics(
        logits,
        response,
        target_profiles;
        train_turn_indices=train_turn_indices,
        lambda_tv=lambda_tv,
        lambda_entropy=lambda_entropy,
        tv_epsilon=tv_epsilon,
        total_charge=effective_charge,
        charge_mode=charge_mode,
    )
    return merge(diagnostics, (
        logits=logits,
        loss_history=loss_history,
        train_turn_indices=train_turn_indices,
    ))
end
