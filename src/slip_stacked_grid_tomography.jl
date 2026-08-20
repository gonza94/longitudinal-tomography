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
