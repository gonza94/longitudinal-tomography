"""
    PhaseSpaceGrid

Cartesian cell-center grid restricted to the interior of the normalized
single-RF separatrix. Matrix-valued fields use `(P index, theta index)` order.
`theta_nodes` and `P_nodes` contain only active cells, in the same order as the
tomographic weights.
"""
struct PhaseSpaceGrid{T<:AbstractFloat}
    theta_axis::Vector{T}
    P_axis::Vector{T}
    mask::BitMatrix
    theta_nodes::Vector{T}
    P_nodes::Vector{T}
    index_map::Matrix{Int}
    edge_from::Vector{Int}
    edge_to::Vector{Int}
    energy_margin::T
end

Base.length(grid::PhaseSpaceGrid) = length(grid.theta_nodes)


"""
    separatrix_grid(; n_theta=41, n_P=41, energy_margin=1e-3,
                    theta_limits=(-pi, pi), P_limits=(-2, 2))

Construct a Cartesian grid using cell centers and retain cells satisfying
`pendulum_energy(theta, P) < 1-energy_margin`. A nonzero margin avoids the
`k -> 1` singular limit of the elliptic libration solution.
"""
function separatrix_grid(;
    n_theta::Integer=41,
    n_P::Integer=41,
    energy_margin::Real=1e-3,
    theta_limits=(-pi, pi),
    P_limits=(-2.0, 2.0),
)
    n_theta >= 3 || throw(ArgumentError("n_theta must be at least 3"))
    n_P >= 3 || throw(ArgumentError("n_P must be at least 3"))
    0 <= energy_margin < 1 || throw(ArgumentError(
        "energy_margin must lie in [0,1).",
    ))
    theta_limits[1] < theta_limits[2] || throw(ArgumentError(
        "theta_limits must be increasing.",
    ))
    P_limits[1] < P_limits[2] || throw(ArgumentError(
        "P_limits must be increasing.",
    ))

    theta_edges = collect(range(
        Float64(theta_limits[1]),
        Float64(theta_limits[2]);
        length=Int(n_theta) + 1,
    ))
    P_edges = collect(range(
        Float64(P_limits[1]),
        Float64(P_limits[2]);
        length=Int(n_P) + 1,
    ))
    theta_axis = (theta_edges[1:end-1] .+ theta_edges[2:end]) ./ 2
    P_axis = (P_edges[1:end-1] .+ P_edges[2:end]) ./ 2

    mask = BitMatrix([
        pendulum_energy(theta_axis[j_theta], P_axis[i_P]) < (1 - energy_margin)
        for i_P in eachindex(P_axis), j_theta in eachindex(theta_axis)
    ])
    any(mask) || throw(ArgumentError("No grid cells lie inside the separatrix."))

    active_cells = findall(mask)
    theta_nodes = [theta_axis[cell[2]] for cell in active_cells]
    P_nodes = [P_axis[cell[1]] for cell in active_cells]

    index_map = zeros(Int, size(mask))
    for (active_index, cell) in enumerate(active_cells)
        index_map[cell] = active_index
    end

    # Store each horizontal and vertical active-cell pair once. These edges
    # define total-variation regularization without mutation in the AD path.
    edge_from = Int[]
    edge_to = Int[]
    for j_theta in axes(index_map, 2), i_P in axes(index_map, 1)
        source = index_map[i_P, j_theta]
        source == 0 && continue

        if j_theta < size(index_map, 2)
            destination = index_map[i_P, j_theta + 1]
            if destination != 0
                push!(edge_from, source)
                push!(edge_to, destination)
            end
        end
        if i_P < size(index_map, 1)
            destination = index_map[i_P + 1, j_theta]
            if destination != 0
                push!(edge_from, source)
                push!(edge_to, destination)
            end
        end
    end

    return PhaseSpaceGrid(
        theta_axis,
        P_axis,
        mask,
        theta_nodes,
        P_nodes,
        index_map,
        edge_from,
        edge_to,
        Float64(energy_margin),
    )
end


"""
    grid_weight_matrix(grid, weights; fill_value=0)

Scatter active-cell weights into a matrix with shape `(n_P, n_theta)`. This is
intended for diagnostics and plotting, not for use inside a differentiated
loss.
"""
function grid_weight_matrix(
    grid::PhaseSpaceGrid,
    weights::AbstractVector;
    fill_value::Real=0.0,
)
    length(weights) == length(grid) || throw(DimensionMismatch(
        "weights must contain one value per active grid cell.",
    ))
    element_type = promote_type(eltype(weights), typeof(float(fill_value)))
    matrix = fill(convert(element_type, fill_value), size(grid.mask))
    matrix[grid.mask] .= weights
    return matrix
end


"""
    GridResponse

Precomputed linear operator mapping initial grid-cell masses to a complete
turn-by-turn RWCM cascade. `matrix` has shape
`(n_turns*n_time_bins, n_active_cells)`. Reshaping a product with
`(n_turns, n_time_bins)` recovers the profile cascade.
"""
struct GridResponse{T<:AbstractFloat}
    matrix::Matrix{T}
    grid::PhaseSpaceGrid{T}
    turns::Vector{T}
    time_bin_centers_s::Vector{T}
    sigma_s::T
end


"""
    build_response_matrix(grid, turns, machine, time_bin_centers_s;
                          sigma, check=true)

Transport every active grid cell analytically and construct its normalized
soft-histogram response at every requested turn. Dynamics and measurement
binning are evaluated once; grid-weight reconstruction subsequently requires
only matrix-vector products.
"""
function build_response_matrix(
    grid::PhaseSpaceGrid{Float64},
    turns::AbstractVector,
    machine::SingleRFMachine,
    time_bin_centers_s::AbstractVector;
    sigma::Real,
    check::Bool=true,
)
    isempty(turns) && throw(ArgumentError("At least one turn is required."))
    length(time_bin_centers_s) > 1 || throw(ArgumentError(
        "At least two time-bin centers are required.",
    ))
    sigma > 0 || throw(ArgumentError("sigma must be positive"))

    turn_vector = Float64.(turns)
    time_bins = Float64.(time_bin_centers_s)
    tau = tau_from_turn(turn_vector, machine)
    trajectory = libration_transport(
        grid.theta_nodes,
        grid.P_nodes,
        tau;
        check=check,
        margin=grid.energy_margin,
    )
    delta_t = delta_t_from_theta(trajectory.theta, machine)
    assignments = soft_assignments(delta_t, time_bins; sigma=sigma)

    n_rows = length(turn_vector) * length(time_bins)
    response_matrix = Matrix(reshape(assignments, n_rows, length(grid)))

    return GridResponse(
        response_matrix,
        grid,
        turn_vector,
        time_bins,
        Float64(sigma),
    )
end


"""
    grid_weights_from_logits(logits)

Stable softmax parameterization of nonnegative grid masses whose sum is one.
"""
function grid_weights_from_logits(logits::AbstractVector)
    isempty(logits) && throw(ArgumentError("At least one logit is required."))
    shifted = logits .- maximum(logits)
    exponentials = exp.(shifted)
    return exponentials ./ sum(exponentials)
end


"""
    profiles_from_response(response, weights; normalize=true)

Apply a precomputed response and return an `(n_turns, n_time_bins)` cascade.
When `normalize=true`, `weights` are interpreted as nonnegative cell masses and
are normalized to unit total charge before projection.
"""
function profiles_from_response(
    response::GridResponse,
    weights::AbstractVector;
    normalize::Bool=true,
)
    length(weights) == length(response.grid) || throw(DimensionMismatch(
        "weights must contain one value per active grid cell.",
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


"""Mean squared profile mismatch, optionally evaluated on complete selected turns."""
function profile_mse(
    prediction::AbstractMatrix,
    target::AbstractMatrix;
    turn_indices=nothing,
)
    size(prediction) == size(target) || throw(DimensionMismatch(
        "prediction and target profiles must have equal size.",
    ))
    residual = if isnothing(turn_indices)
        prediction .- target
    else
        prediction[turn_indices, :] .- target[turn_indices, :]
    end
    isempty(residual) && throw(ArgumentError("No profile samples were selected."))
    return sum(abs2, residual) / length(residual)
end


"""
    grid_total_variation(weights, grid; epsilon=1e-12)

Smoothed mean total variation over horizontal and vertical pairs of active
cells. The small `epsilon` gives a finite derivative when neighboring weights
are equal.
"""
function grid_total_variation(
    weights::AbstractVector,
    grid::PhaseSpaceGrid;
    epsilon::Real=1e-12,
)
    length(weights) == length(grid) || throw(DimensionMismatch(
        "weights must contain one value per active grid cell.",
    ))
    epsilon > 0 || throw(ArgumentError("epsilon must be positive"))
    isempty(grid.edge_from) && return zero(eltype(weights))

    differences = weights[grid.edge_from] .- weights[grid.edge_to]
    return sum(sqrt.(differences.^2 .+ epsilon^2) .- epsilon) /
        length(differences)
end


"""
    grid_entropy_penalty(weights; epsilon=1e-15)

Kullback-Leibler divergence from the uniform active-cell distribution. It is
nonnegative up to roundoff and provides a weak maximum-entropy regularizer.
"""
function grid_entropy_penalty(weights::AbstractVector; epsilon::Real=1e-15)
    epsilon > 0 || throw(ArgumentError("epsilon must be positive"))
    n_cells = length(weights)
    n_cells > 0 || throw(ArgumentError("At least one weight is required."))
    return sum(weights .* log.(n_cells .* weights .+ epsilon))
end


"""
    tomography_loss(logits, response, target_profiles;
                    train_turn_indices=nothing, lambda_tv=0,
                    lambda_entropy=0)

Profile MSE plus optional total-variation and maximum-entropy regularization.
`logits` are converted to physical cell masses with a softmax.
"""
function tomography_loss(
    logits::AbstractVector,
    response::GridResponse,
    target_profiles::AbstractMatrix;
    train_turn_indices=nothing,
    lambda_tv::Real=0.0,
    lambda_entropy::Real=0.0,
    tv_epsilon::Real=1e-12,
)
    lambda_tv >= 0 || throw(ArgumentError("lambda_tv must be nonnegative"))
    lambda_entropy >= 0 || throw(ArgumentError(
        "lambda_entropy must be nonnegative",
    ))

    weights = grid_weights_from_logits(logits)
    prediction = profiles_from_response(response, weights; normalize=false)
    data_loss = profile_mse(
        prediction,
        target_profiles;
        turn_indices=train_turn_indices,
    )
    tv_loss = grid_total_variation(weights, response.grid; epsilon=tv_epsilon)
    entropy_loss = grid_entropy_penalty(weights)
    return data_loss + lambda_tv * tv_loss + lambda_entropy * entropy_loss
end


"""Return the individual reconstruction loss terms and predicted profiles."""
function tomography_diagnostics(
    logits::AbstractVector,
    response::GridResponse,
    target_profiles::AbstractMatrix;
    train_turn_indices=nothing,
    lambda_tv::Real=0.0,
    lambda_entropy::Real=0.0,
    tv_epsilon::Real=1e-12,
)
    weights = grid_weights_from_logits(logits)
    prediction = profiles_from_response(response, weights; normalize=false)
    data_loss = profile_mse(
        prediction,
        target_profiles;
        turn_indices=train_turn_indices,
    )
    tv_loss = grid_total_variation(weights, response.grid; epsilon=tv_epsilon)
    entropy_loss = grid_entropy_penalty(weights)
    total_loss = data_loss + lambda_tv * tv_loss + lambda_entropy * entropy_loss
    return (
        total=total_loss,
        data=data_loss,
        total_variation=tv_loss,
        entropy=entropy_loss,
        weights=weights,
        profiles=prediction,
    )
end


"""
    reconstruct_grid(response, target_profiles; iterations=500,
                     learning_rate=0.05, ...)

Reconstruct grid-cell masses with Adam applied to unconstrained logits. This
small dependency-free optimizer is intended as a transparent tomography
baseline; later neural-generator phases can replace it without changing the
response or loss APIs.
"""
function reconstruct_grid(
    response::GridResponse,
    target_profiles::AbstractMatrix;
    iterations::Integer=500,
    learning_rate::Real=0.05,
    initial_logits=nothing,
    train_turn_indices=nothing,
    lambda_tv::Real=0.0,
    lambda_entropy::Real=0.0,
    tv_epsilon::Real=1e-12,
    beta1::Real=0.9,
    beta2::Real=0.999,
    adam_epsilon::Real=1e-8,
)
    iterations >= 1 || throw(ArgumentError("iterations must be positive"))
    learning_rate > 0 || throw(ArgumentError("learning_rate must be positive"))
    0 <= beta1 < 1 || throw(ArgumentError("beta1 must lie in [0,1)"))
    0 <= beta2 < 1 || throw(ArgumentError("beta2 must lie in [0,1)"))
    adam_epsilon > 0 || throw(ArgumentError("adam_epsilon must be positive"))

    n_cells = length(response.grid)
    logits = if isnothing(initial_logits)
        zeros(Float64, n_cells)
    else
        length(initial_logits) == n_cells || throw(DimensionMismatch(
            "initial_logits must contain one value per active grid cell.",
        ))
        Float64.(initial_logits)
    end

    first_moment = zeros(Float64, n_cells)
    second_moment = zeros(Float64, n_cells)
    loss_history = Float64[]

    objective = current_logits -> tomography_loss(
        current_logits,
        response,
        target_profiles;
        train_turn_indices=train_turn_indices,
        lambda_tv=lambda_tv,
        lambda_entropy=lambda_entropy,
        tv_epsilon=tv_epsilon,
    )

    for iteration in 1:Int(iterations)
        loss_value, pullback = Zygote.pullback(objective, logits)
        gradient = only(pullback(one(loss_value)))
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
    )
    return merge(diagnostics, (
        logits=logits,
        loss_history=loss_history,
        train_turn_indices=train_turn_indices,
    ))
end
