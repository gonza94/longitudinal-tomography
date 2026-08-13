"""
    NonInteractingTwoRFMachine

Two independent stationary RF buckets represented in the phase-space
coordinates of RF1. The approximation deliberately omits the force of RF2 on
family 1 and the force of RF1 on family 2.

The signed slip frequency is defined as

    f_slip_hz = f_rf1 - f_rf2.

Therefore, a lower-frequency RF2 has `f_slip_hz > 0`. `relative_phase0` is the
RF2 bucket-center phase measured in RF1 coordinates at the reference turn used
by [`noninteracting_two_rf_transport`](@ref).
"""
struct NonInteractingTwoRFMachine
    rf1::SingleRFMachine
    rf2::SingleRFMachine
    f_slip_hz::Float64
    relative_phase0::Float64
end


function NonInteractingTwoRFMachine(
    rf1::SingleRFMachine,
    rf2::SingleRFMachine;
    f_slip_hz::Real,
    relative_phase0::Real=0.0,
)
    rf1.harmonic == rf2.harmonic || throw(ArgumentError(
        "The noninteracting model requires equal harmonic numbers.",
    ))

    shared_parameters = (
        (:f_rev_hz, rf1.f_rev_hz, rf2.f_rev_hz),
        (:eta0, rf1.eta0, rf2.eta0),
        (:beta_s, rf1.beta_s, rf2.beta_s),
        (:E_s_eV, rf1.E_s_eV, rf2.E_s_eV),
        (:phi_s, rf1.phi_s, rf2.phi_s),
    )

    for (name, value1, value2) in shared_parameters
        isapprox(value1, value2; rtol=1e-12, atol=0.0) || throw(ArgumentError(
            "RF1 and RF2 must have the same $(name).",
        ))
    end

    isfinite(f_slip_hz) || throw(ArgumentError("f_slip_hz must be finite."))
    isfinite(relative_phase0) || throw(ArgumentError(
        "relative_phase0 must be finite.",
    ))

    return NonInteractingTwoRFMachine(
        rf1,
        rf2,
        Float64(f_slip_hz),
        Float64(relative_phase0),
    )
end


"""
    NonInteractingTwoRFMachine(rf1; f_slip_hz, rf2_kick_eV,
                               relative_phase0=0)

Convenience constructor that copies the Recycler parameters from RF1 and only
changes the RF2 voltage. Equal voltages are used by default.
"""
function NonInteractingTwoRFMachine(
    rf1::SingleRFMachine;
    f_slip_hz::Real,
    rf2_kick_eV::Real=rf1.rf_kick_eV,
    relative_phase0::Real=0.0,
)
    rf2 = SingleRFMachine(
        harmonic=rf1.harmonic,
        f_rev_hz=rf1.f_rev_hz,
        eta0=rf1.eta0,
        beta_s=rf1.beta_s,
        E_s_eV=rf1.E_s_eV,
        rf_kick_eV=rf2_kick_eV,
        phi_s=rf1.phi_s,
        delta_t_s=rf1.delta_t_s,
    )

    return NonInteractingTwoRFMachine(
        rf1,
        rf2;
        f_slip_hz=f_slip_hz,
        relative_phase0=relative_phase0,
    )
end


"""
    slip_tune(machine)

Signed Lee--Ng slip tune

    nu_slip = (f_rf1 - f_rf2) / f_rev.
"""
slip_tune(machine::NonInteractingTwoRFMachine) =
    machine.f_slip_hz / machine.rf1.f_rev_hz

# Lee--Ng notation retained as a convenient API alias.
nu_slip(machine::NonInteractingTwoRFMachine) = slip_tune(machine)


"""
    slip_stacking_parameter(machine)

Lee--Ng slip-stacking parameter and RF2 bucket-center separation in the
normalized RF1 momentum:

    alpha_s = nu_slip / nu_s1.
"""
slip_stacking_parameter(machine::NonInteractingTwoRFMachine) =
    slip_tune(machine) / synchrotron_tune(machine.rf1)

# Lee--Ng notation retained as a convenient API alias.
alpha_s(machine::NonInteractingTwoRFMachine) =
    slip_stacking_parameter(machine)


"""
    fractional_momentum_separation(machine)

Signed physical separation `delta_sep = Delta p_sep / p0`. Below transition,
a lower-frequency RF2 gives a negative value even though its Lee-normalized
momentum center is positive.
"""
fractional_momentum_separation(machine::NonInteractingTwoRFMachine) =
    slip_tune(machine) / (machine.rf1.harmonic * machine.rf1.eta0)


"""
    energy_separation_eV(machine)

Approximate signed RF2 energy-center offset relative to RF1, using
`Delta E = beta_s^2 E_s delta`.
"""
energy_separation_eV(machine::NonInteractingTwoRFMachine) =
    machine.rf1.beta_s^2 * machine.rf1.E_s_eV *
    fractional_momentum_separation(machine)


"""
    rf2_center_phase(turns, machine; N0=0)

Unwrapped RF2 bucket-center phase in RF1 coordinates.
"""
rf2_center_phase(turns, machine::NonInteractingTwoRFMachine; N0=0) =
    machine.relative_phase0 .+ 2pi * slip_tune(machine) .* (turns .- N0)


"""
    rf2_center_P(machine)

RF2 bucket-center momentum in Lee--Ng normalized RF1 coordinates.
"""
rf2_center_P(machine::NonInteractingTwoRFMachine) =
    slip_stacking_parameter(machine)


"""Wrap phase to the half-open interval `[-pi, pi)`."""
wrap_to_pi(phase) = mod.(phase .+ pi, 2pi) .- pi

"""
Convert energy deviation ΔE [eV] to fractional momentum deviation Δp/p₀.

Uses the first-order relation

    Δp/p₀ = ΔE / (βₛ² Eₛ)
"""
function delta_p_over_p0_from_delta_E(delta_E, machine::SingleRFMachine)
    return delta_E ./ (machine.beta_s^2 * machine.E_s_eV)
end

"""
    noninteracting_two_rf_transport(
        theta1_0, P1_0, theta2_0, P2_0, turns, machine;
        N0=nothing, check=true, margin=1e-10,
    )

Transport two uncoupled particle families. Initial coordinates for each family
are local to that family's bucket at reference turn `N0`. The RF2 solution is
then translated into RF1 coordinates:

    theta2_RF1(N) = theta2_local(N)
                    + 2pi*nu_slip*(N-N0) + relative_phase0

    P2_RF1(N) = (nu_s2/nu_s1)*P2_local(N) + alpha_s.

The momentum scale factor makes the transformation valid when the two RF
voltages differ. It equals one for the equal-voltage Lee--Ng case.

Both wrapped and unwrapped RF2 phases are returned. The wrapped phase is useful
for one-period RWCM plots; the unwrapped phase displays the continuous slipping
motion.
"""
function noninteracting_two_rf_transport(
    theta1_0::AbstractVector,
    P1_0::AbstractVector,
    theta2_0::AbstractVector,
    P2_0::AbstractVector,
    turns::AbstractVector,
    machine::NonInteractingTwoRFMachine;
    N0=nothing,
    check::Bool=true,
    margin::Real=1e-10,
)
    isempty(turns) && throw(ArgumentError(
        "At least one observation turn is required.",
    ))

    reference_turn = isnothing(N0) ? first(turns) : N0
    reference_turn isa Real || throw(ArgumentError("N0 must be a real turn."))

    tau1 = tau_from_turn(turns, machine.rf1; N0=reference_turn)
    tau2 = tau_from_turn(turns, machine.rf2; N0=reference_turn)

    trajectory1 = libration_transport(
        theta1_0,
        P1_0,
        tau1;
        check=check,
        margin=margin,
    )

    trajectory2_local = libration_transport(
        theta2_0,
        P2_0,
        tau2;
        check=check,
        margin=margin,
    )

    center_phase = rf2_center_phase(turns, machine; N0=reference_turn)
    center_phase_grid = reshape(center_phase, length(turns), 1)

    theta2_rf1_unwrapped = trajectory2_local.theta .+ center_phase_grid
    theta2_rf1_wrapped = wrap_to_pi(theta2_rf1_unwrapped)

    nu_s1 = synchrotron_tune(machine.rf1)
    nu_s2 = synchrotron_tune(machine.rf2)
    momentum_scale = nu_s2 / nu_s1
    alpha_s = slip_stacking_parameter(machine)
    P2_rf1 = momentum_scale .* trajectory2_local.P .+ alpha_s

    return (
        turns=turns,
        N0=reference_turn,
        nu_slip=slip_tune(machine),
        alpha_s=alpha_s,
        slip_phase=center_phase,
        family1=(
            theta=trajectory1.theta,
            P=trajectory1.P,
            delta_t=delta_t_from_theta(trajectory1.theta, machine.rf1),
            delta_E=delta_energy_from_P(trajectory1.P, machine.rf1),
            energy=trajectory1.energy,
            modulus=trajectory1.modulus,
        ),
        family2=(
            theta_local=trajectory2_local.theta,
            P_local=trajectory2_local.P,
            theta_rf1_unwrapped=theta2_rf1_unwrapped,
            theta_rf1_wrapped=theta2_rf1_wrapped,
            P_rf1=P2_rf1,
            delta_t_rf1_unwrapped=delta_t_from_theta(
                theta2_rf1_unwrapped,
                machine.rf1,
            ),
            delta_t_rf1_wrapped=delta_t_from_theta(
                theta2_rf1_wrapped,
                machine.rf1,
            ),
            delta_E_local=delta_energy_from_P(
                trajectory2_local.P,
                machine.rf2,
            ),
            delta_E_rf1=delta_energy_from_P(P2_rf1, machine.rf1),
            energy=trajectory2_local.energy,
            modulus=trajectory2_local.modulus,
            momentum_scale=momentum_scale,
        ),
    )
end

const SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


function place_rf1_bunches(
    trajectory,
    bucket_indices,
    rf1_machine::SingleRFMachine,
)
    return map(bucket_indices) do bucket_index
        theta_absolute =
            trajectory.family1.theta .+
            2π * bucket_index

        delta_t_absolute =
            delta_t_from_theta(theta_absolute, rf1_machine)

        delta_E_rf1 = trajectory.family1.delta_E

        (
            family=:rf1,
            bucket_index=bucket_index,

            theta_local=trajectory.family1.theta,
            theta_absolute=theta_absolute,
            P=trajectory.family1.P,

            delta_t=delta_t_absolute,
            delta_E=delta_E_rf1,

            delta_p_over_p0=delta_p_over_p0_from_delta_E(
                delta_E_rf1,
                rf1_machine,
            ),

            z=-rf1_machine.beta_s *
              SPEED_OF_LIGHT_M_PER_S .*
              delta_t_absolute,
        )
    end
end


function place_rf2_bunches(
    trajectory,
    bucket_indices,
    rf1_machine::SingleRFMachine,
)
    return map(bucket_indices) do bucket_index
        theta_absolute =
            trajectory.family2.theta_rf1_unwrapped .+
            2π * bucket_index

        delta_t_absolute =
            delta_t_from_theta(theta_absolute, rf1_machine)

        # Energy deviation measured relative to the RF1 reference energy.
        # This includes the momentum separation between the bucket families.
        delta_E_rf1 = trajectory.family2.delta_E_rf1

        (
            family=:rf2,
            bucket_index=bucket_index,

            theta_local=trajectory.family2.theta_local,
            theta_absolute=theta_absolute,
            P=trajectory.family2.P_rf1,

            delta_t=delta_t_absolute,
            delta_E=delta_E_rf1,

            delta_p_over_p0=delta_p_over_p0_from_delta_E(
                delta_E_rf1,
                rf1_machine,
            ),

            z=-rf1_machine.beta_s *
              SPEED_OF_LIGHT_M_PER_S .*
              delta_t_absolute,
        )
    end
end
