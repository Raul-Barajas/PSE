using JuMP
import Ipopt
using Logging

_to_vector_backend(x::Vector) = copy(x)
_to_vector_backend(x::AbstractVector) = collect(x)
_to_vector_backend(x::AbstractMatrix) = vec(copy(x))
_to_vector_backend(x) = vec(collect(x))

function _validate_param_setup(param_setup)
    hasproperty(param_setup, :p0) || throw(ArgumentError("param_setup must contain p0"))
    hasproperty(param_setup, :p_lb) || throw(ArgumentError("param_setup must contain p_lb"))
    hasproperty(param_setup, :p_up) || throw(ArgumentError("param_setup must contain p_up"))

    p0 = _to_vector_backend(param_setup.p0)
    p_lb = _to_vector_backend(param_setup.p_lb)
    p_up = _to_vector_backend(param_setup.p_up)

    length(p0) == length(p_lb) == length(p_up) || throw(ArgumentError("p0, p_lb, and p_up must have the same length"))
    all(isfinite, p0) || throw(ArgumentError("p0 must contain only finite values"))
    all(isfinite, p_lb) || throw(ArgumentError("p_lb must contain only finite values"))
    all(isfinite, p_up) || throw(ArgumentError("p_up must contain only finite values"))
    all(p_lb .<= p_up) || throw(ArgumentError("p_lb must be less than or equal to p_up component-wise"))
    all((p0 .>= p_lb) .& (p0 .<= p_up)) || throw(ArgumentError("p0 must lie within [p_lb, p_up] component-wise"))

    return p0, p_lb, p_up
end

function _strictify_bounds(p0, p_lb, p_up, lower_bound_epsilon)
    p0_adj = copy(p0)
    p_lb_adj = copy(p_lb)

    for i in eachindex(p_lb_adj)
        if p_lb_adj[i] == 0
            p_lb_adj[i] = lower_bound_epsilon
        end
    end

    p0_adj .= clamp.(p0_adj, p_lb_adj, p_up)
    return p0_adj, p_lb_adj, p_up
end

function _default_optim_options()
    return (
        print_level = 5,
        max_iter = 100,
        tol = 1e-6,
        acceptable_tol = 1e-5,
        hessian_approximation = "limited-memory",
        lower_bound_epsilon = 1e-8,
        warn_on_unsolved_status = true,
    )
end

function _merge_optim_options(optim_setup)
    if isnothing(optim_setup)
        return _default_optim_options()
    end

    optim_setup isa NamedTuple || throw(ArgumentError("optim_setup must be a NamedTuple or nothing"))
    return merge(_default_optim_options(), optim_setup)
end

function _build_optimizer_model(optimizer::T) where {T}
    return JuMP.Model(optimizer)
end

function _box_distance_squared(p, p_lb, p_up)
    lower_violation = max.(zero(eltype(p)), p_lb .- p)
    upper_violation = max.(zero(eltype(p)), p .- p_up)
    return sum(abs2, lower_violation) + sum(abs2, upper_violation)
end

function _safe_objective(problem, p, p_lb, p_up; penalty = 1e8)
    if !all(isfinite, p)
        return penalty
    end

    box_distance = _box_distance_squared(p, p_lb, p_up)
    if box_distance > zero(eltype(p))
        return penalty * (one(eltype(p)) + box_distance)
    end

    try
        return Logging.with_logger(Logging.NullLogger()) do
            evaluate_objective(problem, p)
        end
    catch
        return penalty
    end
end

function _objective_gradient_fd(problem, p_lb, p_up; rel_step = 1e-6)
    last_args = Ref{Any}(nothing)
    last_value = Ref{Float64}(NaN)

    function objective_from_tuple(args...)
        if !isnothing(last_args[]) && last_args[] == args
            return last_value[]
        end

        value = _safe_objective(problem, collect(args), p_lb, p_up)
        last_args[] = args
        last_value[] = value
        return value
    end

    function gradient!(g::AbstractVector{T}, x::Vararg{T, N}) where {T <: Real, N}
        xvec = collect(x)
        fx0 = objective_from_tuple(xvec...)

        for i in 1:N
            xi = xvec[i]
            scale = max(abs(xi), one(T))
            h = rel_step * scale

            if xi + h <= p_up[i]
                xp = copy(xvec)
                xp[i] += h
                fxp = objective_from_tuple(xp...)
                g[i] = (fxp - fx0) / h
            else
                xm = copy(xvec)
                xm[i] -= h
                fxm = objective_from_tuple(xm...)
                g[i] = (fx0 - fxm) / h
            end
        end

        return
    end

    return objective_from_tuple, gradient!
end

function _estimate_params_impl(problem::EstimationProblem, param_setup; optim_setup = nothing)
    p0, p_lb, p_up = _validate_param_setup(param_setup)
    optim_options = _merge_optim_options(optim_setup)
    lower_bound_epsilon = optim_options.lower_bound_epsilon
    p0, p_lb, p_up = _strictify_bounds(p0, p_lb, p_up, lower_bound_epsilon)

    objective_from_tuple, gradient! = _objective_gradient_fd(problem, p_lb, p_up)

    model = _build_optimizer_model(Ipopt.Optimizer)
    for (name, value) in pairs(optim_options)
        name in (:lower_bound_epsilon, :warn_on_unsolved_status) && continue
        JuMP.set_attribute(model, String(name), value)
    end

    n_param = length(p0)
    JuMP.@variable(model, p_lb[i] <= p[i = 1:n_param] <= p_up[i], start = p0[i])
    JuMP.@operator(model, op_objective, n_param, objective_from_tuple, gradient!)
    JuMP.@objective(model, Min, op_objective(p...))

    JuMP.optimize!(model)

    term = JuMP.termination_status(model)
    if optim_options.warn_on_unsolved_status &&
       term ∉ (JuMP.MOI.LOCALLY_SOLVED, JuMP.MOI.OPTIMAL, JuMP.MOI.ALMOST_LOCALLY_SOLVED, JuMP.MOI.ALMOST_OPTIMAL)
        @warn "Ipopt did not report a solved status." termination_status = term primal_status = JuMP.primal_status(model)
    end

    return JuMP.value.(p)
end

"""
    estimate_params(problem, param_setup; optim_setup = nothing)

Estimate parameters for a prepared `EstimationProblem` using Ipopt.

`param_setup` must contain `p0`, `p_lb`, and `p_up`. `optim_setup` is an
optional `NamedTuple` of Ipopt options plus package options such as
`warn_on_unsolved_status`.
"""
function estimate_params(problem::EstimationProblem, param_setup; optim_setup = nothing)
    return _estimate_params_impl(problem, param_setup; optim_setup = optim_setup)
end

"""
    estimate_params(sys!, alpha, x0, u, data, obj!, param_setup, T, N; optim_setup = nothing)

Prepare an estimation problem and estimate parameters with Ipopt.
"""
function estimate_params(sys!, alpha, x0, u, data, obj!, param_setup, T, N; optim_setup = nothing)
    problem = prepare_estimation_problem(sys!, alpha, x0, u, data, obj!, T, N)
    return _estimate_params_impl(problem, param_setup; optim_setup = optim_setup)
end
