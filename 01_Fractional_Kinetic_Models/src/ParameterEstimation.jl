struct SeriesMatrix{T} <: AbstractMatrix{Vector{T}}
    data::Matrix{Vector{T}}
end

struct ScenarioSolutions{T} <: AbstractVector{Matrix{T}}
    data::Vector{Matrix{T}}
end

struct EstimationProblem{SYS,ALPHA,OBJ,SC,SYSS,TG,IDX,EXP,TT}
    sys!::SYS
    alpha::ALPHA
    obj!::OBJ
    scenarios::SC
    systems::SYSS
    t_grid::TG
    sample_indices::IDX
    x_exp::EXP
    T::TT
    N::Int
end

Base.length(solutions::ScenarioSolutions) = length(solutions.data)
Base.size(solutions::ScenarioSolutions) = (length(solutions.data),)
Base.size(solutions::ScenarioSolutions, dim::Int) = size(solutions.data, dim)
Base.axes(solutions::ScenarioSolutions) = axes(solutions.data)
Base.axes(solutions::ScenarioSolutions, dim::Int) = axes(solutions.data, dim)
Base.firstindex(solutions::ScenarioSolutions) = firstindex(solutions.data)
Base.lastindex(solutions::ScenarioSolutions) = lastindex(solutions.data)
Base.iterate(solutions::ScenarioSolutions, state...) = iterate(solutions.data, state...)
Base.only(solutions::ScenarioSolutions) = only(solutions.data)
Base.IndexStyle(::Type{<:ScenarioSolutions}) = IndexLinear()
Base.eltype(::Type{ScenarioSolutions{T}}) where {T} = Matrix{T}
Base.eltype(::ScenarioSolutions{T}) where {T} = Matrix{T}

function Base.getindex(solutions::ScenarioSolutions, exp::Int)
    return solutions.data[exp]
end

function Base.getindex(solutions::ScenarioSolutions, i::Int, J...)
    if length(solutions) != 1
        error("Use sol[exp][var, idx] when there is more than one experiment.")
    end

    return only(solutions)[i, J...]
end

Base.length(series::SeriesMatrix) = length(series.data)
Base.size(series::SeriesMatrix) = size(series.data)
Base.size(series::SeriesMatrix, dim::Int) = size(series.data, dim)
Base.axes(series::SeriesMatrix) = axes(series.data)
Base.axes(series::SeriesMatrix, dim::Int) = axes(series.data, dim)
Base.firstindex(series::SeriesMatrix) = firstindex(series.data)
Base.firstindex(series::SeriesMatrix, dim::Int) = firstindex(series.data, dim)
Base.lastindex(series::SeriesMatrix) = lastindex(series.data)
Base.lastindex(series::SeriesMatrix, dim::Int) = lastindex(series.data, dim)
Base.iterate(series::SeriesMatrix, state...) = iterate(series.data, state...)
Base.IndexStyle(::Type{<:SeriesMatrix}) = IndexCartesian()
Base.eltype(::Type{SeriesMatrix{T}}) where {T} = Vector{T}
Base.eltype(::SeriesMatrix{T}) where {T} = Vector{T}

function Base.getindex(series::SeriesMatrix, i::Int, j::Int)
    return series.data[i, j]
end

function Base.getindex(series::SeriesMatrix, i::Int)
    if size(series, 2) != 1
        error("Use x[var, exp] when there is more than one experiment.")
    end

    return series.data[i, 1]
end

_to_vector(x::Vector) = copy(x)
_to_vector(x::AbstractVector) = collect(x)
_to_vector(x::AbstractMatrix) = vec(copy(x))
_to_vector(x) = vec(collect(x))

function build_scenario_system(sys!, scenario)
    scenario_inputs = if hasproperty(scenario, :u)
        scenario.u isa NamedTuple || throw(ArgumentError("u must be a NamedTuple or nothing"))
        hasproperty(scenario.u, :x0) && throw(ArgumentError("u cannot contain an x0 entry"))
        merge((x0=scenario.x0,), scenario.u)
    else
        (x0=scenario.x0,)
    end

    return (dx, x, t, p) -> sys!(dx, x, t, p; u=scenario_inputs)
end

function build_alpha(alpha, p, scenario)
    αraw = alpha isa Function ? alpha(p) : alpha
    αv = _to_vector(αraw)
    n_state = length(scenario.x0)

    if length(αv) != n_state
        @warn "Invalid alpha length. The simulation will stop." expected_length=n_state received_length=length(αv)
        throw(ArgumentError("alpha must have the same length as x0"))
    end

    if !all(isfinite, αv)
        @warn "Non-finite alpha values detected. The simulation will stop." alpha=αv
        throw(ArgumentError("alpha values must be finite"))
    end

    if any((αv .<= 0) .| (αv .> 1))
        @warn "Invalid alpha values. The simulation will stop." alpha=αv
        throw(ArgumentError("alpha values must satisfy 0 < alpha <= 1"))
    end

    return αv
end

function normalize_x_exp(x_exp)
    return x_exp isa AbstractVector ? reshape(_to_vector(x_exp), 1, :) : Matrix(x_exp)
end

function normalize_x0_list(x0)
    if x0 isa AbstractMatrix
        if size(x0, 1) == 1 || size(x0, 2) == 1
            return [_to_vector(x0)]
        end

        return [_to_vector(view(x0, j, :)) for j in axes(x0, 1)]
    end

    if x0 isa AbstractVector
        isempty(x0) && throw(ArgumentError("x0 cannot be empty"))

        if first(x0) isa AbstractVector || first(x0) isa AbstractMatrix
            return [_to_vector(x0j) for x0j in x0]
        end

        return [_to_vector(x0)]
    end

    throw(ArgumentError("x0 must be a vector, matrix, or vector of vectors"))
end

function normalize_u_list(u, n_exp)
    if u === nothing
        return fill(nothing, n_exp)
    end

    if u isa NamedTuple
        return fill(u, n_exp)
    end

    if u isa AbstractVector
        if n_exp == 1 && (isempty(u) || !(first(u) isa NamedTuple))
            return [u]
        end

        if all(ui -> ui isa NamedTuple || ui === nothing, u)
            length(u) == n_exp || throw(ArgumentError("u and x0 must contain the same number of experiments"))
            return collect(u)
        end
    end

    n_exp == 1 && return [u]
    throw(ArgumentError("u must be a NamedTuple, a vector of NamedTuple, or nothing"))
end

function normalize_data_list(data)
    if data isa AbstractVector
        isempty(data) && throw(ArgumentError("data cannot be empty"))
        return collect(data)
    end

    return [data]
end

function build_model_scenario(x0, u)
    x0_vec = _to_vector(x0)
    isempty(x0_vec) && throw(ArgumentError("x0 cannot be empty"))
    all(isfinite, x0_vec) || throw(ArgumentError("x0 must contain only finite values"))

    scenario = (x0=x0_vec,)

    if u === nothing
        return scenario
    end

    u isa NamedTuple || throw(ArgumentError("u must be a NamedTuple or nothing"))
    return merge(scenario, (u=u,))
end

function build_data_scenario(x0, u, data, exp_index::Int)
    hasproperty(data, :t_exp) || throw(ArgumentError("data[$exp_index] must contain t_exp"))
    hasproperty(data, :x_exp) || throw(ArgumentError("data[$exp_index] must contain x_exp"))

    t_exp = _to_vector(data.t_exp)
    x_exp = normalize_x_exp(data.x_exp)

    isempty(t_exp) && throw(ArgumentError("t_exp cannot be empty for experiment $exp_index"))
    isempty(x_exp) && throw(ArgumentError("x_exp cannot be empty for experiment $exp_index"))
    size(x_exp, 2) == length(t_exp) || throw(ArgumentError("x_exp columns must match length(t_exp) for experiment $exp_index"))
    all(isfinite, t_exp) || throw(ArgumentError("t_exp must contain only finite values for experiment $exp_index"))
    all(isfinite, x_exp) || throw(ArgumentError("x_exp must contain only finite values for experiment $exp_index"))
    size(x_exp, 1) <= length(x0) || throw(ArgumentError("x_exp has more observed variables than available states for experiment $exp_index"))

    scenario = merge(
        build_model_scenario(x0, u),
        (t_exp=t_exp, x_exp=x_exp),
    )

    if hasproperty(data, :x_map)
        x_map = data.x_map isa AbstractVector ? collect(data.x_map) : [data.x_map]
        isempty(x_map) && throw(ArgumentError("x_map cannot be empty for experiment $exp_index"))
        all(idx -> idx isa Integer, x_map) || throw(ArgumentError("x_map must contain integer indices for experiment $exp_index"))
        all(idx -> 1 <= idx <= length(x0), x_map) || throw(ArgumentError("x_map contains out-of-range indices for experiment $exp_index"))
        length(x_map) == size(x_exp, 1) || throw(ArgumentError("length(x_map) must match the number of observed variables for experiment $exp_index"))
        scenario = merge(scenario, (x_map=x_map,))
    end

    return scenario
end

function build_model_scenarios(x0, u)
    x0_list = normalize_x0_list(x0)
    u_list = normalize_u_list(u, length(x0_list))
    return [build_model_scenario(x0_list[j], u_list[j]) for j in eachindex(x0_list)]
end

function build_scenarios(x0, u, data)
    x0_list = normalize_x0_list(x0)
    data_list = normalize_data_list(data)
    length(x0_list) == length(data_list) || throw(ArgumentError("x0 and data must contain the same number of experiments"))

    u_list = normalize_u_list(u, length(data_list))
    return [build_data_scenario(x0_list[j], u_list[j], data_list[j], j) for j in eachindex(data_list)]
end

function solve_scenario(sys!, alpha, scenario, p, T, N)
    scenario_sys! = build_scenario_system(sys!, scenario)
    α = build_alpha(alpha, p, scenario)
    return solve!(scenario_sys!, α, scenario.x0, p, T, N)
end

function solve_prepared_scenario(scenario_sys!, alpha, scenario, p, T, N)
    α = build_alpha(alpha, p, scenario)
    return solve!(scenario_sys!, α, scenario.x0, p, T, N)
end

function build_sample_indices(t_grid, t_exp)
    return map(ti -> argmin(abs.(t_grid .- ti)), t_exp)
end

function sample_states(x_model, sample_indices, scenario)
    if hasproperty(scenario, :x_map)
        return x_model[scenario.x_map, sample_indices]
    end

    n_obs = size(scenario.x_exp, 1)
    return x_model[1:n_obs, sample_indices]
end

function build_series_matrix(series_list)
    n_exp = length(series_list)
    n_var = size(first(series_list), 1)
    Tval = eltype(first(series_list))
    data = Matrix{Vector{Tval}}(undef, n_var, n_exp)

    for (j, series) in enumerate(series_list)
        size(series, 1) == n_var || throw(ArgumentError("All experiments must have the same number of observed variables."))

        for i in 1:n_var
            data[i, j] = _to_vector(view(series, i, :))
        end
    end

    return SeriesMatrix(data)
end

function _threaded_collect(f, n_items::Int)
    n_items >= 1 || throw(ArgumentError("At least one experiment is required."))

    first_value = f(1)
    values = Vector{typeof(first_value)}(undef, n_items)
    values[1] = first_value

    if n_items == 1
        return values
    end

    if Threads.nthreads() == 1
        for j in 2:n_items
            values[j] = f(j)
        end
    else
        Threads.@threads for j in 2:n_items
            values[j] = f(j)
        end
    end

    return values
end

function prepare_estimation_problem(sys!, alpha, x0, u, data, obj!, T, N)
    scenarios = build_scenarios(x0, u, data)
    systems = [build_scenario_system(sys!, scenario) for scenario in scenarios]
    t_grid = range(zero(T), T; length=N + 1)
    sample_indices = [build_sample_indices(t_grid, scenario.t_exp) for scenario in scenarios]
    x_exp = build_series_matrix([scenario.x_exp for scenario in scenarios])

    return EstimationProblem(sys!, alpha, obj!, scenarios, systems, t_grid, sample_indices, x_exp, T, N)
end

function simulate_scenario(sys!, alpha, x0, u, p, T, N)
    scenario = build_model_scenario(x0, u)
    return solve_scenario(sys!, alpha, scenario, p, T, N)
end

function simulate_scenarios(sys!, alpha, x0, u, p, T, N)
    scenarios = build_model_scenarios(x0, u)
    systems = [build_scenario_system(sys!, scenario) for scenario in scenarios]
    solutions = _threaded_collect(j -> solve_prepared_scenario(systems[j], alpha, scenarios[j], p, T, N), length(scenarios))
    return ScenarioSolutions(solutions)
end

function evaluate_objective(problem::EstimationProblem, p)
    x_samples = _threaded_collect(j -> begin
        x_model = solve_prepared_scenario(problem.systems[j], problem.alpha, problem.scenarios[j], p, problem.T, problem.N)
        sample_states(x_model, problem.sample_indices[j], problem.scenarios[j])
    end, length(problem.scenarios))

    x = build_series_matrix(x_samples)
    return problem.obj!(x, problem.x_exp)
end

function evaluate_objective(sys!, alpha, x0, u, data, obj!, p, T, N)
    problem = prepare_estimation_problem(sys!, alpha, x0, u, data, obj!, T, N)
    return evaluate_objective(problem, p)
end

function _validate_param_setup(param_setup)
    hasproperty(param_setup, :p0) || throw(ArgumentError("param_setup must contain p0"))
    hasproperty(param_setup, :p_lb) || throw(ArgumentError("param_setup must contain p_lb"))
    hasproperty(param_setup, :p_up) || throw(ArgumentError("param_setup must contain p_up"))

    p0 = _to_vector(param_setup.p0)
    p_lb = _to_vector(param_setup.p_lb)
    p_up = _to_vector(param_setup.p_up)

    length(p0) == length(p_lb) == length(p_up) || throw(ArgumentError("p0, p_lb, and p_up must have the same length"))
    all(isfinite, p0) || throw(ArgumentError("p0 must contain only finite values"))
    all(isfinite, p_lb) || throw(ArgumentError("p_lb must contain only finite values"))
    all(isfinite, p_up) || throw(ArgumentError("p_up must contain only finite values"))
    all(p_lb .<= p_up) || throw(ArgumentError("p_lb must be less than or equal to p_up component-wise"))
    all((p0 .>= p_lb) .& (p0 .<= p_up)) || throw(ArgumentError("p0 must lie within [p_lb, p_up] component-wise"))

    return p0, p_lb, p_up
end

function _default_method()
    return Fminbox(LBFGS())
end

function _default_options()
    return Optim.Options(show_trace=true)
end

function _get_method(optim_setup)
    if isnothing(optim_setup)
        return _default_method()
    end

    return hasproperty(optim_setup, :method) ? optim_setup.method : _default_method()
end

function _get_options(optim_setup)
    if isnothing(optim_setup)
        return _default_options()
    end

    return hasproperty(optim_setup, :options) ? optim_setup.options : _default_options()
end

function estimate_params(problem::EstimationProblem, param_setup; optim_setup=nothing)
    p0, p_lb, p_up = _validate_param_setup(param_setup)

    method = _get_method(optim_setup)
    options = _get_options(optim_setup)

    objective = p -> evaluate_objective(problem, p)
    result = optimize(objective, p_lb, p_up, p0, method, options)
    return Optim.minimizer(result)
end

function estimate_params(sys!, alpha, x0, u, data, obj!, param_setup, T, N; optim_setup=nothing)
    problem = prepare_estimation_problem(sys!, alpha, x0, u, data, obj!, T, N)
    return estimate_params(problem, param_setup; optim_setup=optim_setup)
end
