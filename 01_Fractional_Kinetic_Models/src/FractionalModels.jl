module FractionalModels

"""
    estimate_params(problem, param_setup; optim_setup = nothing)
    estimate_params(sys!, alpha, x0, u, data, obj!, param_setup, T, N; optim_setup = nothing)

Estimate model parameters with the Ipopt backend.
"""
function estimate_params end

include("PECE.jl")
include("ParameterEstimation.jl")
include("IpoptBackend.jl")

export solve, solve!
export SeriesMatrix, ScenarioSolutions
export simulate_scenario, simulate_scenarios
export evaluate_objective, estimate_params
export EstimationProblem, prepare_estimation_problem

end
