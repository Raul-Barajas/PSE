module FractionalModels

using Optim
using SpecialFunctions

include("PECE.jl")
include("Old_PECE.jl")
include("ParameterEstimation.jl")

export solve!, old_solve!
export SeriesMatrix, ScenarioSolutions
export simulate_scenario, simulate_scenarios
export evaluate_objective, estimate_params
export EstimationProblem, prepare_estimation_problem

end
