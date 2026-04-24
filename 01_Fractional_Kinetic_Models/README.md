# FractionalModels

`FractionalModels` is a Julia package for simulating fractional kinetic models
with a PECE integrator and estimating model parameters with an Ipopt/JuMP
backend.

The repository includes two RXN CLC syngas case studies:

- `data/rxn_clc_syngas_133.xlsx`
- `data/rxn_clc_syngas_250.xlsx`

## Project Structure

```text
.
├── Project.toml
├── Manifest.toml
├── setup.jl
├── data/
│   ├── rxn_clc_syngas_133.xlsx
│   └── rxn_clc_syngas_250.xlsx
├── src/
│   ├── FractionalModels.jl
│   ├── PECE.jl
│   ├── ParameterEstimation.jl
│   └── IpoptBackend.jl
├── script/
│   ├── example_133.jl
│   └── example_250.jl
└── test/
    └── test.jl
```

## Installation And Setup

From the project root, run:

```bash
julia setup.jl
```

This activates the project, installs recorded dependencies, precompiles the
package, and checks that `FractionalModels` loads correctly.

Useful setup commands:

```bash
julia setup.jl --help
julia setup.jl --test
julia setup.jl --example
julia setup.jl --dev
```

`--dev` also adds this local package to the default Julia environment with
`Pkg.develop(path=".")`. For normal project work, using `--project=.` is the
cleaner option.

Manual setup is also possible:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()
using FractionalModels
```

## How To Run

Run the integrator test/demo:

```bash
julia --project=. test/test.jl
```

This generates:

```text
fig/test.png
```

Run the parameter-estimation example:

```bash
julia --project=. script/example_133.jl
```

This reads:

```text
data/rxn_clc_syngas_133.xlsx
```

and generates:

```text
fig/rxn_clc_syngas_133.png
```

Run the RXN CLC syngas 250 case study:

```bash
julia --project=. script/example_250.jl
```

This reads:

```text
data/rxn_clc_syngas_250.xlsx
```

and generates:

```text
fig/rxn_clc_syngas_250.png
fig/rxn_clc_syngas_250_yield.png
```

## Public API

Import the package with:

```julia
using FractionalModels
```

### `solve`

```julia
solve(sys!, alpha, x0, p, T, N)
```

Integrates a fractional system with the PECE method.

- `sys!`: dynamic system with signature `sys!(dx, x, t, p)`.
- `alpha`: vector of fractional orders.
- `x0`: initial condition.
- `p`: parameter vector.
- `T`: final simulation time.
- `N`: number of integration steps.

The output is a matrix where rows are states and columns are time points.

### `solve!`

```julia
solve!(sys!, alpha, x0, p, T, N)
```

Compatibility alias for `solve`. New code should prefer `solve`.

### `prepare_estimation_problem`

```julia
prepare_estimation_problem(sys!, alpha, x0, u, data, obj!, T, N)
```

Builds a reusable `EstimationProblem`.

- `alpha` can be a vector or a function `alpha(p)`.
- `x0` contains the initial conditions for one or more experiments.
- `u` contains scenario inputs as a `NamedTuple`, a vector of `NamedTuple`, or
  `nothing`.
- `data` contains experimental data with `t_exp`, `x_exp`, and optionally
  `x_map`.
- `obj!` is the objective function.

### `estimate_params`

```julia
estimate_params(problem, param_setup; optim_setup = nothing)
estimate_params(sys!, alpha, x0, u, data, obj!, param_setup, T, N; optim_setup = nothing)
```

Estimates parameters using Ipopt.

`param_setup` must contain:

```julia
param_setup = (
    p0 = ...,
    p_lb = ...,
    p_up = ...,
)
```

`optim_setup` is optional and can contain Ipopt options plus package options:

```julia
optim_setup = (
    print_level = 5,
    max_iter = 1000,
    tol = 1e-3,
    acceptable_tol = 1e-3,
    warn_on_unsolved_status = false,
)
```

### `evaluate_objective`

```julia
evaluate_objective(problem, p)
evaluate_objective(sys!, alpha, x0, u, data, obj!, p, T, N)
```

Evaluates the objective function for a parameter vector `p`.

### `simulate_scenario`

```julia
simulate_scenario(sys!, alpha, x0, u, p, T, N)
```

Simulates one scenario.

### `simulate_scenarios`

```julia
simulate_scenarios(sys!, alpha, x0, u, p, T, N)
```

Simulates all scenarios and returns a `ScenarioSolutions` container.

### `SeriesMatrix`

Container used inside objective functions.

Use:

```julia
x[var, exp]
x_exp[var, exp]
```

Each entry is a time series already aligned with the experimental times. If
there is only one experiment, `x[var]` and `x_exp[var]` are also accepted.

### `ScenarioSolutions`

Container for simulated trajectories. Index by experiment:

```julia
solution[exp][state, time_index]
```

If there is only one experiment, direct indexing like
`solution[state, time_index]` is also accepted.

### `EstimationProblem`

Prepared object returned by `prepare_estimation_problem`. It stores scenarios,
sampled experimental data, solver settings, and the objective function so the
same problem can be evaluated or optimized repeatedly.

## Minimal Example

```julia
using FractionalModels

function sys!(dx, x, t, p)
    dx[1] = -p[1] * x[1]
    dx[2] = p[1] * x[1] - p[2] * x[2]
end

alpha = [0.9, 1.0]
x0 = [1.0, 0.0]
p = [0.5, 0.2]
T = 10
N = 100

x = solve(sys!, alpha, x0, p, T, N)
```

## Example Objective

Objective functions receive aligned simulated and experimental series:

```julia
function obj!(x, x_exp)
    total = 0.0
    count = 0

    for exp in axes(x, 2)
        for var in axes(x, 1)
            err = x[var, exp] .- x_exp[var, exp]
            total += mean(abs.(err))
            count += 1
        end
    end

    return total / count
end
```

For one experiment, this is also valid:

```julia
function obj!(x, x_exp)
    return mean(abs.(x[1] .- x_exp[1]))
end
```

## Notes

- `script/example_133.jl` is the main parameter-estimation example for syngas 133.
- `script/example_250.jl` is the syngas 250 case-study script.
- `test/test.jl` is currently an integrator demo/test that also saves a figure.
- `fig/` stores generated figures.
- Ipopt can stop at the configured iteration limit and still return a feasible
  parameter vector. Control this behavior through `optim_setup`.
