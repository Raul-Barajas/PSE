#!/usr/bin/env julia

using Pkg

const PROJECT_ROOT = @__DIR__

function print_usage()
    println("""
FractionalModels setup

Usage:
  julia setup.jl
  julia setup.jl --test
  julia setup.jl --example
  julia setup.jl --example-250
  julia setup.jl --dev

Options:
  --test      Prepare the environment and run test/test.jl.
  --example   Prepare the environment and run script/example_133.jl.
  --example-250
              Prepare the environment and run script/example_250.jl.
  --dev       Also add this package to the default Julia environment with Pkg.develop.
  -h, --help  Show this message.

After setup, you can run:
  julia --project=. test/test.jl
  julia --project=. script/example_133.jl
  julia --project=. script/example_250.jl
""")
end

function load_package()
    @eval using FractionalModels
    println("OK: FractionalModels loaded.")
end

function main(args)
    if "-h" in args || "--help" in args
        print_usage()
        return
    end

    println("Preparing FractionalModels")
    println("Project root: ", PROJECT_ROOT)

    Pkg.activate(PROJECT_ROOT)
    Pkg.instantiate()
    Pkg.precompile()
    load_package()

    if "--dev" in args
        Pkg.activate()
        Pkg.develop(path = PROJECT_ROOT)
        println("OK: FractionalModels added to the default Julia environment with Pkg.develop.")
        Pkg.activate(PROJECT_ROOT)
    end

    if "--test" in args
        println("Running test/test.jl")
        include(joinpath(PROJECT_ROOT, "test", "test.jl"))
    end

    if "--example" in args
        println("Running script/example_133.jl")
        include(joinpath(PROJECT_ROOT, "script", "example_133.jl"))
    end

    if "--example-250" in args
        println("Running script/example_250.jl")
        include(joinpath(PROJECT_ROOT, "script", "example_250.jl"))
    end

    println()
    println("Ready. Recommended next commands:")
    println("  julia --project=. test/test.jl")
    println("  julia --project=. script/example_133.jl")
    println("  julia --project=. script/example_250.jl")
end

main(ARGS)
