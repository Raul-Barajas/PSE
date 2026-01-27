include(joinpath(@__DIR__, "10_OPTIMIZER_FOCP.jl"))

# Rango de tiempo y tanaño de paso
T = 1 #seg
N = 20

# u0: Estimación inicial del vector de control.
u0 = LinRange(-0.4, 0, N + 1)

# lb-ub: Rango de busqueda lb: lower_bounds y ub: upper_bounds.
lb = -10
ub = 10

# x0: Condiciones iniciales del vector de estado.
x0 = [1]

# α: Orden de los operadores diferenciales 0 < α < 1
α = [0.5]

# p: Parametros de estado
p = nothing

# q: Peso para la funcion objetivo de tipo Mayer
q = [1]

# r: Peso para la funcion objetivo de tipo Lagrange
r = [0.5]

# sys: Sistema dinamico.
function sys!(dx, x, u, t, p)
    dx[1] = -x[1] + u[1]
end

# Phi: Funcion objetivo de tipo Mayer
function Phi!(x, t, q)
    q[1] * (x[1] + 0.4)^2
end

# Laplacian: Función objetivo de tipo Lagrange
function Laplacian!(x, u, t, r)
    r[1] * (x[1]^2 + u[1]^2)
end

u = ones(1, N + 1) .* transpose(u0)
x = Solve_Forward!(sys!, α, x0, u, p, T, N)




function obj(landa_tf)
    h = T/N
    ξ = h / 25

    landa_h = Phi_x!(Phi!, x[:, end], T, q) * gamma.(3.0 .- α[:]) ./ (h .^ α[:]) + (α[:] .- 1) .* landa_tf[:]

    function f!(x, u, λ, t, p, r)
        return transpose(F_x!(sys!, x, u, (t + ξ), p)) * λ + transpose(L_x!(Laplacian!, x, u, (t + ξ), r)) - landa_tf ./ (gamma.(1.0 .- α[:]) .* ((T - t + ξ) .^ α[:]))
    end


    error = landa_tf[:] .- landa_h[:] .+ h .^ α[:] ./ gamma.(α[:] .+ 2) .* (f!(x[:, N], u[:, N], landa_h[:], T - (h), p, r) .+ α[:] .* f!(x[:, N+1], u[:, N+1], landa_tf[:], T, p, r))

    return (error[1])^2
end

using Optim
landa_tf_0 = Phi_x!(Phi!, x[:, end], T, q) .* gamma.(2.0 .- α[:]) ./ ((T/N/10000) .^ (1.0 .- α[:])) 

obj(landa_tf_0)


internal_optimization = optimize(
    obj,
    landa_tf_0,
    NelderMead()
)

landa_tf = Optim.minimizer(internal_optimization)
obj(landa_tf)
landa_tf
landa_tf_0