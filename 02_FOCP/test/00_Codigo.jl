include(joinpath(@__DIR__, "..", "src", "10_OPTIMIZER_FOCP.jl"))

# Rango de tiempo y tanaño de paso
T = 10 #seg
H = 0.05
N = Int(T / H)

# u0: Estimación inicial del vector de control.
u0 = [0; 0]

# lb-ub: Rango de busqueda lb: lower_bounds y ub: upper_bounds.
lb = 0
ub = 5

# x0: Condiciones iniciales del vector de estado.
x0 = [0; 0]

# α: Orden de los operadores diferenciales 0 < α < 1
α = [1; 1]

# p: Parametros de estado
p = nothing

# q: Peso para la funcion objetivo de tipo Mayer
q = 1

# r: Peso para la funcion objetivo de tipo Lagrange
r = 1

# sys: Sistema dinamico.
function sys!(dx, x, u, t, p)
    dx[1] = x[2]
    dx[2] = u[1]
end

# Phi: Funcion objetivo de tipo Mayer
function Phi!(x, t, q)
    q[1] * (x[1] - 100)^2
end

# Laplacian: Función objetivo de tipo Lagrange
function Laplacian!(x, u, t, r)
    r[1] * u[1]^2
end




##########################################################

##########################################################

# Parametros generales
h = T / N           # h: Tamaño del paso.
n = size(u0, 1)     # n: Número de variables de control especificadas.
ξ = h / 25          # ξ: Desface de tiempo para evitar singularidades.
tol_u = 1E-4         # tol_u: tolerancia para el vector de control
step = float(1)
Step = step
Step_min = 0.1 * step

# Inicializacion de los arreglos
u = ones(n, N + 1) .* u0                # Inicialización del array de soluciones
grad = zeros(n, N + 1)                  # Crear el arreglo del gradiente
grad_new = zeros(n, N + 1)              # Crear el arreglo del nuevo gradiente
p_k = zeros(n, N + 1)                   # Crear el vector de direccion p_k

# Inicializa listas (historiales)
u_hist = Vector{Array{Float64}}()
x_hist = Vector{Array{Float64}}()
λ_hist = Vector{Array{Float64}}()
grad_hist = Vector{Array{Float64}}()
H_hist = Vector{Array{Float64}}()

# Establecer el valor inicial de la matriz hessiana
In = Matrix{Float64}(I, n, n)  # Identidad de tamaño n
H = In

# Valor inicial de las variables x y u
u = clamp.(u, lb, ub)
x = Solve_Forward!(sys!, α, x0, u, p, T, N)
λ = Solve_Backward!(Phi!, Laplacian!, sys!, α, x, u, p, q, r, T, N)

# Calcular gradiente inicial
@fastmath @inbounds @simd for j in 1:N+1
    grad[:, j] = H_u!(sys!, Laplacian!, x[:, j], u[:, j], λ[:, j], ((j - 1) * h) + ξ, p, r)
end


@fastmath @inbounds @simd for j in 1:N+1
    p_k[:, j] = -H * grad[:, j]
end

p_k
-H * grad[:, 1]
grad[:, 1]



H


iter = 1

# Paso 3: Actualizar control con nuevo step
u_new = u .+ Step .* p_k
u_new = clamp.(u_new, lb, ub)

# Paso 4: Calcular nuevos valores con u_new
x_new = Solve_Forward!(sys!, α, x0, u_new, p, T, N)
λ_new = Solve_Backward!(Phi!, Laplacian!, sys!, α, x_new, u_new, p, q, r, T, N)

# Calcular nuevo gradiente
@fastmath @inbounds @simd for j in 1:N+1
    grad_new[:, j] = H_u!(sys!, Laplacian!, x_new[:, j], u_new[:, j], λ_new[:, j], ((j - 1) * h) + ξ, p, r)
end

# Actualización global promediada
s = integral_promedio(u_new) - integral_promedio(u)
y = integral_promedio(grad_new) - integral_promedio(grad)

yTs = dot(y, s)

s' * y

ρ = 1.0 / yTs

s * y'

V = In .- ρ .* (s * y')

H = V * H * V' + ρ .* (s * s')


# Paso 6: Actualizar variables para la próxima iteración
u .= u_new
x .= x_new
λ .= λ_new
grad .= grad_new
