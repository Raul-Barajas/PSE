"""
===============================================================================
 ÓPTIMIZADOR PARA PROBLEMAS DE CONTROL ÓPTIMO FRACCIONALES
===============================================================================
 Método de optimización:    BFGS (quasi-Newton)
 Condición final para λ:    Forma simplificada (ϕ * Γ(2-α) / h ^ (1-α))

 Autor:                     Luis Raúl Barajas Villarruel
 Fecha de modificación:     21 de octubre de 2025

 Historial de cambios:
 - v1.4 (15-10-2025): Se corrigen errores graves en el integrador hacia atras.

 - v1.3 (20-05-2025): Se elimina el metodo para calcular s-y.
 
 - v1.2 (20-05-2025): Se incluyen las condiciones de Wolfe para mejorar el 
 tamaño de paso.

 - v1.1 (14-05-2025): Metodo para calcular los vectores s_y para una 
 actualización tipo BFGS.
 
 - v1.0 (01-01-2025): Versión inicial del optimizador.
===============================================================================
"""

# Dependencias
using LinearAlgebra, ForwardDiff, Statistics, SpecialFunctions, Printf, LineSearches

struct Solucion
    x::Matrix{Float64}
    u::Matrix{Float64}
    λ::Matrix{Float64}
    Hu::Matrix{Float64}
end

# Vector de estado
function F!(sys!, x, u, t, p)
    dx = similar(x)
    sys!(dx, x, u, t, p)
    return dx
end

function Fu!(sys!, x, u, t, p)
    dx = similar(x, eltype(u))
    sys!(dx, x, u, t, p)
    return dx
end

# Derivada del vector de estado con respecto a x
function F_x!(sys!, x, u, t, p)
    return ForwardDiff.jacobian(x -> F!(sys!, x, u, t, p), x)
end

# Derivada del vector de estado con respecto a u
function F_u!(sys!, x, u, t, p)
    return ForwardDiff.jacobian(u -> Fu!(sys!, x, u, t, p), u)
end

# Derivada de Phi con respecto a x
function Phi_x!(Phi!, x, t, q)
    return ForwardDiff.jacobian(x -> [Phi!(x, t, q)], x)
end

# Derivada de Laplacian con respecto a x
function L_x!(Laplacian!, x, u, t, r)
    return ForwardDiff.jacobian(x -> [Laplacian!(x, u, t, r)], x)
end

# Derivada de Laplacian con respecto a u
function L_u!(Laplacian!, x, u, t, r)
    return ForwardDiff.jacobian(u -> [Laplacian!(x, u, t, r)], u)
end

# Derivada del Hamiltonian con respecto a u
function H_u!(sys!, Laplacian!, x, u, λ, t, p, r)
    return transpose(L_u!(Laplacian!, x, u, t, r)) + transpose(F_u!(sys!, x, u, t, p)) * λ
end

# Derivada del Hamiltonian con respecto a x
function H_x!(sys!, Laplacian!, x, u, λ, t, p, r)
    return transpose(L_x!(Laplacian!, x, u, t, r)) + transpose(F_x!(sys!, x, u, t, p)) * λ
end

# Hamiltonian
function Hamiltonian!(sys!, Laplacian!, x, u, λ, t, p, r)
    return Laplacian!(x, u, t, r) + transpose(F!(sys!, x, u, t, p)) * λ
end

function Integral_Laplacian(Laplacian!, x, u, r, T, N)

    # Parametros
    h = T / N           # h: Tamaño del paso
    ξ = h / 25          # ξ: Desface de tiempo

    suma = Laplacian!(x[:, 1], u[:, 1], ξ, r) + Laplacian!(x[:, end], u[:, end], N * h, r)

    for j in 2:N
        suma += 2 * Laplacian!(x[:, j], u[:, j], (j - 1) * h, r)
    end

    return (h / 2) * suma
end

# Integrador hacia adelante
function Solve_Forward!(sys!, α, x0, u, p, T, N)

    # Parametros
    h = T / N                                                       # h: Tamaño del paso
    ξ = h / 25                                                      # ξ: Desface de tiempo
    n = length(α[:])                                                # n: Elementos del vector de estado
    x = zeros(length(α[:]), N + 1)                                  # Inicialización del array de soluciones
    x[:, 1] = x0[:]                                                 # Establecer el valor inicial

    # Estructura de la función
    f!(x, u, t, p) = F!(sys!, x, u, t + ξ, p)

    # Integración numérica
    @fastmath @inbounds @simd for i in 0:N-1

        a = zeros(n)
        b = zeros(n)
        P = zeros(n)
        Fp = zeros(n)
        Fy = zeros(n)


        @fastmath @inbounds @simd for j in 0:i
            b[:] = (i + 1 - j) .^ α[:] .- (i - j) .^ α[:]

            if j == 0
                a[:] = (i) .^ (α[:] .+ 1) .- (i .- α[:]) .* (i + 1) .^ α[:]
            else
                a[:] = (i - j + 2) .^ (α[:] .+ 1) .+ (i - j) .^ (α[:] .+ 1) .- 2 .* (i - j + 1) .^ (α[:] .+ 1)
            end

            Fp[:] += b[:] .* f!(x[:, j+1], u[:, j+1], (j * h), p)
            Fy[:] += a[:] .* f!(x[:, j+1], u[:, j+1], (j * h), p)
        end

        # Predictor
        P[:] = x0[:] .+ h .^ α[:] ./ gamma.(α[:] .+ 1) .* Fp[:]

        # Corrector
        x[:, i+2] = x0[:] .+ h .^ α[:] ./ gamma.(α[:] .+ 2) .* (f!(P[:], u[:, i+2], (i + 1) * h, p) .+ Fy[:])

    end

    # Reemplazar NaN por 0
    x[isnan.(x)] .= 0
    # Reemplazar Inf por 0
    x[isinf.(x)] .= 0

    return x
end

# Integrador hacia atras
function Solve_Backward!(Phi!, Laplacian!, sys!, α, x, u, p, q, r, T, N)

    # Parametros
    h = T / N                                                       # h: Tamaño del paso
    ξ = h / 25                                                      # ξ: Desface de tiempo
    ϕ_x = transpose(Phi_x!(Phi!, x[:, end], T, q))                  # ϕ : Derivada de Phi con respecto a x
    λ_tf = ϕ_x .* gamma.(2.0 .- α[:]) ./ ((ξ) .^ (1.0 .- α[:]))     # λf: Condicion final
    n = length(α[:])                                                # Elementos del vector de co-estado
    λ = zeros(n, N + 1)                                             # Inicialización del array de soluciones `λ`
    λ[:, N+1] = λ_tf[:]                                             # Establecer el valor inicial

    # Estructura de la función
    f!(x, u, λ, t, p, r) = H_x!(sys!, Laplacian!, x, u, λ, t + ξ, p, r) - λ_tf ./ (gamma.(1.0 .- α[:]) .* ((T - t + ξ) .^ α[:]))

    # Integración numérica
    @fastmath @inbounds @simd for i in 0:N-1

        a = zeros(n)
        b = zeros(n)
        P = zeros(n)
        Fp = zeros(n)
        Fy = zeros(n)


        @fastmath @inbounds @simd for j in 0:i
            b[:] = (i + 1 - j) .^ α[:] .- (i - j) .^ α[:]

            if j == 0
                a[:] = (i) .^ (α[:] .+ 1) .- (i .- α[:]) .* (i + 1) .^ α[:]
            else
                a[:] = (i - j + 2) .^ (α[:] .+ 1) .+ (i - j) .^ (α[:] .+ 1) .- 2 .* (i - j + 1) .^ (α[:] .+ 1)
            end

            Fp[:] += b[:] .* f!(x[:, N+1-j], u[:, N+1-j], λ[:, N+1-j], T - (j * h), p, r)
            Fy[:] += a[:] .* f!(x[:, N+1-j], u[:, N+1-j], λ[:, N+1-j], T - (j * h), p, r)
        end

        # Predictor
        P[:] = λ_tf[:] .+ h .^ α[:] ./ gamma.(α[:] .+ 1) .* Fp[:]

        # Corrector
        λ[:, N-i] = λ_tf[:] .+ h .^ α[:] ./ gamma.(α[:] .+ 2) .* (f!(x[:, N-i], u[:, N-i], P[:], T - ((i + 1) * h), p, r) .+ Fy[:])

    end

    # Reemplazar NaN por 0
    λ[isnan.(λ)] .= 0
    # Reemplazar Inf por 0
    λ[isinf.(λ)] .= 0

    return λ
end

# Funciones para imprimir en terminal
function print_centered_left_1(text::Vector{String}, width::Int)
    horizontal_line = repeat("─", width)  # Línea horizontal de ancho especificado
    println(horizontal_line)
    for line in text
        println(lpad(line, div(width + length(line), 2)))
    end
    println(horizontal_line)
end

function print_centered_left_2(text::Vector{String}, width::Int)
    for line in text
        println(lpad(line, div(width + length(line), 2)))
    end
end

function integral_promedio(var)
    suma = var[:, 1] .+ var[:, end]
    for j in 2:N
        suma .+= 2 .* var[:, j]
    end
    return (1 / (2 * N)) .* suma
end


function print_end(iter, Obj, Norm_Grad, time_total, txt, width)
    horizontal_line = repeat("─", width)  # Línea horizontal de ancho especificado
    println(horizontal_line)
    println("")
    @printf " Numero de Iteraciones ........: %0d\n" iter
    @printf " Tiempo de CPU ................: %.4f s\n" time_total
    @printf " Norma del Gradiente ..........: %.10e\n" Norm_Grad
    @printf " Función Objetivo .............: %.10e\n" Obj

    println("")
    println("\e[31m Salida: \e[0m", txt)
    println("")
    println("")
end

# Optimizador con metodo BFGS
function Optimize!(Phi!, Laplacian!, sys!, α, x0, u0, lb, ub, T, N; p=nothing, q=nothing, r=nothing, tol=1e-8, max_iters=1e4, step=1.0, linesearch=StrongWolfe())

    print_centered_left_1([
            "OPTIMIZADOR DE PROBLEMAS DE CONTROL ÓPTIMO FRACCIONAL",
            "Desarrollado por: Luis R. Barajas-Villarruel, Vicente Rico-Ramirez.",
            "Basado en el método BFGS"
        ], 117)

    # Guarda el tiempo inicial
    start_time = time()
    time_total = 0

    # Parametros generales
    h = T / N           # h: Tamaño del paso.
    n = size(u0, 1)     # n: Número de variables de control especificadas.
    ξ = h / 25          # ξ: Desface de tiempo para evitar singularidades.
    tol_u = tol         # tol_u: tolerancia para el vector de control
    step = float(step)
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

    # Iteración numérica
    for iter in 1:max_iters

        for (hist, val) in zip((u_hist, x_hist, λ_hist, grad_hist, H_hist), (u, x, λ, grad, H))
            if iter == 1
                append!(hist, [copy(val) for _ in 1:3])
            else
                push!(hist, copy(val))
                deleteat!(hist, 1)
            end
        end

        # Paso 1: Calcular dirección BFGS
        @fastmath @inbounds @simd for j in 1:N+1
            p_k[:, j] = -H * grad[:, j]
        end

        # Datos que se imprime en cada iteracion
        Int_Lapla = Integral_Laplacian(Laplacian!, x, u, r, T, N)
        Val_Phi = Phi!(x[:, end], T, q)
        Norm_Grad = norm(grad)
        omega_0 = Int_Lapla + Val_Phi
        mean_H = norm(diag(H))
        time_iter = time() - start_time
        time_total += time_iter
        start_time = time()
        delta_u = norm(Step .* p_k)

        # Volver a imprimir los datos cada 50 iteraciones

        if iter % 50 == 0 || iter == 1
            iter != 1 && println(repeat("-", 117))
            @printf "%6s %8s %12s %10s %13s %12s %8s %12s %13s %13s\n" "Iter" "Obj" "φ" "∫L" "||∇H||" "||∇²H⁻¹||" "Step" "||Δu||" "Iter T(s)" "CPU T(s)"
            println(repeat("-", 117))
        end

        # Imprimir datos
        @printf "%-105s\n" @sprintf("%4d  %14.6e %10.2e %10.2e %12.4e %10.2e %10.2e %12.4e %9.2f %12.2f", iter, omega_0, Val_Phi, Int_Lapla, Norm_Grad, mean_H, Step, delta_u, time_iter, time_total)


        if abs(Norm_Grad) < tol # Verificar convergencia con el gradiente actual
            print_end(iter, omega_0, Norm_Grad, time_total, "Solución Estacionaria (Óptimo local).", 117)
            break
        elseif isnan(Norm_Grad) # Verificar singularidades del gradiente
            print_end(iter, omega_0, Norm_Grad, time_total, "Error.", 117)
            break
        elseif abs(delta_u) < tol_u # Verificar convergencia en el vector de control
            print_end(iter, omega_0, Norm_Grad, time_total, "Solución Estacionaria (||Δu|| < tol).", 117)
            break
        elseif iter == max_iters
            print_end(iter, omega_0, Norm_Grad, time_total, "Límite de Iteraciones Alcanzado.", 117)
        end

        function omega_step(step)
            u_trial = u .+ step .* p_k
            u_trial = clamp.(u_trial, lb, ub)
            x_trial = Solve_Forward!(sys!, α, x0, u_trial, p, T, N)
            omega_step = Integral_Laplacian(Laplacian!, x_trial, u_trial, r, T, N) + Phi!(x_trial[:, end], T, q)
            return omega_step
        end

        function omega_prime_step(step)
            u_trial = u .+ step .* p_k
            u_trial = clamp.(u_trial, lb, ub)
            x_trial = Solve_Forward!(sys!, α, x0, u_trial, p, T, N)
            λ_trial = Solve_Backward!(Phi!, Laplacian!, sys!, α, x_trial, u_trial, p, q, r, T, N)
            grad_trial = similar(u_trial)
            @fastmath @inbounds @simd for j in 1:N+1
                grad_trial[:, j] = H_u!(sys!, Laplacian!, x_trial[:, j], u_trial[:, j], λ_trial[:, j], ((j - 1) * h) + ξ, p, r)
            end
            omega_prime_step = dot(grad_trial, p_k)
            return omega_prime_step
        end

        function dual(step)
            u_trial = u .+ step .* p_k
            u_trial = clamp.(u_trial, lb, ub)
            x_trial = Solve_Forward!(sys!, α, x0, u_trial, p, T, N)
            λ_trial = Solve_Backward!(Phi!, Laplacian!, sys!, α, x_trial, u_trial, p, q, r, T, N)
            grad_trial = similar(u_trial)
            @fastmath @inbounds @simd for j in 1:N+1
                grad_trial[:, j] = H_u!(sys!, Laplacian!, x_trial[:, j], u_trial[:, j], λ_trial[:, j], ((j - 1) * h) + ξ, p, r)
            end
            omega_step = Integral_Laplacian(Laplacian!, x_trial, u_trial, r, T, N) + Phi!(x_trial[:, end], T, q)
            omega_prime_step = dot(grad_trial, p_k)
            return (omega_step, omega_prime_step)
        end

        omega_prime_0 = dot(grad, p_k)

        Step_line, omega_line = linesearch(omega_step, omega_prime_step, dual, step, omega_0, omega_prime_0)

        if tol_u > Step_line && step > Step_min || omega_line > omega_0

            memo = 2
            step *= 0.5
            linesearch = BackTracking(maxstep=step)

            u = copy(u_hist[end-memo])
            x = copy(x_hist[end-memo])
            λ = copy(λ_hist[end-memo])
            grad = copy(grad_hist[end-memo])
            H = copy(H_hist[end-memo])

            continue
        else
            Step = Step_line
        end

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

        if yTs > 1e-6
            ρ = 1.0 / yTs
            V = In .- ρ .* (s * y')
            H = V * H * V' + ρ .* (s * s')
        else
            H = In
        end

        # Paso 6: Actualizar variables para la próxima iteración
        u .= u_new
        x .= x_new
        λ .= λ_new
        grad .= grad_new

    end

    global FOCP = Solucion(x, u, λ, grad)
    return nothing

end