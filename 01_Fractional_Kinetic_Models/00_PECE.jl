using SpecialFunctions

"""
INPUT VARIABLES:
* `sys!`: The real function of two real variables that defines the right-hand side of the differential equation.
* `α`: The order of the differential equation (a positive real number).
* `y0`: An array of real numbers containing the initial values y(0), y'(0), ..., y^(α-1)(0).
* `p`: The value of the parameters of the function.
* `T`: The upper limit of the interval where the solution will be approximated (a positive real number).
* `N`: The number of time steps the algorithm must take (a positive integer).

OUTPUT VARIABLES:
* `y`: An array of N + 1 real numbers containing the approximate solutions y(T/N * j), j = 0, 1, ..., N.

INTERNAL VARIABLES:
* `h`: The step size of the algorithm (a positive real number).
* `m`: The number of specified initial conditions (a positive integer).
* `j`, `k`: Integer variables used as indices.
* `a`, `b`: Arrays of N + 1 real numbers containing the weights of the corrector and predictor formulas, respectively.
* `P`: The predicted value (a real variable).
"""

function solve!(sys!, α, y0, p, T, N)
    # Estructura de la función
    function f(t, u, p)
        du = zeros(length(α[:]))
        sys!(du, u, p, t)
        return du
    end
    # h: Tamaño del paso
    h = T / N
    # m: Número de condiciones iniciales especificadas
    m = Int.(ones(size(y0, 2)) .* size(y0, 1))
    # ξ: Desface de tiempo
    ξ = h/25
    # Inicialización de los arrays `a` y `b` para los pesos
    a = zeros(length(α[:]), N + 1)
    b = zeros(length(α[:]), N + 1)
    # Inicialización del array de soluciones `y`
    y = zeros(length(α[:]), N + 1)
    # Establecer el valor inicial
    y[:, 1] = y0[1, :]

    # Calcular los pesos de las fórmulas correctoras y predictoras
    @fastmath @inbounds @simd for k in 1:N
        b[:, k+1] = k .^ α[:] - (k - 1) .^ α[:]
        a[:, k+1] = (k + 1) .^ (α[:] .+ 1) - 2 * k .^ (α[:] .+ 1) + (k - 1) .^ (α[:] .+ 1)
    end
    # Integración numérica
    @fastmath @inbounds @simd for j in 1:N
        P = zeros(length(α[:]))
        Y0 = zeros(length(α[:]))
        Fp = zeros(length(α[:]))
        Fy = zeros(length(α[:]))

        @fastmath @inbounds @simd for i in 1:length(α[:])
            @fastmath @inbounds @simd for k in 0:m[i]-1
                Y0[i] += ((j * h)^k / factorial(k)) .* y0[k+1, i]
            end
        end

        @fastmath @inbounds @simd for k in 0:(j-1)
            Fp[:] += b[:, j-k+1] .* f((k * h) + ξ, y[:, k+1], p)
        end

        @fastmath @inbounds @simd for k in 1:(j-1)
            Fy[:] += a[:, j-k+1] .* f(k * h, y[:, k+1], p)
        end

        # Predictor
        P[:] = Y0[:] .+ h .^ α[:] ./ gamma.(α[:] .+ 1) .* Fp[:]
        # Corrector
        y[:, j+1] = Y0[:] .+ h .^ α[:] ./ gamma.(α[:] .+ 2) .* (f(j * h, P[:], p) .+ ((j - 1) .^ (α[:] .+ 1) .- (j .- 1 .- α[:]) .* (j .^ α[:])) .* f(ξ, y[:, 1], p) .+ Fy[:])
    end
    # Reemplazar NaN por 0
    y[isnan.(y)] .= 0
    # Reemplazar Inf por 0
    y[isinf.(y)] .= 0
    return y
end