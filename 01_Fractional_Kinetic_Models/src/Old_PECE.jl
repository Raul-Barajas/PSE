using SpecialFunctions

"""
INPUT VARIABLES:
* `f`: La función real de dos variables reales que define el lado derecho de la ecuación diferencial.
* `α`: El orden de la ecuación diferencial (un número real positivo).
* `y0`: Un arreglo de números reales que contiene los valores iniciales y(0), y'(0), ..., y(α-1)(0).
* `p`: El valor de los parametros de la función.
* `T`: El límite superior del intervalo donde se va a aproximar la solución (un número real positivo).
* `N`: El número de pasos de tiempo que el algoritmo debe tomar (un número entero positivo).

OUTPUT VARIABLES:
* `y`: Un arreglo de N + 1 números reales que contiene las soluciones aproximadas y(T/N * j), j = 0, 1,...,N.

INTERNAL VARIABLES:
* `h`: El tamaño del paso del algoritmo (un número real positivo).
* `m`: El número de condiciones iniciales especificadas (un número entero positivo).
* `j`, `k`: Variables enteras utilizadas como índices.
* `a`, `b`: Arreglos de N + 1 números reales que contienen los pesos de las fórmulas de corrector y predictor, respectivamente.
* `P`: El valor predicho (una variable real).
"""

function old_solve!(sys!, α, y0, p, T, N)
    # Estructura de la función
    function f(t, u, p)
        du = zeros(length(α[:]))
        sys!(du, u, t, p)
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
