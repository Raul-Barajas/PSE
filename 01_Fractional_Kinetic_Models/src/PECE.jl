using SpecialFunctions: gamma

# Vector de estado
function F!(dx, sys!, x, t, p)
    sys!(dx, x, t, p)
    return dx
end

"""
    solve(sys!, alpha, x0, p, T, N)

Integrate a fractional system with the PECE method.

`sys!` must use the signature `sys!(dx, x, t, p)`. The solution is returned as
a matrix with one state per row and one time point per column.
"""
function solve(sys!, α, x0, p, T, N)

    # Parametros
    h = T / N                                                       # h: Tamaño del paso principal
    ξ = h / 10                                                      # ξ: Tiempo pequeño usado para regularizar el bloque inicial
    ξf = ξ / 10                                                     # ξf: Tiempo usado solo en la primera evaluación regularizada de f
    des = 1e-4                                                      # des: Perturbación para evitar singularidades en x(0)
    n = length(α)                                                   # n: Número de ecuaciones del sistema
    Tval = promote_type(typeof(float(h)), eltype(α), eltype(x0))    # Tval: Tipo numérico común para toda la integración
    x = zeros(Tval, n, N + 1)                                       # x: Matriz de solución, una columna por instante
    fun = zeros(Tval, n, N + 1)                                     # fun: Historial de la función evaluada en cada paso

    αv = Vector{Tval}(undef, n)
    x0v = Vector{Tval}(undef, n)
    ξγ1 = Vector{Tval}(undef, n)
    hγ1 = Vector{Tval}(undef, n)
    hγ2 = Vector{Tval}(undef, n)
    powα = Matrix{Tval}(undef, n, N + 2)
    powα1 = Matrix{Tval}(undef, n, N + 2)

    hT = Tval(h)
    ξT = Tval(ξ)
    ξfT = Tval(ξf)
    desT = Tval(des)
    zeroT = zero(Tval)
    oneT = one(Tval)
    twoT = oneT + oneT

    @inbounds for k in 1:n
        αk = Tval(α[k])
        x0k = Tval(x0[k])
        αv[k] = αk
        x0v[k] = x0k
        x[k, 1] = x0k

        ξpow = ξT^αk
        hpow = hT^αk
        ξγ1[k] = ξpow / gamma(αk + oneT)
        hγ1[k] = hpow / gamma(αk + oneT)
        hγ2[k] = hpow / gamma(αk + twoT)

        for m in 0:N+1
            mT = Tval(m)
            powα[k, m+1] = mT^αk
            powα1[k, m+1] = mT^(αk + oneT)
        end
    end

    P = zeros(Tval, n)
    Fp = zeros(Tval, n)
    Fy = zeros(Tval, n)
    x1_des = zeros(Tval, n)
    f_x1 = zeros(Tval, n)
    f_eval = zeros(Tval, n)

    # Estructura de la función
    f!(dx, state, t, p) = F!(dx, sys!, state, t, p)

    # Predictor inicial regularizado
    @inbounds @simd for k in 1:n
        x1_des[k] = x[k, 1] + desT
    end

    f!(f_x1, x1_des, ξfT, p)
    x2 = @view x[:, 2]
    @inbounds @simd for k in 1:n
        x2[k] = x0v[k] + ξγ1[k] * f_x1[k]
    end

    # Modificar segun el numero de iteraciones para la condicion inicial
    for _ in 1:10
        f!(f_eval, x2, ξT, p)
        @inbounds @simd for k in 1:n
            x2[k] = x0v[k] + ξγ1[k] * f_eval[k]
        end
    end

    f!(f_eval, x2, ξT, p)
    @inbounds @simd for k in 1:n
        fun[k, 1] = f_eval[k]
    end

    # Integración numérica
    @inbounds for i in 0:N-1

        fill!(Fp, zeroT)
        fill!(Fy, zeroT)
        iT = Tval(i)

        for j in 0:i
            d = i - j
            funj = @view fun[:, j+1]

            if j == 0
                @simd for k in 1:n
                    b = powα[k, d+2] - powα[k, d+1]
                    a = powα1[k, i+1] - (iT - αv[k]) * powα[k, i+2]
                    val = funj[k]
                    Fp[k] += b * val
                    Fy[k] += a * val
                end
            else
                @simd for k in 1:n
                    b = powα[k, d+2] - powα[k, d+1]
                    a = powα1[k, d+3] + powα1[k, d+1] - twoT * powα1[k, d+2]
                    val = funj[k]
                    Fp[k] += b * val
                    Fy[k] += a * val
                end
            end
        end

        # Predictor
        @inbounds @simd for k in 1:n
            P[k] = x0v[k] + hγ1[k] * Fp[k]
        end

        # Corrector
        tnext = (i + 1) * hT
        f!(f_eval, P, tnext, p)
        xnext = @view x[:, i+2]
        @inbounds @simd for k in 1:n
            xnext[k] = x0v[k] + hγ2[k] * (f_eval[k] + Fy[k])
        end

        f!(f_eval, xnext, tnext, p)
        @inbounds @simd for k in 1:n
            fun[k, i+2] = f_eval[k]
        end

    end

    # Reemplazar NaN e Inf por 0 sin crear máscaras temporales
    @inbounds for idx in eachindex(x)
        if isnan(x[idx]) || isinf(x[idx])
            x[idx] = zeroT
        end
    end

    return x
end

"""
    solve!(sys!, alpha, x0, p, T, N)

Compatibility alias for [`solve`](@ref). Prefer `solve` for new code.
"""
const solve! = solve
