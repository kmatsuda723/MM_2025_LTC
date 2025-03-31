# ======================================================= #
# Model of Aiyagari (1994)                                #
# By Sagiri Kitao (Translated in Julia by Taiki Ono)      #
# ======================================================= #

# import libraries
using Plots
using NLsolve
using Optim
using Random
using Distributions
using LaTeXStrings
using Parameters # enable @unpack

# making nlsove easier
function nls(func, params...; ini=[0.0])
    if typeof(ini) <: Number
        r = nlsolve((vout, vin) -> vout[1] = func(vin[1], params...), [ini])
        v = r.zero[1]
    else
        r = nlsolve((vout, vin) -> vout .= func(vin, params...), ini)
        v = r.zero
    end
    return v, r.f_converged
end

function tauchen(N, rho, sigma, param)
    """
    ---------------------------------------------------
    === AR(1)過程をtauchenの手法によって離散化する関数 ===
    ---------------------------------------------------
    ※z'= ρ*z + ε, ε~N(0,σ_{ε}^2) を離散化する

    <input>
    ・N: 離散化するグリッドの数
    ・rho: AR(1)過程の慣性(上式のρ)
    ・sigma: AR(1)過程のショック項の標準偏差(上式のσ_{ε})
    ・m: 離散化するグリッドの範囲に関するパラメータ
    <output>
    ・Z: 離散化されたグリッド
    ・Zprob: 各グリッドの遷移行列
    ・Zinv: Zの定常分布
    """
    Zprob = zeros(N, N) # 遷移確率の行列
    Zinv = zeros(N, 1)  # 定常分布

    # 等間隔のグリッドを定める
    # 最大値と最小値
    zmax = param * sqrt(sigma^2 / (1 - rho^2))
    zmin = -zmax
    # グリッド間の間隔
    w = (zmax - zmin) / (N - 1)

    Z = collect(range(zmin, zmax, length=N))

    # グリッド所与として遷移確率を求める
    for j in 1:N # 今期のZのインデックス
        for k in 1:N  # 来期のZのインデックス
            if k == 1
                Zprob[j, k] = cdf_normal((Z[k] - rho * Z[j] + w / 2) / sigma)
            elseif k == N
                Zprob[j, k] = 1 - cdf_normal((Z[k] - rho * Z[j] - w / 2) / sigma)
            else
                Zprob[j, k] = cdf_normal((Z[k] - rho * Z[j] + w / 2) / sigma) - cdf_normal((Z[k] - rho * Z[j] - w / 2) / sigma)
            end
        end
    end

    # 定常分布を求める
    dist0 = (1 / N) .* ones(N)
    dist1 = copy(dist0)

    err = 1.0
    errtol = 1e-8
    iter = 1
    while err > errtol

        dist1 = Zprob' * dist0
        err = sum(abs.(dist0 - dist1))
        dist0 = copy(dist1)
        iter += 1

    end

    Zinv = copy(dist1)

    return Z, Zprob, Zinv

end


function cdf_normal(x)
    """
    --------------------------------
    === 標準正規分布の累積分布関数 ===
    --------------------------------
    <input>
    ・x: 
    <output>
    ・c: 標準正規分布にしたがう確率変数Xがx以下である確率
    """
    d = Normal(0, 1) # 標準正規分布
    c = cdf(d, x)

    return c

end

function interp(x, grid)
    # Find indices of the closest grids and the weights for linear interpolation
    ial = searchsortedlast(grid, x)  # Index of the grid just above or equal to x
    ial = max(1, ial)  # Ensure index is within bounds

    if ial > length(grid) - 1
        ial = length(grid) - 1  # Handle case where x is beyond the grid
    end

    iar = ial + 1  # The index just below ial

    # Compute the weights for interpolation
    varphi = (grid[iar] - x) / (grid[iar] - grid[ial])
    return ial, iar, varphi
end


function setParameters(;
    JJ=10,
    NI=2,
    NH=3,
    NZ=16,
    mu=3.0,             # risk aversion (=3 baseline)             
    beta=0.96,            # subjective discount factor 
    delta=0.08,            # depreciation
    alpha=0.36,            # capital's share of income
    zeta=2.0,
    c_e=1.0,
    c_f=1.0,
    b=0.0,             # borrowing limit
    NTHETA=4,             # number of discretized states
    rho=0.6,           # first-order autoregressive coefficient
    sig=0.4           # intermediate value to calculate sigma (=0.4 BASE)
)

    # ================================================= #
    #  COMPUTE TRANSITION MATRIX OF LABOR PRODUCTIVITY  #
    # ================================================= #

    # ROUTINE tauchen.param TO COMPUTE TRANSITION MATRIX, GRID OF AN AR(1) AND
    # STATIONARY DISTRIBUTION
    # approximate labor endowment shocks with seven states Markov chain
    # log(s_{t}) = rho*log(s_{t-1})+e_{t} 
    # e_{t}~ N(0,sig^2)

    M = 2.0
    logs, prob, invdist = tauchen(NTHETA, rho, sig, M)
    s = exp.(logs)

    logz, prob_z, inv_z = tauchen(NZ, rho, sig, M)
    z = exp.(logz)

    logh, prob_h, invdist = tauchen(NH, rho, sig, M)
    h = exp.(logh)

    # ================================================= #
    #  HUMAN CAPITAL INVESTMENT                         #
    # ================================================= #

    # is = 1: home care, is = 2: public is = 3: private

    # tuition
    NS = 3
    tuition = zeros(NS)
    tuition[1] = 0.0
    tuition[2] = 0.0
    tuition[3] = 0.0


    # productivity
    prod = zeros(NS)
    prod[1] = 0.0 # productivity not increasing in ability with no education
    prod[2] = 3.0 # productivity increasing in ability with education
    prod[3] = 2.0 # productivity increasing in ability with education

    pi_i = 0.5

    NA = 20                                     # grid size for STATE 
    NA2 = 20                                     # grid size for CONTROL

    return (JJ=JJ, NI=NI, NH=NH, prob_h=prob_h, mu=mu, beta=beta, delta=delta, alpha=alpha, b=b, NTHETA=NTHETA, s=s, prob=prob, tuition=tuition, prod=prod,
        NA=NA, NA2=NA2, NS=NS, NZ=NZ, z=z, pi_i=pi_i, zeta=zeta, prob_z=prob_z, inv_z=inv_z, c_e=c_e, c_f=c_f)
end

function set_prices(param, KL)

    # ================================================= #
    #  SETTING INTEREST, WAGE, and ASSET GRIDS          #
    # ================================================= #

    r = param.alpha * ((KL)^(param.alpha - 1)) - param.delta # interest rate
    wage = (1 - param.alpha) * ((param.alpha / (r + param.delta))^param.alpha)^(1 / (1 - param.alpha)) # wage

    # -phi is borrowing limit, b is adhoc
    # the second term is natural limit
    if r <= 0.0
        phi = param.b
    else
        phi = min(param.b, wage * param.s[1] / r)
    end

    # capital grid (need define in each iteration since it depends on r/phi)
    maxK = 20                                    # maximum value of capital grid
    minK = -phi                                  # borrowing constraint
    curvK = 2.0

    # grid for state
    gridk = zeros(param.NA)
    gridk[1] = minK
    for ia in 2:param.NA
        gridk[ia] = gridk[1] + (maxK - minK) * ((ia - 1) / (param.NA - 1))^curvK
    end

    # grid for optimal choice
    gridk2 = zeros(param.NA2)
    gridk2[1] = minK
    for ia in 2:param.NA2
        gridk2[ia] = gridk2[1] + (maxK - minK) * ((ia - 1) / (param.NA2 - 1))^curvK
    end

    return (r=r, wage=wage, phi=phi, gridk=gridk, gridk2=gridk2)

end

function solve_firm(p, param, prices)
    @unpack JJ, NI, NA, NZ, NTHETA, NH, NS, NA2, tuition, prod, mu, s, beta, prob, prob_h, prob_z, pi_i, zeta, z, c_f = param
    @unpack r, wage = prices

    # initialize some variables
    v_f = zeros(NI, NZ)
    v_f_new = zeros(NI, NZ)
    n = zeros(NI, NZ)
    exit = zeros(NI, NZ)

    err = 10   # error between old policy index and new policy index
    maxiter = 2000 # maximum number of iteration 
    iter = 1    # iteration counter

    while (err > 1e-3) & (iter < maxiter)

        # tabulate the utility function such that for zero or negative
        # consumption utility remains a large negative number so that
        # such values will never be chosen as utility maximizing

        @inbounds for ii in 1:NI
            Threads.@threads for iz in 1:NZ
                n[ii, iz] = (p[ii]*z[iz]/wage)^(1.0 / zeta)

                vpr = 0.0 # next period's value function given (l,k')
                for izp in 1:NZ # expectation of next period's value function
                    vpr += prob_z[iz, izp] * v_f[ii, izp]
                end
                vpr = p[ii]*z[iz]*n[ii, iz] - wage/(1.0 + zeta)*(n[ii, iz]^(1.0 + zeta)) - c_f + vpr/(1.0 + r)

                if vpr < 0.0
                    v_f_new[ii, iz] = 0.0
                    exit[ii, iz] = 1.0
                else
                    v_f_new[ii, iz] = vpr
                    exit[ii, iz] = 0.0
                end
            end
        end
        err = maximum(abs.(v_f[:, :] - v_f_new[:, :]))
        v_f .= v_f_new
        iter += 1
    end


    if iter == maxiter
        println("WARNING!! @aiyagari_vfi2.jl VFI (firm): iteration reached max: iter=$iter,e rr=$err")
    end

    # Return household decisions as a struct
    return (n=n, exit=exit, v_f=v_f)
end

function solve_p(p, param, prices)
    @unpack c_e, inv_z, NI = param
    n, exit, v_f = solve_firm(p, param, prices)
    dist = zeros(NI)
    for ii in 1:NI
        dist[ii] = sum(inv_z .* v_f[ii, :]) - c_e
    end
    return dist
end


function solve_household(param, prices)
    @unpack JJ, NI, NA, NTHETA, NH, NS, NA2, tuition, prod, mu, s, beta, prob, prob_h, pi_i = param
    @unpack r, wage, gridk, gridk2 = prices

    # initialize some variables
    kfunG = zeros(JJ, NI, NTHETA, NA, NH, NS)    # new index of policy function 
    kfun = similar(kfunG)     # policy function   
    v = zeros(JJ, NI, NTHETA, NA, NH, NS)        # old value function
    v_new = zeros(JJ, NI, NTHETA, NA, NH, NS)        # old value function

    tv = similar(kfunG)       # new value function
    # kfunG_old = zeros(JJ, NI, NTHETA, NA, NS) # old policy function 
    sfun = zeros(Int, JJ, NI, NTHETA, NA, NH, NS) # old policy function 
    # sfun_old = zeros(Int, JJ, NI, NTHETA, NA, NS) # old policy function 

    err = 10   # error between old policy index and new policy index
    maxiter = 2000 # maximum number of iteration 
    iter = 1    # iteration counter

    while (err > 1e-3) & (iter < maxiter)

        # tabulate the utility function such that for zero or negative
        # consumption utility remains a large negative number so that
        # such values will never be chosen as utility maximizing

        @inbounds for ij in JJ:-1:1
            for ii in 1:NI
                Threads.@threads for ia in 1:NA # k(STATE)
                    for itheta in 1:NTHETA # h(STATE)
                        for ih in 1:NH
                            for is in 1:NS

                                kccmax = NA2 # maximum index that satisfies c>0.0 
                                vtemp = fill(-1e6, NA2, NS)

                                if is == 1
                                    min_isp = 1
                                    max_isp = NS
                                else
                                    min_isp = is
                                    max_isp = is
                                end

                                for ikp in 1:NA2 # k'(CONTROL)
                                    for isp in min_isp:max_isp

                                        # amount of consumption given (k,l,k')
                                        cons = s[itheta]^prod[is] * wage + (1 + r) * gridk[ia] - gridk2[ikp] - tuition[isp]

                                        if cons <= 0.0
                                            # penalty for c<0.0
                                            # once c becomes negative, vtemp will not be updated(=large negative number)
                                            kccmax = ikp - 1
                                            break
                                        end

                                        util = (cons^(1 - mu)) / (1 - mu)

                                        # interpolation of next period's value function
                                        # find node and weight for gridk2 (evaluating gridk2 in gridk) 
                                        ial, iar, varphi = interp(gridk2[ikp], gridk)

                                        vpr = 0.0 # next period's value function given (l,k')
                                        for ithetap in 1:NTHETA # expectation of next period's value function
                                            for ihp in 1:NH
                                                if ij == JJ
                                                    vpr += prob[itheta, ithetap] * prob_h[ih, ihp] * (varphi * v[1, ii, ithetap, ial, ihp, 1] + (1.0 - varphi) * v[1, ii, ithetap, iar, ihp, 1])
                                                else
                                                    if is == 1 && isp == 2
                                                        vpr += pi_i * prob[itheta, ithetap] * prob_h[ih, ihp] * (varphi * v[ij+1, ii, ithetap, ial, ihp, 2] + (1.0 - varphi) * v[ij+1, ii, ithetap, iar, ihp, 2])
                                                        vpr += (1.0 - pi_i) * prob[itheta, ithetap] * prob_h[ih, ihp] * (varphi * v[ij+1, ii, ithetap, ial, ihp, 1] + (1.0 - varphi) * v[ij+1, ii, ithetap, iar, ihp, 1])
                                                    else
                                                        vpr += prob[itheta, ithetap] * prob_h[ih, ihp] * (varphi * v[ij+1, ii, ithetap, ial, ihp, isp] + (1.0 - varphi) * v[ij+1, ii, ithetap, iar, ihp, isp])
                                                    end
                                                end
                                            end
                                        end

                                        vtemp[ikp, isp] = util + beta * vpr
                                    end

                                end

                                # find k' that  solves bellman equation
                                # max_val, max_index = findmax(vtemp[1:kccmax, 1:NS]) # subject to k' achieves c>0
                                max_index = argmax(vtemp[1:kccmax, 1:NS])  # subject to k' achieves c>0

                                t2, t3 = Tuple(max_index)
                                tv[ij, ii, itheta, ia, ih, is] = vtemp[t2, t3]
                                kfunG[ij, ii, itheta, ia, ih, is] = t2
                                kfun[ij, ii, itheta, ia, ih, is] = gridk2[t2]
                                sfun[ij, ii, itheta, ia, ih, is] = t3


                            end
                        end
                    end
                end
            end
        end

        v_new .= tv
        # err = maximum(abs.(kfunG - kfunG_old)) + maximum(abs.(sfun - sfun_old))
        err = maximum(abs.(v[1, :, :, :, :, :] - v_new[1, :, :, :, :, :]))

        v .= v_new

        iter += 1

    end

    if iter == maxiter
        println("WARNING!! @aiyagari_vfi2.jl VFI: iteration reached max: iter=$iter,e rr=$err")
    end

    # Return household decisions as a struct
    return (
        kfun=kfun, kfunG=kfunG, sfun=sfun
    )
end


function get_distribution(param, dec, prices)
    @unpack JJ, NI, NA, NH, NTHETA, NS, NA2, tuition, prod, mu, pi_i, prob, prob_h = param
    @unpack kfun, kfunG, sfun = dec
    @unpack r, wage, gridk, gridk2 = prices

    mea0 = zeros(JJ, NI, NTHETA, NA, NH, NS) # new distribution

    # calculate stationary distribution
    # for ij in 1:JJ
    mea0[1, :, :, :, :, :] = ones(NI, NTHETA, NA, NH, NS) / (NI * NTHETA * NA * NH * NS) # old distribution
    # end
    # mea1 = zeros(JJ, NI, NTHETA, NA, NH, NS) # new distribution
    err = 1
    errTol = 0.00001
    maxiter = 2000
    iter = 1
    meanL = 0.0


    while (err > errTol) & (iter < maxiter)
        meanL = 0.0
        mea1 = zeros(JJ, NI, NTHETA, NA, NH, NS)

        @inbounds for ij in 1:JJ
            for ii in 1:NI
                for ia in 1:NA # k
                    for itheta in 1:NTHETA # l
                        for ih in 1:NH
                            for is in 1:NS # s

                                if is == 1
                                    isp = sfun[ij, ii, itheta, ia, ih, is] # index of s'(k,l,s) next gen's education
                                else
                                    isp = is
                                end

                                # interpolation of policy function 
                                # split to two grids in gridk
                                ial, iar, varphi = interp(kfun[ij, ii, itheta, ia, ih, is], gridk)

                                for ithetap in 1:NTHETA # l'
                                    for ihp in 1:NH
                                        if ij == JJ
                                            mea1[1, ii, ithetap, ial, ihp, 1] += prob[itheta, ithetap] * prob_h[ih, ihp] * varphi * mea0[ij, ii, itheta, ia, ih, is]
                                            mea1[1, ii, ithetap, iar, ihp, 1] += prob[itheta, ithetap] * prob_h[ih, ihp] * (1.0 - varphi) * mea0[ij, ii, itheta, ia, ih, is]
                                        else
                                            if is == 1 && isp == 2
                                                mea1[ij+1, ii, ithetap, ial, ihp, 2] += pi_i * prob[itheta, ithetap] * prob_h[ih, ihp] * varphi * mea0[ij, ii, itheta, ia, ih, is]
                                                mea1[ij+1, ii, ithetap, iar, ihp, 2] += pi_i * prob[itheta, ithetap] * prob_h[ih, ihp] * (1.0 - varphi) * mea0[ij, ii, itheta, ia, ih, is]
                                                mea1[ij+1, ii, ithetap, ial, ihp, 1] += (1.0 - pi_i) * prob[itheta, ithetap] * prob_h[ih, ihp] * varphi * mea0[ij, ii, itheta, ia, ih, is]
                                                mea1[ij+1, ii, ithetap, iar, ihp, 1] += (1.0 - pi_i) * prob[itheta, ithetap] * prob_h[ih, ihp] * (1.0 - varphi) * mea0[ij, ii, itheta, ia, ih, is]
                                            else
                                                mea1[ij+1, ii, ithetap, ial, ihp, isp] += prob[itheta, ithetap] * prob_h[ih, ihp] * varphi * mea0[ij, ii, itheta, ia, ih, is]
                                                mea1[ij+1, ii, ithetap, iar, ihp, isp] += prob[itheta, ithetap] * prob_h[ih, ihp] * (1.0 - varphi) * mea0[ij, ii, itheta, ia, ih, is]
                                            end
                                        end
                                    end
                                end

                                meanL += param.s[itheta]^prod[is] * mea0[ij, ii, itheta, ia, ih, is]
                            end
                        end
                    end
                end
            end
        end

        err = maximum(abs.(mea1[1, :, :, :, :, :] - mea0[1, :, :, :, :, :]))
        # println([iter,err])

        mea0 .= mea1
        iter += 1

    end

    # error(sum(mea0))

    # error(vec(sum(mea0, dims=(1, 2, 3, 4))))

    # mass_y = vec(sum(mea0, dims=(1, 2, 3, 4)))


    if iter == maxiter
        println("WARNING!! @aiyagari_vfi2.jl INVARIANT DIST: iteration reached max: iter=$iter, err=$err")
    end

    meank = sum(sum(mea0 .* kfun))

    return (
        meank=meank,
        meanL=meanL
    )
end

function get_distribution_firm(param, dec, firm_dec, prices)
    @unpack NI, NZ, prob_z, inv_z = param
    @unpack exit = firm_dec

    err = 1
    errTol = 0.00001
    maxiter = 2000
    iter = 1
    mea0 = zeros(NI, NZ)

    while (err > errTol) & (iter < maxiter)
        mea1 = zeros(NI, NZ)
        @inbounds for ii in 1:NI
            for iz in 1:NZ
                for izp in 1:NZ
                    mea1[ii, izp] += prob_z[iz, izp]*(1.0 - exit[ii, iz])*(mea0[ii, iz] + inv_z[iz])
                end
            end
        end
        err = maximum(abs.(mea1[:, :] - mea0[:, :]))
        mea0 .= mea1
        iter += 1
    end

    inc_m_f = zeros(NI, NZ)
    @inbounds for ii in 1:NI
        for iz in 1:NZ
            inc_m_f[ii, iz] = (1.0 - exit[ii, iz])*(mea0[ii, iz] + inv_z[iz])
        end
    end

    if iter == maxiter
        println("WARNING!! @aiyagari_vfi2.jl INVARIANT DIST: iteration reached max: iter=$iter, err=$err")
    end

    return (inc_m_f=inc_m_f)
end


function get_Steadystate(param)

    # ======================= #
    #  COMPUTE K and r in EQ  #
    # ======================= #

    K0 = 6.8 # initial guess
    L0 = 1.0

    err2 = 1
    errTol = 0.01
    maxiter = 100
    iter = 1
    adj = 0.1
    # gridk = zeros(param.NA)

    KL0 = K0 / L0

    while (err2 > errTol) && (iter < maxiter)

        # set prices given K/L
        prices = set_prices(param, KL0)



        p = nls(solve_p, param, prices, ini=[1.0, 1.0])[1]

        firm_dec = solve_firm(p, param, prices)

        # solve household problems for decision rules
        dec = solve_household(param, prices)

        # solve stationary distribution for aggregates K and L
        K1, L1 = get_distribution(param, dec, prices)

        KL1 = K1 / L1

        err2 = abs(KL0 - KL1) / abs(KL1)

        # UPDATE GUESS AS K0+adj*(K1-K0)

        inc_m_f = get_distribution_firm(param, dec, firm_dec, prices)

        error(inc_m_f)

        println([iter, KL0, KL1, err2])

        if err2 > errTol
            KL0 += adj * (KL1 - KL0)
            iter += 1
        end

    end

    if iter == maxiter
        println("WARNING!! iter=$maxiter, err=$err2")
    end

    prices = set_prices(param, KL0)
    kfun0, kfunG, sfun = solve_household(param, prices)
    gridk0 = prices.gridk

    return (kfun0=kfun0, sfun=sfun, gridk0=gridk0)

end

# ======================= #
#  MAIN                   #
# ======================= #

# set parameters
param = setParameters()

# solve for steady state
kfun0, sfun, gridk = get_Steadystate(param)

# plot
plot(gridk, kfun0[1, 1, 1, :, :, 1], color=:blue, linestyle=:solid, linewidth=2, label=L"hs,l_{low}",
    title="Policy function", xlabel=L"a", ylabel=L"a'=g(a,l)", xlims=(-3, 10), ylims=(-3, 10), legend=:topleft)
plot!(gridk, kfun0[1, 1, 4, :, :, 1], color=:red, linestyle=:solid, linewidth=2, label=L"hs,l_{mid}")
plot!(gridk, kfun0[1, 1, 4, :, :, 1], color=:black, linestyle=:solid, linewidth=2, label=L"hs,l_{high}")
plot!(gridk, kfun0[1, 1, 1, :, :, 2], color=:blue, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
plot!(gridk, kfun0[1, 1, 4, :, :, 2], color=:red, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
plot!(gridk, kfun0[1, 1, 4, :, :, 2], color=:black, linestyle=:dash, linewidth=2, label=L"cg,l_{high}")
savefig("fig_kfun.pdf")

plot(gridk, sfun[1, 1, 1, :, :, 1], color=:blue, linestyle=:solid, linewidth=2, label=L"hs,l_{low}",
    title="Policy function", xlabel=L"a", ylabel=L"a'=g(a,l)", xlims=(-3, 10), ylims=(0, 3), legend=:topleft)
plot!(gridk, sfun[1, 1, 4, :, :, 1], color=:red, linestyle=:solid, linewidth=2, label=L"hs,l_{mid}")
plot!(gridk, sfun[1, 1, 4, :, :, 1], color=:black, linestyle=:solid, linewidth=2, label=L"hs,l_{high}")
plot!(gridk, sfun[1, 1, 1, :, :, 2], color=:blue, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
plot!(gridk, sfun[1, 1, 4, :, :, 2], color=:red, linestyle=:dash, linewidth=2, label=L"cg,l_{mid}")
plot!(gridk, sfun[1, 1, 4, :, :, 2], color=:black, linestyle=:dash, linewidth=2, label=L"cg,l_{high}")
savefig("fig_sfun.pdf")



