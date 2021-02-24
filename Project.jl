using Plots
using LaTeXStrings
using DelimitedFiles
using Statistics
using FFTW
using QuadGK
using NumericalIntegration
using LsqFit
using StatsBase
using LinearAlgebra

include("helpers.jl")

theme(:wong)
pyplot() # for "speed" (draft)
pgfplotsx() # for detailed latex plots
default(titlefont = (20, "times"), legendfontsize = 12, guidefont = (20, :black),
    tickfont = (14, :black), framestyle = :box,legend = :topright, markerstrokewidth = 0.3,
    minorgrid = true, minorgridlinewidth= 0.2 ,gridalpha = 0.12 , minorgridalpha = 0.15, minorticks = true ,
    gridlinewidth = 1.5  ,  dpi = 100) # , gridalpha = 0.18 , minorgridalpha = 0.21 for svg export to pdf




# Read the data and create the time scale. =======================================================
τ = 1/20000
v = readdlm("ME467_Project20/veldata/veldata.txt", ' ', Float64)[:,1]
N = length(v)
t = 0:τ:N*τ


# Check taylor frozen flow approx + transform time in space ======================================
u = mean(v)
V2 = mean((v.-u).^2)
Taylor_fr_fl_cond = 1/u*sqrt(V2)
# Taylor flow is valid if it is << 1 , here we have 0.125448, valid but not perfect.
x = u .* t
v = reverse(v) # we flip the speed (should not matter much) to have the last measure in time at x = 0
L =x[end]
dx = abs(x[1]-x[2])

# COmputation of the Autocorrelation funciton  ===================================================
v_m = v .-  u
Correlation = 1/N.*ifft(abs2.(fft(v_m)))./mean((v_m).^2)
L_c = x[argmin(abs.(abs.(Correlation) .- 1/ℯ))]
L_c_graph = round(L_c , digits =3)


head = 100
tail_fit= 2830
to_fit_y = log.(abs.(Correlation[head:tail_fit]))
to_fit_x = x[head:tail_fit]
@. model(x, p) = p[1]*x + p[2]
p0 = [0.6, 0.5]
fit = curve_fit(model, to_fit_x, to_fit_y, p0)
L_c_acc = abs(1/fit.param[1])
L_c_acc_graph = round(L_c_acc , digits =3)
Error_L_acc = round(1/L_c_acc^2*stderror(fit)[1],digits = 3) # using error propagatipon

tail = tail_fit
L_int = integrate(x[1:tail], abs.(Correlation[1:tail]))


# COmputation of Energy spectrum  ================================================================
E_t = 1/4 * 1/(pi) .* abs2.(fft(v_m))./N .* (u*τ)
k = 0: 1 / N /τ ./ u *2pi : 1/2  /τ / u *2pi
half= Int(ceil(N/ 2))
E = 2 .* fftshift(E_t)[half:end]

sg=SG(79,3)
E_smooth = apply_filter(sg[:,1],E)
E_red = smooth_reduce(E_smooth)
k_red = 0: 1 / N /τ ./ u *10 *2pi : 1/2  /τ / u *2pi
k_red_2 = 0: 1 / N /τ ./ u *1000 *2pi : 1/2  /τ / u *2pi
k_pow = k_red_2.^(-5/3)

int_E = integrate(k[1:end], (E[1:end])) # for normalisation check


# Reynolds Taylor Number  =====================================================================

ν = 1.516 * 10^(-5) # at 20 C air visc
V2 = mean((v_m).^2)
ϵ = 1/2 * V2 ^(3. / 2) / L_c_acc
λ = sqrt(15 * ν * V2 / ϵ)
Re_l= sqrt(V2) * λ / ν
Re = 0.1*u / ν

λ_graph = round(λ, digits = 2)

ϵ_error = 1/2 * V2 ^(3. / 2) / L_c_acc^2 * Error_L_acc
λ_error = 1/2 * sqrt(15 * ν * V2 ) * ϵ^(-3/2) * ϵ_error
Re_l_error = sqrt(V2) / ν * λ_error
#using the errors on L_c, compute the errors on the rest of the values.

# Velocity increment computation  =============================================================
l = [0.0005 0.01 0.1 10]
Delta_data = Int.(ceil.(l ./ dx))

δv = zeros(4,N)

for k = 1:4
    Threads.@threads for i = 1:N-Delta_data[k] -1
        δv[k,i] = v_m[i+Delta_data[k]] - v_m[i]
    end
end


# Pdf computation (fits) ======================================================================

Hist_fit_1 = normalize(StatsBase.fit(Histogram , δv[1,1:end-100000], nbins = 10000))
Hist_fit_2 = normalize(StatsBase.fit(Histogram , δv[2,1:end-10000], nbins = 10000))
Hist_fit_3 = normalize(StatsBase.fit(Histogram , δv[3,1:end-1000], nbins = 10000))
Hist_fit_4 = normalize(StatsBase.fit(Histogram , δv[4,1:end-100], nbins = 10000))

xfit_1 = Hist_fit_1.edges[1][1:end-1]
xfit_2 = Hist_fit_2.edges[1][1:end-1]
xfit_3 = Hist_fit_3.edges[1][1:end-1]
xfit_4 = Hist_fit_4.edges[1][1:end-1]

m(x, p) = p[1].*exp.( -1/2 .*(x .- p[2]).^2 ./ p[3]^2 )
p0 = [0, 0 , 3.1]
fit_1 = curve_fit(m, vec(xfit_1) , vec(Hist_fit_1.weights), p0)
fit_2 = curve_fit(m, vec(xfit_2) , vec(Hist_fit_2.weights), p0)
fit_3 = curve_fit(m, vec(xfit_3) , vec(Hist_fit_3.weights), p0)
fit_4 = curve_fit(m, vec(xfit_4) , vec(Hist_fit_4.weights), p0)

yfit_1 = m(xfit_1,fit_1.param)
yfit_2 = m(xfit_2,fit_2.param)
yfit_3 = m(xfit_3,fit_3.param)
yfit_4 = m(xfit_4,fit_4.param)

m_pow(x,p) = p[1] ./ ( (p[2] .- x).^(2) .- p[3] )  .+ p[4]
lb = [0.1, 0, -15 , -1]
ub = [3, 12.,  0 , 1]
p0_bounds = [0.1, 0.03, -0.1, 0]
fit_power_law = curve_fit(m_pow, vec(xfit_2) , vec(Hist_fit_2.weights), p0_bounds, lower=lb, upper=ub)
y_fit_power_2 = m_pow(xfit_2,fit_power_law.param)

m_pow_4(x,p) = p[1] ./ ( abs.(p[2] .- x).^(3) .- p[3] ) .+ p[4]
lb = [0.1, 0, -15 , -1]
ub = [3, 12.,  0 , 1]
p0_bounds = [0.1, 0.03, -0.1, 0]
fit_power_law_4 = curve_fit(m_pow_4, vec(xfit_2) , vec(Hist_fit_2.weights), p0_bounds, lower=lb, upper=ub)
y_fit_power_2_4 = m_pow_4(xfit_2,fit_power_law_4.param)

sum_rees_gau = round(sum(abs.(fit_2.resid)),digits =3)
sum_rees_2 = round(sum(abs.(fit_power_law.resid)),digits =3)
sum_rees_3 = round(sum(abs.(fit_power_law_4.resid)),digits =3)

# Structure function computation  =============================================================
range_l = 300
δ_struct = ones(range_l,3)


Threads.@threads for k = 1:range_l

    shift = Int(floor(10^(k*0.02)))
    δ_struct[k,1] = mean((v_m[1+shift:end] .- v_m[1:end-shift]).^2)
    δ_struct[k,2] = mean((v_m[1+shift:end] .- v_m[1:end-shift]).^3)
    δ_struct[k,3] = mean((v_m[1+shift:end] .- v_m[1:end-shift]).^4)

    # print("Iteration $k over $range_l \n")
end

sg=SG(7,3)
for k = 1:3
    δ_struct[:,k] = apply_filter(sg[:,1],δ_struct[:,k])
end

l_struct = 1:range_l
l_struct = 10 .^(0.02.*l_struct) .* dx
l_2_pow = l_struct.^(2/3)
l_3_pow = 4/5 * ϵ .* l_struct
l_4_pow = l_struct.^(4/3)

# fits for E and the struct function

head_fit = argmin(abs.(abs.(k_red) .- 2pi/L_c_acc))
tail_fit= argmin(abs.(abs.(k_red) .- 2pi/λ))
to_fit_y = (abs.(E_red)[head_fit:tail_fit])
to_fit_x = k_red[head_fit:tail_fit]
@. model(x, p) = p[1]*x^p[2]
p0 = [-1.6, -1.5]
fit_E = curve_fit(model, to_fit_x, to_fit_y, p0)
E_fit = fit_E.param[2]
Error_E_fit = stderror(fit_E)[2]
y_fit_E = (model(k_red,fit_E.param))

head_fit = argmin(abs.(abs.(l_struct) .- λ)) +4
tail_fit= argmin(abs.(abs.(l_struct) .- L_c_acc))-9
to_fit_y = abs.(δ_struct[head_fit:tail_fit,1])
to_fit_x = l_struct[head_fit:tail_fit]
@. model(x, p) = p[1]*x^p[2]
p0 = [1.6, 0.3]
fit_S2 = curve_fit(model, to_fit_x, to_fit_y, p0)
S2_fit = fit_S2.param[2]
Error_S2_fit = stderror(fit_S2)[2]
eps_fit = fit_S2.param[1]
Error_eps_fit = stderror(fit_S2)[1]
y_fit_S2 = (model(l_struct,fit_S2.param))

eps_S3 = (eps_fit/2.1)^(3/2)
err_eps_S3 = (eps_fit/2.1)^(1/2)*3/2*Error_eps_fit

to_fit_y = abs.(δ_struct[head_fit:tail_fit,2])
to_fit_x = l_struct[head_fit:tail_fit]
@. model(x, p) = p[1]*x
p0 = [2.3]
fit_S3 = curve_fit(model, to_fit_x, to_fit_y, p0)
S3_fit = fit_S3.param[1]
Error_S3_fit = stderror(fit_S3)[1]
y_fit_S3 = (model(l_struct,fit_S3.param))

eps_S3 = 5/4*S3_fit
err_eps_S3 = 5/4*Error_S3_fit


to_fit_y = abs.(δ_struct[head_fit:tail_fit,3])
to_fit_x = l_struct[head_fit:tail_fit]
@. model(x, p) = p[1]*x^p[2]
p0 = [1.6, 1.2]
fit_S4 = curve_fit(model, to_fit_x, to_fit_y, p0)
S4_fit = fit_S4.param[2]
Error_S4_fit = stderror(fit_S4)[2]
y_fit_S4 = (model(l_struct,fit_S4.param))



# Flateness computation   ======================================================================

flat = δ_struct[:,3] ./ δ_struct[:,1].^2


# ==============================================================================================








#  =============================================================================================
# Plots  =======================================================================================
#  =============================================================================================

# First time plot of the signal ================================================================

tail =  Int(floor(N/100)) #10000
plot(t[1:tail] , v[1:tail] , lw = 0.5, legend = false)
xaxis!("Time [s]")
yaxis!("Veloctity [m/s]")
save_figs("Plots/","First_plot_signal_small")


# Transformed in x coord with taylor frozen flow ===============================================

tail = Int(floor(N/100)) #10000
plot(x[1:tail] , v[1:tail] ,lw = 0.5 , legend= false)
xaxis!("X [m]")
yaxis!("Veloctity [m/s]")
save_figs("Plots/","First_plot_signal_X_small")

# Correlation length plots, first with 1/e second on semilogscale===============================

tail = 3000
plot(x[1:tail] , abs.(Correlation[1:tail]),  legend= false, lw = 2)
hline!([1/ℯ], line =  :dot, lw = 2)
annotate!((.05, 0.4, Plots.text("1/e", color = :blue, 16)))
annotate!((0.5, 0.5, Plots.text(L"$ Lc \approx  $ "*"$L_c_graph",  16)))
xaxis!("l [m]")
yaxis!("C(l) []")
save_figs("Plots/","C_l",true)

tail = 3100
plot(x[1:tail] , abs.(Correlation[1:tail]) ,lw = 2 , label = "Numerical results")
plot!(x[1:tail] , 0.9.*ℯ.^(.-x[1:tail] ./ L_c_acc).+0.001 ,lw = 2 , line = :dash , label = "Exponential fit", color = :black)
plot!(x[head].*ones(11), 0.001:0.1:1.1  , label = "fitting window", line = :dot, color = :blue, lw = 1.5)
plot!(x[tail_fit].*ones(11), 0.001:0.1:1.1  , label = nothing, line = :dot, color = :blue, lw = 1.5)
annotate!((0.7, 0.4, Plots.text(L"$ Lc $ "*" = $L_c_acc_graph"*L"\pm"*"$(Error_L_acc/2)",  16)))
xaxis!("l [m]")
yaxis!("C(l) []", :log10)
save_figs("Plots/","C_l_acc",true)

# Plot of the Energy spectrum =================================================================

plot( k[5:end], E[5:end-1] ,lw = 0.5 , legend = false ,xscale = :log10, yscale = :log10)
xaxis!("k [1/m]")
yaxis!("E(k) [J]")
save_figs("Plots/","E_spec_rough")

tail =length(E_red)-20
plot(k_red[5:tail], abs.(E_red)[5:tail],xscale = :log10, yscale = :log10,lw =1 , label = "Numerical Data", minorticks = 10)
plot!(k_red_2[2:5500],0.65.*(k_pow[2:5500]), lw =2 ,line = :dash, label = L"$ f(k) \propto k^{-\frac53}$", color = :black)
annotate!((2pi/L_c_acc*0.8, 1e-12, Plots.text(L"$  \frac2 L \frac \pi c  $ ",  16, color = :blue)))
annotate!((2pi/λ*0.8, 1e-12, Plots.text(L"$ 2 \frac \pi  \lambda $ ",  16, color = :blue)))
plot!(2pi/L_c_acc.*ones(1000), 1e-12:0.001:1 , label = nothing,line = :dot, color = :blue, lw = 1.2)
plot!(2pi/λ.*ones(1000), 1e-12:0.001:1  , label = nothing, line = :dot, color = :blue, lw = 1.2)
xaxis!("k [1/m]")
yaxis!("E(k) [J]")
save_figs("Plots/","E_spec_smooth",true)

#  delta_v  ====================================================================================

tail = 100000
plot(x[1:tail] , δv[4,1:tail]   ,lw = 1 , label = "l = $(l[4]) m")
plot!(x[1:tail] , δv[3,1:tail]   ,lw = 1 , label = "l = $(l[3]) m")
plot!(x[1:tail] , δv[2,1:tail]   ,lw = 1 , label = "l = $(l[2]) m ")
plot!(x[1:tail] , δv[1,1:tail]   ,lw = 1 , label = "l = $(l[1]) m")
xaxis!("x [m]")
yaxis!(L"$\delta_{v_{||}} (x,l)$ [m/s]")
save_figs("Plots/","delta_v_large")

tail = 10000
pyplot()
plot_1 = plot(x[1:tail] / l[4] , δv[4,1:tail] ./ fit_4.param[3]  ,lw = 0.3 , label = "l = $(l[4]) m", titlefontsize = 1, legendfontsize = 8, guidefont = 8, tickfont = 8)
plot_2 = plot(x[1:tail] / l[3]  , δv[3,1:tail]  ./ fit_3.param[3] ,lw = 0.3 , label = "l = $(l[3]) m", titlefontsize = 1, legendfontsize = 8, guidefont = 12, tickfont = 8)
plot_3 = plot(x[1:tail] / l[2] , δv[2,1:tail]  ./ fit_2.param[3] ,lw = 0.3 , label = "l = $(l[2]) m ", titlefontsize = 1, legendfontsize = 8, guidefont = 14, tickfont = 8)
yaxis!(L"$ \frac{\delta_{v_{||}} (x,l) }{ var (\delta_{v_{||}})} } $ ")
plot_4 = plot(x[1:tail] / l[1] , δv[1,1:tail]  ./ fit_1.param[3] ,lw = 0.3 , label = "l = $(l[1]) m", titlefontsize = 1, legendfontsize = 8, guidefont = 14, tickfont = 8)
xaxis!(L"x / $l_i$ ")
p = plot(plot_1 , plot_2 ,plot_3 ,plot_4 , layout = (4, 1) )
save_figs("Plots/","delta_v_scaling")
pgfplotsx()

# Pdf histogramm of delta_v  ===================================================================

histogram(δv[4,1:end-200000] , label = "l = $(l[4]) m", bins = 100 , normalize = :pdf, lw = 2, fill = true, fillalpha = 0.7)
histogram!(δv[3,1:end-20000] , label = "l = $(l[3]) m", bins = 100 , normalize = :pdf, lw = 2, fill = true, fillalpha = 0.6)
histogram!(δv[2,1:end-2000] ,label = "l = $(l[2]) m", bins = 100 ,  normalize = :pdf, lw = 2, fill = true, fillalpha = 0.5)
histogram!(δv[1,1:end-200] , label = "l = $(l[1]) m", bins = 20  , normalize = :pdf, lw = 2, fill = true, fillalpha = 0.4)
xaxis!(L"$\delta_{v_{||}} (l)$ [m/s]",(-5,5))
yaxis!("PDF")
save_figs("Plots/","hist_4")

histogram(δv[4,1:end-200000] , label = "Numerical data", bins = 1000 , normalize = :pdf, lw = 2, fill = true, fillalpha = 0.7)
plot!(vec(xfit_4) , vec(yfit_4), lw = 2 , line = :dash , color = :black , label = "Gaussian fit : "*L"\mu="*"$(round(fit_4.param[2] , digits = 3 ) ) ,"*L"\sigma="*"$(round(fit_4.param[3] , digits = 3 ) )" )
xaxis!(L"$\delta_{v_{||}} (l)$ [m/s]",(-7,7))
yaxis!("PDF" , (-0.01 , 0.27))
save_figs("Plots/","hist_L_10")

histogram(δv[3,1:end-20000] , label = "Numerical data", bins = 1000 , normalize = :pdf, lw = 2, fill = true, fillalpha = 0.7, legend = :topleft)
plot!(vec(xfit_3) , vec(yfit_3), lw = 2 , line = :dash , color = :black ,label = "Gaussian fit : "*L"\mu="*"$(round(fit_3.param[2] , digits = 3 ) ) ,"*L"\sigma="*"$(round(fit_3.param[3] , digits = 3 ) )" )
xaxis!(L"$\delta_{v_{||}} (l)$ [m/s]",(-5, 5))
yaxis!("PDF", (-0.01 , 0.55 ))
save_figs("Plots/","hist_L_1")

histogram(δv[2,1:end-2000] , label = "Numerical data", bins = 1000 , normalize = :pdf, lw = 2, fill = true, fillalpha = 0.7)
plot!(vec(xfit_2) , vec(yfit_2), lw = 2 , line = :dash , color = :black , label = "Gaussian fit : "*L"\mu="*"$(round(fit_2.param[2] , digits = 3 ) ) ,"*L"\sigma="*"$(round(fit_2.param[3] , digits = 3 ) )"  )
xaxis!(L"$\delta_{v_{||}} (l)$ [m/s]",(-2.5, 2.55))
yaxis!("PDF",(-0.01 , 1.35))
save_figs("Plots/","hist_L_01")

histogram(δv[1,1:end-200] , label = "Numerical data", bins = 1000 , normalize = :pdf, lw = 2, fill = true, fillalpha = 0.7)
plot!(vec(xfit_1) , vec(yfit_1), lw = 2 , line = :dash , color = :black ,label = "Gaussian fit : "*L"\mu="*"$(round(fit_1.param[2] , digits = 3 ) ) ,"*L"\sigma="*"$(round(fit_1.param[3] , digits = 3 ) )" )
xaxis!(L"$\delta_{v_{||}} (l)$ [m/s]", (-0.25, 0.25))
yaxis!("PDF",(-0.01 , 17))
save_figs("Plots/","hist_L_0005")

# power law fit comparison
histogram(δv[2,1:end-2000] , label = "Numerical data", bins = 1000 , normalize = :pdf, lw = 2, fill = true, fillalpha = 0.7, legend = :topleft)
plot!(vec(xfit_2) , vec(yfit_2), lw = 2 , line = :dash , color = :blue , label = "Gaussian fit , "*"Residual = $sum_rees_gau"  )
plot!(vec(xfit_2) , vec(y_fit_power_2), lw = 2 , line = :dot , color = :red , label =  L"f(x) \propto 1/x^2 , "*" Residual = $sum_rees_2"  )
plot!(vec(xfit_2) , vec(y_fit_power_2_4), lw = 2 , line = :dot , color = :green , label = L"f(x) \propto 1/x^3 , "*" Residual = $sum_rees_3"  )
xaxis!(L"$\delta_{v_{||}} (l)$ [m/s]",(-2.5, 2.5))
yaxis!("PDF",(-0.01 , 1.8))
save_figs("Plots/","hist_L_01_power_large")

histogram(δv[2,1:end-2000] , label = "Numerical data", bins = 1000 , normalize = :pdf, lw = 2, fill = true, fillalpha = 0.7, legend = :topleft)
plot!(vec(xfit_2) , vec(yfit_2), lw = 2 , line = :dash , color = :blue , label = "Gaussian fit " )
plot!(vec(xfit_2) , vec(y_fit_power_2), lw = 2 , line = :dot , color = :red , label =  L"f(x) \propto 1/x^2"  )
plot!(vec(xfit_2) , vec(y_fit_power_2_4), lw = 2 , line = :dot , color = :green , label = L"f(x) \propto 1/x^3"  )
xaxis!(L"$\delta_{v_{||}} (l)$ [m/s]",(-2.5, -0.8))
yaxis!("PDF",(-0.01 , 0.5))
save_figs("Plots/","hist_L_01_power_small")


# Structure function plots =====================================================================
head = 17
tail = 250
plot( l_struct[head:tail], abs.(δ_struct[head:tail,1]) ,lw = 2 , label = L"$S_2 \rm [m^2/s^2]$", yscale = :log10, xscale = :log10, legend = :bottomright)
plot!( l_struct[head:tail],  abs.(δ_struct[head:tail,2]) ,lw = 2 , label = L"$S_3 \rm [m^3/s^3]$", yscale = :log10, xscale = :log10)
plot!( l_struct[head:tail],  abs.(δ_struct[head:tail,3]) ,lw = 2 , label = L"$S_4 \rm [m^4/s^4]$", yscale = :log10, xscale = :log10)
xaxis!("l[m]")
save_figs("Plots/","struct_ftc_compariason")

head = 17
tail = 250
plot( l_struct[head:tail], abs.(δ_struct[head:tail,1]) ,lw = 2 , label = "Numerical data", yscale = :log10, xscale = :log10, legend = :bottomright)
plot!(l_struct[30:200] ,4.5 .*l_2_pow[30:200] , lw = 1.5 , line = :dash , label = L"$ f(l) \propto l^{\frac23}$", color = :black)
annotate!((L_c_acc*0.8, 1e-2, Plots.text(L"$  Lc  $ ",  16, color = :blue)))
annotate!((λ*0.8, 1e-2, Plots.text(L"$ \lambda $ ",  16, color = :blue)))
plot!(L_c_acc.*ones(1000), 1e-2:0.01:10 , label = nothing,line = :dot, color = :blue, lw = 1.2)
plot!(λ.*ones(1000), 1e-2:0.01:10  , label = nothing, line = :dot, color = :blue, lw = 1.2)
xaxis!("l[m]")
yaxis!(L"$\rm S_2  [m^2/s^2]$")
save_figs("Plots/","S2", true)

head = 17
tail = 250
plot( l_struct[head:tail],  abs.(δ_struct[head:tail,2]) ,lw = 2 , label = "Numerical data", yscale = :log10, xscale = :log10, legend = :bottomright)
plot!(l_struct[30:200] ,l_3_pow[30:200] , lw = 1.5 , line = :dash , label = L"$ f(l) =\frac45 \epsilon l$", color = :black)
annotate!((L_c_acc*0.8, 1e-3, Plots.text(L"$  Lc  $ ",  16, color = :blue)))
annotate!((λ*0.8, 1e-3, Plots.text(L"$ \lambda $ ",  16, color = :blue)))
plot!(L_c_acc.*ones(500), 1e-3:0.01:5 , label = nothing,line = :dot, color = :blue, lw = 1.2)
plot!(λ.*ones(500), 1e-3:0.01:5  , label = nothing, line = :dot, color = :blue, lw = 1.2)
xaxis!("l[m]")
yaxis!(L"$\rm S_3  [m^3/s^3]$")
save_figs("Plots/","S3",true)

head = 17
tail = 250
plot( l_struct[head:tail],  abs.(δ_struct[head:tail,3]) ,lw = 2 , label = "Numerical data", yscale = :log10, xscale = :log10, legend = :bottomright)
plot!(l_struct[30:200] ,360 .*l_4_pow[30:200]*0.23 , lw = 1.5 , line = :dash , label = L"$ f(l) \propto l^{\frac43}$", color = :black)
annotate!((L_c_acc*0.8, 1e-3, Plots.text(L"$  Lc  $ ",  16, color = :blue)))
annotate!((λ*0.8, 1e-3, Plots.text(L"$ \lambda $ ",  16, color = :blue)))
plot!(L_c_acc.*ones(5000), 1e-3:0.1:500 , label = nothing,line = :dot, color = :blue, lw = 1.2)
plot!(λ.*ones(5000), 1e-3:0.1:500  , label = nothing, line = :dot, color = :blue, lw = 1.2)
xaxis!("l[m]")
yaxis!(L"$\rm S_4  [m^4/s^4]$")
save_figs("Plots/","S4",true)


# fits of the diffferenet needeed values.
tail =length(E_red)-20
plot(k_red[5:tail], abs.(E_red)[5:tail],xscale = :log10, yscale = :log10,lw =1 , label = "Numerical Data", legend = :bottomleft) # , minorticks = 10
plot!(k_red[100:65000], abs.(y_fit_E[100:65000]), lw =2 ,line = :dash, label = "exponential fit , b = $(round(E_fit,digits=3))", color = :black)
annotate!((2pi/L_c_acc*0.8, 1e-12, Plots.text(L"$  \frac2 L \frac \pi c  $ ",  16, color = :blue)))
annotate!((2pi/λ*0.8, 1e-12, Plots.text(L"$ 2 \frac \pi  \lambda $ ",  16, color = :blue)))
plot!(2pi/L_c_acc.*ones(1000), 1e-12:0.001:1 , label = nothing,line = :dot, color = :blue, lw = 1.2)
plot!(2pi/λ.*ones(1000), 1e-12:0.001:1  , label = nothing, line = :dot, color = :blue, lw = 1.2)
xaxis!("k [1/m]")
yaxis!("E(k) [J]")
save_figs("Plots/","Ee_fit",true)

head = 17
tail = 250
plot( l_struct[head:tail], abs.(δ_struct[head:tail,1]) ,lw = 2 , label = "Numerical data", yscale = :log10, xscale = :log10, legend = :topright)
plot!(l_struct[50:200] ,abs.(y_fit_S2[50:200]) , lw = 1.5 , line = :dash , label = "exponential fit , a = $(round(eps_fit,digits=3)), b = $(round(S2_fit,digits=3)) ", color = :black)
annotate!((L_c_acc*0.8, 1e-2, Plots.text(L"$  Lc  $ ",  16, color = :blue)))
annotate!((λ*0.8, 1e-2, Plots.text(L"$ \lambda $ ",  16, color = :blue)))
plot!(L_c_acc.*ones(1000), 1e-2:0.01:10 , label = nothing,line = :dot, color = :blue, lw = 1.2)
plot!(λ.*ones(1000), 1e-2:0.01:10  , label = nothing, line = :dot, color = :blue, lw = 1.2)
xaxis!("l[m]")
yaxis!(L"$\rm S_2  [m^2/s^2]$")
save_figs("Plots/","S2_fit", true)

head = 17
tail = 250
plot( l_struct[head:tail],  abs.(δ_struct[head:tail,2]) ,lw = 2 , label = "Numerical data", yscale = :log10, xscale = :log10, legend = :bottomright)
plot!(l_struct[30:200] ,abs.(y_fit_S3[30:200]) , lw = 1.5 , line = :dash , label = "linear fit , a = $(round(S3_fit,digits=3))", color = :black)
annotate!((L_c_acc*0.8, 1e-3, Plots.text(L"$  Lc  $ ",  16, color = :blue)))
annotate!((λ*0.8, 1e-3, Plots.text(L"$ \lambda $ ",  16, color = :blue)))
plot!(L_c_acc.*ones(500), 1e-3:0.01:5 , label = nothing,line = :dot, color = :blue, lw = 1.2)
plot!(λ.*ones(500), 1e-3:0.01:5  , label = nothing, line = :dot, color = :blue, lw = 1.2)
xaxis!("l[m]")
yaxis!(L"$\rm S_3  [m^3/s^3]$")
save_figs("Plots/","S3_fit",true)

head = 17
tail = 250
plot( l_struct[head:tail],  abs.(δ_struct[head:tail,3]) ,lw = 2 , label = "Numerical data", yscale = :log10, xscale = :log10, legend = :bottomright)
plot!(l_struct[30:200] ,abs.(y_fit_S4[30:200]) , lw = 1.5 , line = :dash ,label = "exponential fit , b = $(round(S4_fit,digits=3))", color = :black)
annotate!((L_c_acc*0.8, 1e-3, Plots.text(L"$  Lc  $ ",  16, color = :blue)))
annotate!((λ*0.8, 1e-3, Plots.text(L"$ \lambda $ ",  16, color = :blue)))
plot!(L_c_acc.*ones(5000), 1e-3:0.1:500 , label = nothing,line = :dot, color = :blue, lw = 1.2)
plot!(λ.*ones(5000), 1e-3:0.1:500  , label = nothing, line = :dot, color = :blue, lw = 1.2)
xaxis!("l[m]")
yaxis!(L"$\rm S_4  [m^4/s^4]$")
save_figs("Plots/","S4_fit",true)




# Flantness plot =============================================================================
head = 17
tail = 300
plot( l_struct[head:tail], abs.(flat[head:tail]) ,lw = 2 , legend = false, xscale = :log10)
annotate!((L_c_acc*1.6, 8, Plots.text(L"$  Lc  $ ",  14, color = :blue)))
annotate!((λ*1.6, 8, Plots.text(L"$ \lambda $ ",  14, color = :blue)))
plot!(L_c_acc.*ones(51), 3:0.1:8 , label = nothing,line = :dot, color = :blue, lw = 1.2)
plot!(λ.*ones(51), 3:0.1:8  , label = nothing, line = :dot, color = :blue, lw = 1.2)
xaxis!("l[m]")
yaxis!("flatness   f(l)")
save_figs("Plots/","Flatness",true)
