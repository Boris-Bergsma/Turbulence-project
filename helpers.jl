# Please, keep this ref:
# https://pixorblog.wordpress.com/2016/07/13/savitzky-golay-filters-julia/

using LinearAlgebra
using DSP

function vandermonde(halfWindow::Int, polyDeg::Int,T::Type=Float64)

    @assert halfWindow>=0
    @assert polyDeg>=0

    x=T[i for i in -halfWindow:halfWindow]

    n = polyDeg+1
    m = length(x)

    V = zeros(m, n)

    for i = 1:m
        V[i,1] = T(1)
    end
    for j = 2:n
        for i = 1:m
            V[i,j] = x[i] * V[i,j-1]
        end
    end

    return V
end

#________________________________________________________________

function SG(halfWindow::Int, polyDeg::Int,T::Type=Float64)

    @assert 2*halfWindow>polyDeg

    V=vandermonde(halfWindow,polyDeg,T)
    Q,R=qr(V)
    SG=inv(R)*Q'

    for i in 1:size(SG,1)
        SG[i,:]*=factorial(i-1)
    end

# CAVEAT: returns the transposed matrix

    return SG'
end

#________________________________________________________________

function apply_filter(filter::StridedVector,signal::StridedVector)

    @assert isodd(length(filter))

    halfWindow = round(Int,(length(filter)-1)/2)

    padded_signal =
	    [signal[1]*ones(halfWindow);
         signal;
         signal[end]*ones(halfWindow)]

    filter_cross_signal = conv(filter[end:-1:1], padded_signal)

    return filter_cross_signal[2*halfWindow+1:end-2*halfWindow]
end


function save_figs(path ,name, svg = false)
    if svg
        savefig("$path/SVGs/$name.svg")
    end
    savefig("$path/PNGs/$name.png")
    savefig("$path/PDFs/$name.pdf")
end


function smooth_reduce(data)

    temp = floor(length(data)/10)
    new_data = zeros(Int(temp))

    for i = 1:temp-20
        new_data[Int(i)] = mean(data[Int(10i):Int(10i+10)])
    end

    return new_data
end
