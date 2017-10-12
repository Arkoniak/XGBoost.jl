module CrossValidation

export unique_inverse,
    CrossValGenerator,
    Kfold,
    StratifiedKfold

# Taken from https://github.com/JuliaStats/MLBase.jl/blob/master/src/crossval.jl

## return the unique values of A and a vector of vectors of indexes to reconstruct
## the original array

function unique_inverse{T}(A::Vector{T})
    out = Vector{T}(0)
    out_idx = Vector{Vector{Int}}(0)
    seen = Dict{T, Int}()
    for (idx, x) in enumerate(A)
        if !in(x, keys(seen))
            seen[x] = length(seen) + 1
            push!(out, x)
            push!(out_idx, Int[])
        end
        push!(out_idx[seen[x]], idx)
    end
    out, out_idx
end

#
# a = [:a, :a, :b, :c, :b, :a]
# ui = unique_inverse(a)
#

## cross validation generators

abstract type CrossValGenerator end

# K-fold

struct Kfold <: CrossValGenerator
    permseq::Vector{Int}
    k::Int
    n::Int
    coeff::Float64

    function Kfold(n::Int, k::Int, seed = nothing)
        2 <= k <= n || error("The value of k must be in [2, length(a)].")
        if !isa(seed, Void)
            srand(seed)
        end
        new(randperm(n), k, n, n / k)
    end
end

Base.length(c::Kfold) = c.k

struct KfoldState
    i::Int      # the i-th of the subset
    s::Int      # starting index
    e::Int      # ending index
end

Base.start(c::Kfold) = KfoldState(1, 1, round.(Integer,c.coeff))
function Base.next(c::Kfold, s::KfoldState)
    i = s.i+1

    return [c.permseq[s.s:s.e], setdiff(1:c.n, c.permseq[s.s:s.e])],
        KfoldState(i, s.e+1, round.(Integer,c.coeff * i))
end
Base.done(c::Kfold, s::KfoldState) = (s.i > c.k)

# Stratified K-fold

struct StratifiedKfold <: CrossValGenerator
    permseqs::Vector{Vector{Int}}  #Vectors of vectors of indexes for each stratum
    n::Int                         #Total number of observations
    k::Int                         #Number of splits
    coeffs::Vector{Float64}        #About how many observations per strata are in a val set

    function StratifiedKfold(strata, k, seed = nothing)
        2 <= k <= length(strata) || error("The value of k must be in [2, length(strata)].")
        if !isa(seed, Void)
            srand(seed)
        end
        strata_labels, permseqs = unique_inverse(strata)
        map(shuffle!, permseqs)
        coeffs = Float64[]
        for (stratum, permseq) in zip(strata_labels, permseqs)
            k <= length(permseq) || error("k is greater than the length of stratum $stratum")
            push!(coeffs, length(permseq) / k)
        end
        new(permseqs, length(strata), k, coeffs)
    end
end

Base.length(c::StratifiedKfold) = c.k

Base.start(c::StratifiedKfold) = 1
function Base.next(c::StratifiedKfold, s::Int)
    r = Int[]
    for (permseq, coeff) in zip(c.permseqs, c.coeffs)
        a, b = round.(Integer, [s-1, s] .* coeff)
        append!(r, view(permseq, a+1:b))
    end

    return [r, setdiff(1:c.n, r)], s+1
end
Base.done(c::StratifiedKfold, s::Int) = (s > c.k)

end
