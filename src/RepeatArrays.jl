module RepeatArrays

export myrepeat

import Base.size

struct RepeatArray{T, N}
    A::Array{T, N}

    # Array of tuples (dim, inner, outer), giving a number of inner and outer repetitions
    # and the dimension to repeat along.
    #
    # An inner repetition repeats values one at a time. For example, an inner repetition
    # of 2 turns [a, b, c] into [a, a, b, b, c, c].
    #
    # An outer repetition repeats the whole array. For example, an outer repetition of
    # 2 turns [a, b, c] into [a, b, c, a, b, c].
    #
    # This is guaranteed to be "complete" and sorted by dimension.
    # By complete we mean that it contains elements (dim, 1, 1) as needed
    # to fill in entries until the maximum value of dim.
    repetitions::Array{Tuple{Int, Int, Int}}

    maxdim::Int
end

Base.eltype(R::RepeatArray{T, N}) where {T, N} = T

Base.similar(R::RepeatArray{T, N}, eltype) where {T, N} = RepeatArray(similar(R.A, eltype), copy(R.repetitions), R.maxdim)

myrepeat(A::Array{T, N}; inner=[], outer=[], dims=nothing) where {T, N} = begin
    inner = [inner...]
    outer = [outer...]
    if !isnothing(dims)
        length(dims) == length(unique(dims)) || throw(ArgumentError("Repeated dimension specifier"))
        length(inner) == length(outer) || throw(ArgumentError("If dimensions are specified, inner and outer specifications must match in length. Use 1 to indicate no repetition."))
        length(dims) == length(inner) || throw(ArgumentError("Number of dimension specifiers must match number of repetition specifiers"))
    end

    if length(inner) < length(outer)
        append!(inner, ones(length(outer) - length(inner)))
    end
    if length(outer) < length(inner)
        append!(outer, ones(length(inner) - length(outer)))
    end

    dims = if isnothing(dims)
        1:max(length(inner), length(outer))
    else
        dims
    end

    maxdim = max(maximum(dims), length(size(A)))
    repetitions = Tuple{Int, Int, Int}[]

    for d in 1:maxdim
        if d in dims
            inner_rep = inner[findfirst(x -> x == d, dims)]
            outer_rep = outer[findfirst(x -> x == d, dims)]
            push!(repetitions, (d, inner_rep, outer_rep))
        else
            push!(repetitions, (d, 1, 1))
        end
    end

    RepeatArray(A, repetitions, maxdim)
end

size(R::RepeatArray{T, N}) where {T, N} = begin
    result = Int[]
    for (dim, inner, outer) in R.repetitions
        if dim <= length(size(R.A))
            push!(result, size(R.A)[dim] * inner * outer)
        else
            push!(result, inner * outer)
        end
    end
    tuple(result...)
end

Base.collect(R::RepeatArray{T, N}) where {T, N} = begin
    result = Array{T}(undef, size(R)...)

    missing_dims = length(size(R)) - length(size(R.A))
    dims = [size(R.A)...]
    append!(dims, Array{Int}(ones(missing_dims)))

    A = reshape(R.A, dims...)

    repeat(A, inner=[rep[2] for rep in R.repetitions], outer=[rep[3] for rep in R.repetitions])
end

###########
# Broadcasting
###########

struct RepeatStyle <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:RepeatArray}) = RepeatStyle()

Base.Broadcast.broadcastable(R::RepeatArray) = R

Base.BroadcastStyle(::RepeatStyle, ::Broadcast.DefaultArrayStyle{0}) = RepeatStyle()

Base.BroadcastStyle(::RepeatStyle, ::Broadcast.DefaultArrayStyle{N}) where {N} = Broadcast.DefaultArrayStyle{N}()

Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(*), x::Number, R::RepeatArray{T, N}) where {T, N} = RepeatArray(x .* R.A, R.repetitions, R.maxdim)
Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(*), R::RepeatArray{T, N}, x::Number) where {T, N} = RepeatArray(R.A .* x, R.repetitions, R.maxdim)

Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(/), x::Number, R::RepeatArray{T, N}) where {T, N} = RepeatArray(R.A ./ x, R.repetitions, R.maxdim)
Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(/), R::RepeatArray{T, N}, x::Number) where {T, N} = RepeatArray(x ./ R.A, R.repetitions, R.maxdim)

Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, R::RepeatArray{T, N}, v::Base.RefValue{Val{x}}) where {T, N, x} =
    RepeatArray(Base.literal_pow.(^, R.A, v), R.repetitions, R.maxdim)
Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(^), R::RepeatArray{T, N}, x::Number) where {T, N} = RepeatArray(R.A .^ x, R.repetitions, R.maxdim)

Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(+), R::RepeatArray{T1, N1}, S::RepeatArray{T2, N2}) where {T1, N1, T2, N2} = begin
    size1 = size(R)
    size2 = size(S)
    output_size = Base.Broadcast._bcs(size1, size2)

    outer_reps = Int[]
    for i in 1:length(output_size)
        reps1 = get(R.repetitions, i, (undef, undef, 1))
        reps2 = get(R.repetitions, i, (undef, undef, 1))
        push!(outer_reps, gcd(reps1[3], reps2[3]))
    end

    inner_reps = Int[]

    R_expansions = Int[]
    R_elongations = Int[]
    S_expansions = Int[]
    S_elongations = Int[]

    for i in 1:length(output_size)
        chunk_size = Int(output_size[i] / outer_reps[i])

        reps1 = get(R.repetitions, i, (undef, 1, undef))
        inner_reps1 = reps1[2]
        reps2 = get(S.repetitions, i, (undef, 1, undef))
        inner_reps2 = reps2[2]

        base_dim = lcm(size(R.A, i), size(S.A, i))

        inner_rep = chunk_size / base_dim
        push!(inner_reps, inner_rep)

        push!(R_expansions, Int(inner_reps1 / inner_rep))
        push!(R_elongations, Int(chunk_size / (size(R.A, i) * inner_reps1)))
        push!(S_expansions, Int(inner_reps2 / inner_rep))
        push!(S_elongations, Int(chunk_size / (size(S.A, i) * inner_reps2)))
    end

    A = repeat(R.A, inner=R_expansions, outer=R_elongations) .+ repeat(S.A, inner=S_expansions, outer=S_elongations)
    myrepeat(A, inner=inner_reps, outer=outer_reps)
end

end
