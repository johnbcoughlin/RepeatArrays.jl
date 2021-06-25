module RepeatArrays

export myrepeat

import Base.size

struct RepeatArray{T, InnerType}
    A::InnerType

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

Base.eltype(R::RepeatArray{T, InnerType}) where {T, InnerType} = T

Base.similar(R::RepeatArray{T, InnerType}, eltype) where {T, InnerType} = RepeatArray(similar(R.A, eltype), copy(R.repetitions), R.maxdim)

myrepeat(R::RepeatArray{T, InnerType}; inner=[], outer=[], dims=nothing) where {T, InnerType} = begin
    repetitions, maxdim = compute_repetitions(ndims=length(size(R)), inner=inner, outer=outer, dims=dims)
    RepeatArray{T, typeof(R)}(R, repetitions, maxdim)
end

myrepeat(A::Array{T, N}; inner=[], outer=[], dims=nothing) where {T, N} = begin
    repetitions, maxdim = compute_repetitions(ndims=N, inner=inner, outer=outer, dims=dims)
    RepeatArray{T, typeof(A)}(A, repetitions, maxdim)
end

compute_repetitions(; ndims::Int, inner=[], outer=[], dims=nothing) = begin
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

    maxdim = max(maximum(dims), ndims)
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

    [repetitions, maxdim]
end

size(R::RepeatArray{T, InnerType}) where {T, InnerType} = begin
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

size(R::RepeatArray{T, InnerType}, i::Int) where {T, InnerType} = begin
    s = size(R)
    if i > length(s)
        return 1
    else
        return s[i]
    end
end

get_outer_reps(R::RepeatArray{T, InnerType}, dim::Int; broadcasting_to=nothing) where {T, InnerType} = begin
    if dim > length(R.repetitions)
        if isnothing(broadcasting_to)
            return 1
        else
            return broadcasting_to
        end
    else
        return R.repetitions[dim][3]
    end
end

get_inner_reps(R::RepeatArray{T, InnerType}, dim::Int) where {T, InnerType} = begin
    if dim > length(R.repetitions)
        return 1
    else
        return R.repetitions[dim][2]
    end
end

Base.collect(R::RepeatArray{T, Array{T, N}}) where {T, N} = collect_by_repeating(R, R.A)

Base.collect(R::RepeatArray{T, RepeatArray{T, InnerType}}) where {T, InnerType} = collect_by_repeating(R, collect(R.A))

collect_by_repeating(R::RepeatArray{T, InnerType}, A::Array{T, N}) where {T, InnerType, N} = begin
    missing_dims = length(size(R)) - length(size(A))
    dims = [size(A)...]
    append!(dims, Array{Int}(ones(missing_dims)))

    reshaped = reshape(A, dims...)

    repeat(reshaped, inner=[rep[2] for rep in R.repetitions], outer=[rep[3] for rep in R.repetitions])
end

###########
# Broadcasting
###########

struct RepeatStyle <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:RepeatArray}) = RepeatStyle()

Base.Broadcast.broadcastable(R::RepeatArray) = R

Base.BroadcastStyle(::RepeatStyle, ::Broadcast.DefaultArrayStyle{0}) = RepeatStyle()

Base.BroadcastStyle(::RepeatStyle, ::Broadcast.DefaultArrayStyle{N}) where {N} = Broadcast.DefaultArrayStyle{N}()

#########
# Unary operators
#########

Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(*), x::Number, R::RepeatArray{T, InnerType}) where {T, InnerType} = RepeatArray(x .* R.A, R.repetitions, R.maxdim)
Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(*), R::RepeatArray{T, InnerType}, x::Number) where {T, InnerType} = RepeatArray(R.A .* x, R.repetitions, R.maxdim)

Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(/), x::Number, R::RepeatArray{T, InnerType}) where {T, InnerType} = RepeatArray(R.A ./ x, R.repetitions, R.maxdim)
Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(/), R::RepeatArray{T, InnerType}, x::Number) where {T, InnerType} = RepeatArray(x ./ R.A, R.repetitions, R.maxdim)

Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, R::RepeatArray{T, InnerType}, v::Base.RefValue{Val{x}}) where {T, InnerType, x} =
    RepeatArray(Base.literal_pow.(^, R.A, v), R.repetitions, R.maxdim)
Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(^), R::RepeatArray{T, InnerType}, x::Number) where {T, InnerType} = RepeatArray(R.A .^ x, R.repetitions, R.maxdim)

#########
# Broadcasting between RepeatArrays
#########

struct BinaryBroadcastPlan
    outer_reps::Array{Int}
    inner_reps::Array{Int}
    arg1Expansions::Array{Int}
    arg1Elongations::Array{Int}
    arg2Expansions::Array{Int}
    arg2Elongations::Array{Int}
end

planBinaryBroadcast(R::RepeatArray{T1, InnerType1}, S::RepeatArray{T2, InnerType2}) where {T1, InnerType1, T2, InnerType2} = begin
    output_size = Base.Broadcast._bcs(size(R), size(S))

    outer_reps = Int[]
    for i in 1:length(output_size)
        push!(outer_reps, gcd(get_outer_reps(R, i, broadcasting_to=output_size[i]),
                              get_outer_reps(S, i, broadcasting_to=output_size[i])))
    end

    inner_reps = Int[]

    R_expansions = Int[]
    R_elongations = Int[]
    S_expansions = Int[]
    S_elongations = Int[]


    for i in 1:length(output_size)
        chunk_size = Int(output_size[i] / outer_reps[i])

        inner_reps1 = get_inner_reps(R, i)
        inner_reps2 = get_inner_reps(S, i)

        inner_rep = gcd(inner_reps1, inner_reps2)
        base_dim = Int(chunk_size / inner_rep)

        push!(inner_reps, inner_rep)

        push!(R_expansions, Int(inner_reps1 / inner_rep))
        push!(R_elongations, Int(chunk_size / (size(R.A, i) * inner_reps1)))
        push!(S_expansions, Int(inner_reps2 / inner_rep))
        push!(S_elongations, Int(chunk_size / (size(S.A, i) * inner_reps2)))
    end

    BinaryBroadcastPlan(outer_reps, inner_reps, R_expansions, R_elongations, S_expansions, S_elongations)
end

doBinaryBroadcast(f::Any, R::RepeatArray{T1, InnerType1}, S::RepeatArray{T2, InnerType2}) where {T1, InnerType1, T2, InnerType2} = begin
    plan = planBinaryBroadcast(R, S)
    arg1 = repeat(R.A, inner=plan.arg1Expansions, outer=plan.arg1Elongations)
    arg2 = repeat(S.A, inner=plan.arg2Expansions, outer=plan.arg2Elongations)
    A = broadcast(f, arg1, arg2)
    myrepeat(A, inner=plan.inner_reps, outer=plan.outer_reps)
end

Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(+), R::RepeatArray{T1, InnerType1}, S::RepeatArray{T2, InnerType2}) where {T1, InnerType1, T2, InnerType2} = doBinaryBroadcast(+, R, S)
Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(-), R::RepeatArray{T1, InnerType1}, S::RepeatArray{T2, InnerType2}) where {T1, InnerType1, T2, InnerType2} = doBinaryBroadcast(-, R, S)
Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(*), R::RepeatArray{T1, InnerType1}, S::RepeatArray{T2, InnerType2}) where {T1, InnerType1, T2, InnerType2} = doBinaryBroadcast(*, R, S)
Base.Broadcast.broadcasted(::RepeatStyle, ::typeof(/), R::RepeatArray{T1, InnerType1}, S::RepeatArray{T2, InnerType2}) where {T1, InnerType1, T2, InnerType2} = doBinaryBroadcast(/, R, S)

end
