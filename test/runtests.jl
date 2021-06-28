using RepeatArrays
using Test

@testset "RepeatArrays.jl" begin
    a = [1. 2.
         3. 4.]

    @testset "Inner repetition" begin
        @test size(myrepeat(a, inner=(2,))) == (4, 2)
        @test size(myrepeat(a, inner=(1, 2))) == (2, 4)
        @test collect(myrepeat(a, inner=(2,))) == repeat(a, inner=(2, 1))
        @test collect(myrepeat(a, inner=(1, 2))) == repeat(a, inner=(1, 2))

    end

    @testset "Outer repetition" begin
        @test size(myrepeat(a, outer=(2,))) == (4, 2)
        @test collect(myrepeat(a, outer=(2,))) == repeat(a, outer=(2, 1))
    end

    @testset "Mixed repetition" begin
        @test size(myrepeat(a, inner=(1, 3), outer=(2,))) == (4, 6)
    end

    @testset "Broadcasting" begin
        R = myrepeat([1, 2, 3], inner=(10,), outer=(2,))

        @testset "One dimension" begin
            a = myrepeat([1, 2], outer=(10,))
            b = myrepeat([.1, .2], inner=(10,))
            @test collect(a .+ b) == collect(a) .+ collect(b)

            S = myrepeat([.1, .2], inner=(5), outer=(6,))
            expected = repeat([1, 2, 3], inner=10, outer=2) .+ repeat([.1, .2], inner=5, outer=6)
            @test collect(R .+ S) == expected
        end

        @testset "Two dimensions" begin
            Q = myrepeat([1 2
                          3 4], inner=(5, 2), outer=(6, 3))
            expected = collect(R) .+ collect(Q)
            @test collect(R .+ Q) == expected
        end

        @testset "Promotion" begin
            a = myrepeat([1, 2], outer=(10,))
            b = myrepeat([1 2
                          3 4], inner=(10, 5))
            expected = collect(a) .* collect(b)
            check = myrepeat([1, 2], outer=(10, 10))
            @test collect(a .* b) == expected
            @test collect(a .* b) == collect(check) .* collect(b)

            c = myrepeat([4 5], outer=(1, 5))
            @test collect(b .* c) == collect(b) .* collect(c)
            @test collect(a .* c) == collect(a) .* collect(c)
        end
    end

    @testset "Nested" begin
        a = myrepeat([1, 2], inner=(10,))
        b = myrepeat(a, outer=(2, 3))
        @test collect(b) == repeat(repeat([1, 2], inner=10), outer=(2, 3)) == repeat(collect(a), outer=(2, 3))
        c = myrepeat(b, inner=(4, 5))
        @test collect(c) == repeat(collect(b), inner=(4, 5))

        @testset "Nested broadcasting" begin
            d = myrepeat([4, 5], outer=(5,))
            e = myrepeat(d, inner=(4, 3))
            @test collect(e .+ b) == collect(e) .+ collect(b)
        end
    end
end

nothing
