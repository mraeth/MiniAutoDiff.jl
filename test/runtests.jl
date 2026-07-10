using MiniAutoDiff
import MiniAutoDiff: diff as autodiff
using Test

function gradient_check(f, xs::Vector{Float64}; ε=1e-5)
    vars = [Variable(x) for x in xs]
    out  = f(vars...)
    autodiff(out)
    g_auto = [v.grad for v in vars]

    g_num = map(1:length(xs)) do i
        xp = copy(xs); xp[i] += ε
        xm = copy(xs); xm[i] -= ε
        (f(xp...) - f(xm...)) / (2ε)
    end

    abs_err = abs.(g_auto .- g_num)
    rel_err = abs_err ./ max.(1.0, abs.(g_auto), abs.(g_num))
    (g_auto=g_auto, g_num=g_num, abs_err=abs_err, rel_err=rel_err)
end

@testset "MiniAutoDiff.jl" begin

    @testset "Primitive ops" begin
        @test all(gradient_check((x,y) -> x + y,    [2.0, 3.0]).rel_err .≤ 1e-5)
        @test all(gradient_check((x,y) -> x - y,    [5.0, 2.0]).rel_err .≤ 1e-5)
        @test all(gradient_check((x,y) -> x * y,    [3.0, 4.0]).rel_err .≤ 1e-5)
        @test all(gradient_check((x,y) -> x / y,    [6.0, 2.0]).rel_err .≤ 1e-5)
        @test all(gradient_check(x     -> x^3,      [2.0]     ).rel_err .≤ 1e-5)
        @test all(gradient_check(x     -> exp(x),   [1.5]     ).rel_err .≤ 1e-5)
        @test all(gradient_check(x     -> log(x),   [2.0]     ).rel_err .≤ 1e-5)
        @test all(gradient_check(x     -> sqrt(x),  [4.0]     ).rel_err .≤ 1e-5)
        @test all(gradient_check(x     -> sin(x),   [1.0]     ).rel_err .≤ 1e-5)
        @test all(gradient_check(x     -> cos(x),   [1.0]     ).rel_err .≤ 1e-5)
        @test all(gradient_check(x     -> tanh(x),  [0.5]     ).rel_err .≤ 1e-5)
    end

    @testset "Composed functions" begin
        # f(a,b,c) = a * exp(b*c) + c^2 * a
        f1 = (a,b,c) -> a * exp(b*c) + c^2 * a
        @test all(gradient_check(f1, [5.0, 3.0, 2.0]).rel_err .≤ 1e-5)

        # softplus: f(x) = log(exp(x) + 1)
        f2 = x -> log(exp(x) + 1)
        @test all(gradient_check(f2, [1.0]).rel_err .≤ 1e-5)

        # f(x,y) = tanh(x*y + y)
        f3 = (x,y) -> tanh(x*y + y)
        @test all(gradient_check(f3, [0.8, 1.2]).rel_err .≤ 1e-5)
    end

    @testset "Linear regression" begin
        xs_data = [1.0, 2.0, 3.0]
        ys_data = [2.0, 4.0, 6.0]  # y = 2x

        mse_f = (wv, bv) -> begin
            preds = [wv * x + bv for x in xs_data]
            sum((p - y)^2 for (p, y) in zip(preds, ys_data)) / length(xs_data)
        end

        w = Variable(0.0); b = Variable(0.0)
        out = mse_f(w, b)
        autodiff(out)
        loss_before = out.value

        @test isfinite(w.grad)
        @test isfinite(b.grad)

        w_new = w.value - 0.01 * w.grad
        b_new = b.value - 0.01 * b.grad
        loss_after = mse_f(w_new, b_new)
        @test loss_after < loss_before

        @test all(gradient_check(mse_f, [0.5, 0.1]).rel_err .≤ 1e-5)
    end

end
