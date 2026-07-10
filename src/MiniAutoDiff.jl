module MiniAutoDiff

    using Base
    using LinearAlgebra

    include("core.jl")
    include("autodiff.jl")
    include("layers.jl")
    include("optimizers.jl")

    export Variable, diff
    export Layer, Linear, Tanh, Model, forward, parameters
    export Optimizer, GradientDescent, update

end
