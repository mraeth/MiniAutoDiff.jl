module MiniAutoDiff

    using Base
    using LinearAlgebra

    include("core.jl")
    include("autodiff.jl")
    include("layers.jl")
    include("optimizers.jl")
    include("losses.jl")

    export Variable, diff
    export Layer, Linear, Tanh, ReLU, Sigmoid, Softplus, Model, forward, parameters
    export relu, sigmoid, softplus
    export Optimizer, GradientDescent, update
    export mse_loss

end
