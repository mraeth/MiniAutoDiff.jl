module MiniAutoDiff

    using Base
    using LinearAlgebra

    include("core.jl")
    include("autodiff.jl")
    include("layers.jl")
    include("optimizers.jl")
    include("losses.jl")
    include("draw.jl")

    
    export Variable, diff
    export Layer, Linear, Tanh, ReLU, Sigmoid, Softplus, Model, forward, parameters, draw
    export relu, sigmoid, softplus
    export Optimizer, GradientDescent, update
    export mse_loss

end
