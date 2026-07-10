abstract type Layer{DT} end

struct Linear{DT} <: Layer{DT}
    weights::Matrix{Variable{DT}}
    bias::Vector{Variable{DT}}

    function Linear(m::Int, n::Int)
        limit = sqrt(6.0 / (m + n))
        weights = [Variable(2 * limit * rand() - limit) for _ in 1:n, _ in 1:m]
        bias = [Variable(0.0) for _ in 1:n]
        new{Float64}(weights, bias)
    end
end

function parameters(layer::Linear)
    par = Variable[]
    for weight in layer.weights
        push!(par, weight)
    end
    for b in layer.bias
        push!(par, b)
    end
    par
end

struct Tanh{DT} <: Layer{DT}
    parameter::Vector{Variable{DT}}

    function Tanh{DT}() where DT
        new{DT}(Variable{DT}[])
    end
end

function parameters(::Tanh)
    Variable[]
end

Tanh(::Type{DT}=Float64) where DT = Tanh{DT}()

struct ReLU{DT} <: Layer{DT}
    parameter::Vector{Variable{DT}}
    function ReLU{DT}() where DT
        new{DT}(Variable{DT}[])
    end
end
parameters(::ReLU) = Variable[]
ReLU(::Type{DT}=Float64) where DT = ReLU{DT}()
forward(::ReLU, v::AbstractVector) = map(relu, v)

struct Sigmoid{DT} <: Layer{DT}
    parameter::Vector{Variable{DT}}
    function Sigmoid{DT}() where DT
        new{DT}(Variable{DT}[])
    end
end
parameters(::Sigmoid) = Variable[]
Sigmoid(::Type{DT}=Float64) where DT = Sigmoid{DT}()
forward(::Sigmoid, v::AbstractVector) = map(sigmoid, v)

struct Softplus{DT} <: Layer{DT}
    parameter::Vector{Variable{DT}}
    function Softplus{DT}() where DT
        new{DT}(Variable{DT}[])
    end
end
parameters(::Softplus) = Variable[]
Softplus(::Type{DT}=Float64) where DT = Softplus{DT}()
forward(::Softplus, v::AbstractVector) = map(softplus, v)

function forward(layer::Linear{DT}, v::AbstractVector) where DT
    @assert size(layer.weights, 2) == length(v) "Input length $(length(v)) does not match layer input size $(size(layer.weights, 2))"
    layer.weights * v + layer.bias
end

function forward(::Tanh, v::AbstractVector)
    map(tanh, v)
end

struct Model{DT}
    layers::Vector{Layer{DT}}
end

function parameters(model::Model)
    vcat([parameters(layer) for layer in model.layers]...)
end

function forward(model::Model, v::AbstractVector)
    for layer in model.layers
        v = forward(layer, v)
    end
    return v
end
