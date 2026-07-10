mutable struct Variable{DT<:Number} <: Number
    value::DT
    parents::Vector{Tuple{Variable{DT}, DT}}
    grad::DT

    function Variable(value::DT) where DT<:Number
        new{DT}(value, Tuple{Variable{DT}, DT}[], zero(DT))
    end

end

Base.zero(::Type{Variable{DT}}) where DT<:Number = Variable(zero(DT))
Base.zero(x::Variable) = zero(typeof(x))
Base.one(::Type{Variable{DT}}) where DT<:Number = Variable(one(DT))
Base.one(x::Variable) = one(typeof(x))

# Promote plain numbers to Variable{DT} so generic Number methods compose.
# Converting a constant produces a leaf Variable with no parents (grad stays zero).
Base.promote_rule(::Type{Variable{DT}}, ::Type{<:Number}) where DT<:Number = Variable{DT}
Base.convert(::Type{Variable{DT}}, x::Number) where DT<:Number = Variable(DT(x))
Base.convert(::Type{Variable{DT}}, x::Variable{DT}) where DT<:Number = x

function Base.:+(a::Variable{DT}, b::Variable{DT}) where DT<:Number
    out = Variable(a.value + b.value)
    push!(out.parents, (a, one(DT)))
    push!(out.parents, (b, one(DT)))
    return out
end

function Base.:+(a::Variable{DT}, b::Number) where DT<:Number
    out = Variable(DT(b) + a.value)
    push!(out.parents, (a, one(DT)))
    return out
end

function Base.:+(a::Number, b::Variable{DT}) where DT<:Number
    return b+a
end

function Base.:*(a::Variable{DT}, b::Variable{DT}) where DT<:Number
    out = Variable(a.value * b.value)
    push!(out.parents, (a, b.value))
    push!(out.parents, (b, a.value))
    return out
end

function Base.:*(a::Variable{DT}, b::Number) where DT<:Number
    out = Variable(a.value * DT(b))
    push!(out.parents, (a, DT(b)))
    return out
end

function Base.:*(a::Number, b::Variable{DT}) where DT<:Number
    return b * a
end

function Base.:-(a::Variable{DT}, b::Variable{DT}) where DT<:Number
    out = Variable(a.value - b.value)
    push!(out.parents, (a, one(DT)))
    push!(out.parents, (b, -one(DT)))
    return out
end

function Base.:-(a::Variable{DT}, b::Number) where DT<:Number
    out = Variable(a.value - DT(b))
    push!(out.parents, (a, one(DT)))
    return out
end

function Base.:-(a::Number, b::Variable{DT}) where DT<:Number
    out = Variable(DT(a) - b.value)
    push!(out.parents, (b, -one(DT)))
    return out
end

function Base.:-(a::Variable{DT}) where DT<:Number
    out = Variable(-a.value)
    push!(out.parents, (a, -one(DT)))
    return out
end

function Base.:/(a::Variable{DT}, b::Variable{DT}) where DT<:Number
    out = Variable(a.value / b.value)
    push!(out.parents, (a, one(DT) / b.value))
    push!(out.parents, (b, -a.value / (b.value^2)))
    return out
end

function Base.:/(a::Variable{DT}, b::Number) where DT<:Number
    out = Variable(a.value / DT(b))
    push!(out.parents, (a, one(DT) / DT(b)))
    return out
end

function Base.:/(a::Number, b::Variable{DT}) where DT<:Number
    out = Variable(DT(a) / b.value)
    push!(out.parents, (b, -DT(a) / (b.value^2)))
    return out
end

function Base.:^(a::Variable{DT}, b::Integer) where DT<:Number
    out = Variable(a.value^b)
    push!(out.parents, (a, DT(b) * a.value^(b - 1)))
    return out
end

function Base.:^(a::Variable{DT}, b::Number) where DT<:Number
    out = Variable(a.value^b)
    push!(out.parents, (a, DT(b) * a.value^(b - 1)))
    return out
end

function Base.exp(a::Variable{DT}) where DT<:Number
    val = exp(a.value)
    out = Variable(val)
    push!(out.parents, (a, val))
    return out
end

function Base.log(a::Variable{DT}) where DT<:Number
    out = Variable(log(a.value))
    push!(out.parents, (a, one(DT) / a.value))
    return out
end

function Base.sqrt(a::Variable{DT}) where DT<:Number
    val = sqrt(a.value)
    out = Variable(val)
    push!(out.parents, (a, one(DT) / (2 * val)))
    return out
end

function Base.sin(a::Variable{DT}) where DT<:Number
    out = Variable(sin(a.value))
    push!(out.parents, (a, cos(a.value)))
    return out
end

function Base.cos(a::Variable{DT}) where DT<:Number
    out = Variable(cos(a.value))
    push!(out.parents, (a, -sin(a.value)))
    return out
end

function Base.tanh(a::Variable{DT}) where DT<:Number
    val = tanh(a.value)
    out = Variable(val)
    push!(out.parents, (a, one(DT) - val^2))
    return out
end

function Base.abs(a::Variable{DT}) where DT<:Number
    out = Variable(abs(a.value))
    push!(out.parents, (a, sign(a.value)))
    return out
end

function relu(a::Variable{DT}) where DT<:Number
    val = max(zero(DT), a.value)
    out = Variable(val)
    push!(out.parents, (a, a.value > zero(DT) ? one(DT) : zero(DT)))
    return out
end
relu(x::Number) = max(zero(x), x)

sigmoid(a::Variable{DT}) where DT<:Number = one(DT) / (one(DT) + exp(-a))
sigmoid(x::Number) = one(x) / (one(x) + exp(-x))

softplus(a::Variable{DT}) where DT<:Number = log(one(DT) + exp(a))
softplus(x::Number) = log(one(x) + exp(x))

