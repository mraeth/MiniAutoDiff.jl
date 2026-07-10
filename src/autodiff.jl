
function collect!(register, v::Variable{DT}) where DT
    if v ∉ register
        for (parent, _) in v.parents
            collect!(register, parent)
        end
        v.grad = zero(DT)
        push!(register, v)
    end
end

function diff(v::Variable{DT}) where DT
    register = []
    collect!(register, v)
    v.grad = one(DT)
    for v in reverse(register)
        for (parent, localGrad) in v.parents
            parent.grad += v.grad * localGrad
        end
    end
end