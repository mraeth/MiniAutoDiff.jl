abstract type Optimizer{DT} end

struct GradientDescent{DT} <: Optimizer{DT}
    learning_rate::DT
end

function update(parameter::Variable, optimizer::GradientDescent)
    parameter.value = parameter.value - optimizer.learning_rate * parameter.grad
end

function update(layer::Layer, optimizer::Optimizer)
    for parameter in parameters(layer)
        update(parameter, optimizer)
    end
end

function update(model::Model, optimizer::Optimizer)
    for layer in model.layers
        update(layer, optimizer)
    end
end
