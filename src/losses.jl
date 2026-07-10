function mse_loss(ŷ::AbstractVector, y::AbstractVector)
    @assert length(ŷ) == length(y) "Prediction and target vectors must have equal length (got $(length(ŷ)) vs $(length(y)))"
    sum((ŷᵢ - yᵢ)^2 for (ŷᵢ, yᵢ) in zip(ŷ, y)) / length(y)
end

# TODO: binary_crossentropy — defer until sigmoid outputs are tested
