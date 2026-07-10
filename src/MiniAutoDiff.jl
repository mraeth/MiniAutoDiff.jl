module MiniAutoDiff

    using Base

    include("core.jl")
    include("autodiff.jl")


export Variable, diff

end
