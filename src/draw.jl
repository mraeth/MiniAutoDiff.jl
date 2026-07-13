struct SVGImage
    svg::String
end

Base.show(io::IO, ::MIME"image/svg+xml", img::SVGImage) = print(io, img.svg)
Base.show(io::IO, ::MIME"text/plain", img::SVGImage) = print(io, "SVGImage($(length(img.svg)) bytes)")

# --- helpers -----------------------------------------------------------

function layer_sizes(model::Model)
    linears = filter(l -> l isa Linear, model.layers)
    isempty(linears) && return Int[]
    [size(linears[1].weights, 2); [size(l.weights, 1) for l in linears]]
end

# for each Linear layer, find the activation (if any) immediately following it
function layer_activations(model::Model)
    linears_idx = findall(l -> l isa Linear, model.layers)
    acts = Vector{Union{Nothing,Any}}(nothing, length(linears_idx))
    for (k, i) in enumerate(linears_idx)
        if i + 1 <= length(model.layers) && !(model.layers[i+1] isa Linear)
            acts[k] = model.layers[i+1]
        end
    end
    acts  # acts[k] is the activation after the k-th Linear (i.e. feeding into layer k+1's neurons)
end

# small embeddable activation curve, as an <g> group at (x0, y0) with size (w, h)
function _activation_curve_group(activation; x0, y0, w=120, h=70, xrange=(-5.0, 5.0), samples=80)
    xs = collect(range(xrange[1], xrange[2], length=samples))
    ys = Vector{Float64}(forward(activation, xs))

    ymin, ymax = minimum(ys), maximum(ys)
    yspan = ymax == ymin ? 1.0 : ymax - ymin
    xspan = xrange[2] - xrange[1]

    px(x) = x0 + (x - xrange[1]) / xspan * w
    py(y) = y0 + h - (y - ymin) / yspan * h

    io = IOBuffer()
    println(io, """<g>""")
    println(io, """<rect x="$x0" y="$y0" width="$w" height="$h" fill="whitesmoke" stroke="lightgray" stroke-width="1" rx="4"/>""")

    if ymin < 0 < ymax
        println(io, """<line x1="$x0" y1="$(py(0))" x2="$(x0+w)" y2="$(py(0))" stroke="gray" stroke-width="0.75" stroke-dasharray="2,2"/>""")
    end

    pts = join(["$(px(x)),$(py(y))" for (x, y) in zip(xs, ys)], " ")
    println(io, """<polyline points="$pts" fill="none" stroke="firebrick" stroke-width="1.8"/>""")
    println(io, """<text x="$(x0+w/2)" y="$(y0+h+16)" font-size="11" fill="gray30" text-anchor="middle">$(nameof(typeof(activation)))</text>""")
    println(io, "</g>")
    String(take!(io))
end

# --- combined network + activations visualization -----------------------

function draw(model::Model)
    sizes = layer_sizes(model)
    acts  = layer_activations(model)
    n     = length(sizes)
    max_n = maximum(sizes)

    layer_gap  = 160
    margin     = 60
    net_height = max(max_n * 40, 300)
    act_height = 110              # extra strip reserved for activation plots
    width      = n * layer_gap
    height     = net_height + act_height + 40

    xs(i) = margin + (i - 1) * layer_gap
    ys(s) = s == 1 ? [net_height/2] :
            [net_height/2 + y for y in range(-max_n*20, max_n*20, length=s)]

    positions = [[(xs(i), y) for y in ys(s)] for (i, s) in enumerate(sizes)]

    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" style="background:white">""")

    # edges
    for i in 1:n-1, (ax, ay) in positions[i], (bx, by) in positions[i+1]
        println(io, """<line x1="$ax" y1="$ay" x2="$bx" y2="$by" stroke="steelblue" stroke-opacity="0.2" stroke-width="1.2"/>""")
    end

    # neurons
    styles = [("honeydew", "seagreen"); fill(("white", "steelblue"), n - 2); ("mistyrose", "firebrick")]
    for (pts, (fillc, strokec)) in zip(positions, styles)
        for (px, py) in pts
            println(io, """<circle cx="$px" cy="$py" r="11" fill="$fillc" stroke="$strokec" stroke-width="2"/>""")
        end
    end

    # size labels just under the network
    for (i, s) in enumerate(sizes)
        println(io, """<text x="$(xs(i))" y="$(net_height+15)" font-size="14" fill="gray30" text-anchor="middle">$s</text>""")
    end

    # activation mini-plots: acts[k] sits after Linear k, i.e. under column k+1
    plot_w, plot_h = 120, 70
    for (k, act) in enumerate(acts)
        act === nothing && continue
        col = k + 1
        gx = xs(col) - plot_w/2
        gy = net_height + 35
        println(io, _activation_curve_group(act; x0=gx, y0=gy, w=plot_w, h=plot_h))
    end

    println(io, "</svg>")
    SVGImage(String(take!(io)))
end