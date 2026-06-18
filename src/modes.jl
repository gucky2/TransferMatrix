
export mode, modeDecomp, axionModes, modes2field




function mode(coords::Coordinates,m::Integer,l::Integer)
    kr = besselj_zero(l,m)/coords.diskR

    mode = @. besselj(l,kr*coords.R)*cis(-l*coords.Φ)
    @. mode *= coords.diskmaskin
    mode ./= sqrt(sum(abs2.(mode)))
    mode = reshape(mode,(size(mode)...,1))

    return kr, mode
end

function modeDecomp(E::Union{Matrix{ComplexF64},Array{ComplexF64,3}},modes::Modes;)
    @assert size(E,1) == size(E,2) == size(modes,1) "Grids of field and modes don't match."
    @assert size(E,3) == size(modes,3) "Dimensionality of field and modes doesn't match."
    
    ML = modes.M*(2modes.L+1)

    N = sqrt(sum(abs2.(E)))
    coeffs = zeros(ComplexF64,ML)

    for ml in 1:ML; coeffs[ml] = sum(@. conj(modes.modes[:,:,:,ml])*E)/N; end

    return coeffs
end

const modeDecomposition = const decomp = const field2modes = modeDecomp



function axionModes(coords::Coordinates,modes::Modes,f::Real,velocity_x::Real)
    d = fieldDims(modes)
    Ea = zeros(ComplexF64,length(coords.X),length(coords.X),d)
    Ea[:,:,1+Int(d==3)] .= 1; Ea .*= coords.diskmaskin

    # inaccuracies of the emitted fields: B-field and velocity effects
    if velocity_x != 0
        k_a = 2π*f/c0 # k = 2pi/lambda (c/f = lambda)

        for (i,x) in enumerate(coords.X); Ea[i,:,:] .*= cis(k_a*x*velocity_x); end
    end

    return modeDecomp(Ea,modes)
end

function axionModes(coords::Coordinates,modes::Modes)
    d = fieldDims(modes)
    Ea = zeros(ComplexF64,length(coords.X),length(coords.X),d)
    Ea[:,:,1+Int(d==3)] .= 1; Ea .*= coords.diskmaskin

    return modeDecomp(Ea,modes)
end

function modes2field(coeffs::AbstractVector{ComplexF64},modes::Modes)
    field = zeros(ComplexF64,size(modes,1),size(modes,2),size(modes,3))

    for ml in 1:modes.M*(2modes.L+1); @. field += coeffs[ml]*modes.modes[:,:,:,ml]; end

    return field
end


function axionModes(coords::Coordinates,modes::Modes,freqs::AbstractVector{<:Real},
        tilts::AbstractVector{<:Real}; velocity_x::Real=0)

    d = fieldDims(modes)
    Ea = Matrix{ComplexF64}(undef,length(coords.X),length(coords.X))


    ML = modes.M*(2modes.L+1)
    ax = Array{ComplexF64}(undef,ML,length(freqs),length(tilts),length(tilts))

    for ty in eachindex(tilts), tx in eachindex(tilts)
        for f in eachindex(freqs)
            ka = 2π*freqs[f]/c0

            Ea .= 1; Ea .*= coords.diskmaskin
            
            # inaccuracies of the emitted fields: B-field and velocity effects
            if velocity_x != 0
                for (i,x) in enumerate(coords.X); Ea[i,:] .*= cis(ka*x*velocity_x); end
            end

            tiltField!(Ea,ka,coords,tilts[tx],tilts[ty])

            coeffs = modeDecomp(Ea,modes)
            @views copyto!(ax[:,f,tx,ty],coeffs)
        end
    end

    return ax
end