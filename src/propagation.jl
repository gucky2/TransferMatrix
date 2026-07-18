
export propagate!, tiltField!, propagationCoeffs




function propagate!(E0::Matrix{ComplexF64},
        k0::Number,
        coords::Coordinates,
        dz::Real)
    """
    Propagates the field E0 by distance dz in the z-direction, using explicit FFT-based propagation.
    """
    fft!(E0)
    @. E0 *= cis(-conj(sqrt(k0^2-coords.kR)*dz))
    ifft!(E0)
    
    return
end

function tiltField!(E0::Matrix{ComplexF64},
        k0::Number,
        coords::Coordinates,
        tiltx::Real,
        tilty::Real)
    """
    Tilts the field E0 by tiltx and tilty in the x and y directions.
    """
    E0 .*= cis.(-k0*(tiltx*coords.R .* cos.(coords.Φ) + tilty*coords.R .* sin.(coords.Φ)))

    return
end

function propagate!(E0::Matrix{ComplexF64},
        k0::Number,
        coords::Coordinates,
        dz::Real,
        tiltx::Real,
        tilty::Real)
    """
    Decides whether to propagate or tilt the field E0, based on dz and tiltx/tilty.
    """
    if dz != 0
        propagate!(E0,k0,coords,dz)
    end
    if tiltx + tilty != tiltx - tilty
        tiltField!(E0,k0,coords,tiltx,tilty)
    end

    return
end



function propagationCoeffs(freq::Real,
        distance::Real,
        tiltx::Real,
        tilty::Real,
        eps::Number,
        modes::Modes,
        coords::Coordinates)
    """
    Constructs the MLxML propagation matrix for a set of modes, given a frequency, distance, tilts, and permittivity.
    """

    ML = modes.M*(2modes.L+1)
    P = Array{ComplexF64}(undef,ML,ML)

    k0 = 2π*freq/c0*sqrt(eps)

    for ml in 1:ML
        mode = copy(modes[:,:,1,ml])
        propagate!(mode,k0,coords,distance,tiltx,tilty)
        mode .*= coords.diskmaskin
        coeffs = modeDecomp(mode,modes)
        @views copyto!(P[:,ml],coeffs)
    end

    return P
end

