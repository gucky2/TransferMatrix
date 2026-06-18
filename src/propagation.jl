
export propagate!, tiltField!, propMatFreeSpace, propMatWaveGuide, propagationMatrix




function propagate!(E0::Matrix{ComplexF64},k0::Number,coords::Coordinates,dz::Real)
    fft!(E0)
    @. E0 *= cis(-conj(sqrt(k0^2-coords.kR))*dz)
    ifft!(E0)

    return
end

function propagate!(E0::Matrix{ComplexF64},k0::Number,coords::Coordinates,dz::Real,
        tiltx::Real,tilty::Real)
        
    propagate!(E0,k0,coords,dz)
    tiltField!(E0,k0,coords,tiltx,tilty)

    return
end

function tiltField!(E0::Matrix{ComplexF64},k0::Number,coords::Coordinates,
        tiltx::Real,tilty::Real)
    
        E0 .*= cis.(-k0*(tiltx*coords.R .* cos.(coords.Φ) + tilty*coords.R .* sin.(coords.Φ)))


    return
end


function propMatFreeSpace(freqs::Union{Real,AbstractVector{<:Real}},
        distances::AbstractVector{<:Real},
        tilts::AbstractVector{<:Real},
        eps::Number,modes::Modes,coords::Coordinates)

    ML = modes.M*(2modes.L+1)
    P = Array{ComplexF64}(undef,ML,ML,length(freqs),
                    length(distances),length(tilts),length(tilts))

    for ty in eachindex(tilts), tx in eachindex(tilts)
        for j in eachindex(distances),i in eachindex(freqs)
            k0 = 2π*freqs[i]/c0*sqrt(eps)

            for ml in 1:ML#, k in axes(modes,3) # CHECK: kr here?
                mode = copy(modes[:,:,1,ml])
                propagate!(mode,k0,coords,distances[j],tilts[tx],tilts[ty])
                coeffs = modeDecomp(mode,modes)
                @views copyto!(P[:,ml,i,j,tx,ty],coeffs)
            end
        end
    end

    return P
end

function propMatWaveGuide(freqs::AbstractVector{<:Real},distances::AbstractVector{<:Real},
        eps::Number,modes::Modes,coords::Coordinates)

    ML = modes.M*(2*modes.L+1)
    P = Array{ComplexF64}(undef,ML,ML,length(distances),length(freqs))

    for j in eachindex(freqs)
        k0 = 2π*freqs[i]/c0*sqrt(eps)

        for i in eachindex(distances)
            for ml in 1:ML
                P[ml,ml,i,j] .= cis(-k0*distances[i])
            end
        end
    end

    return P
end


function propagationMatrix(freqs::Union{Real,AbstractVector{<:Real}},
        distances::AbstractVector{<:Real},
        tilts::AbstractVector{<:Real},
        eps::Number,modes::Modes,coords::Coordinates; waveguide::Bool=false)

    @assert all(distances .>= 0) "All propagated distances dz must be positive."
    @assert all(freqs .> 0) "All frequencies freqs must be positive."

    eps = complex(eps)

    if waveguide
        return propMatWaveGuide(freqs,distances,eps,modes,coords)
    else
        return propMatFreeSpace(freqs,distances,tilts,eps,modes,coords)
    end
end

const propMatrix = const propMat = const prop = propagationMatrix
