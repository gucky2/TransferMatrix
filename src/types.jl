using Interpolations

export Coordinates, Modes, GrandPropagationMatrix

struct Coordinates
    X::Vector{Float64}
    kX::Vector{Float64}
    R::Matrix{Float64}
    kR::Matrix{Float64}
    Φ::Matrix{Float64}

    diskR::Float64
    diskmaskin::BitMatrix
    diskmaskout::BitMatrix

    function Coordinates(X::AbstractVector=-0.5:0.01:0.5; diskR::Real=0.15)
        @assert X==-reverse(X) && X[1]*X[end]<0 "Coordinates must be symmetrical around 0."

        kX = kSpace(X)
        R  = [sqrt(x^2+y^2) for x in X, y in X]
        m  = R .<= diskR
        
        new(X,kX,R,
            [kx^2+ky^2 for kx in kX, ky in kX],
            [atan(y,x) for  x in  X,  y in  X],
            diskR,
            m,.!m
        )
    end

    function Coordinates(xsize::Real,dx::Real; diskR::Real=0.15)
        @assert xsize*dx > 0 "Inputs must be larger than 0."
    
        nx = ceil(xsize/2dx);
        X = -nx*dx:dx:nx*dx
        kX = kSpace(X)

        R  = [sqrt(x^2+y^2) for x in X, y in X]
        m  = R .<= diskR
    
        new(X,kX,R,
            [kx^2+ky^2 for kx in kX, ky in kX],
            [atan(y,x) for  x in  X,  y in  X],
            diskR,
            m,.!m
        )
    end
end





mutable struct Modes
    M::Int64
    L::Int64
    modes::Array{ComplexF64,4}
    kt::Vector{ComplexF64}
    id::Matrix{ComplexF64}
    zero::Matrix{ComplexF64}

    function Modes(M,L,modes,kt)                
        @assert M > 0 "m needs to be larger than 0."

        ML = M*(2L+1)
        id = Matrix{ComplexF64}(I,ML,ML)
        z = zeros(ComplexF64,ML,ML)

        new(M,L,modes,kt,id,z)
    end

    function Modes(coords,M,L)
        @assert M > 0 "m needs to be larger than 0."

        ML = M*(2L+1)
        modes = zeros(ComplexF64,length(coords.X),length(coords.X),1,ML)
        kt = zeros(ComplexF64,ML)
        
        for m in 1:M, l in -L:L
            ml = modeidx(m,l,L)
            kt[ml], modes[:,:,:,ml] = mode(coords,m,l)
        end

        return Modes(M,L,modes,kt)
    end
end

fieldDims(modes::Modes) = size(modes.modes,3)

function modeidx(m::Int,l::Int,L::Int)
    return (m-1)*(2L+1)+l+L+1
end

function modeidx(ml::Int,L::Int)
    return div(ml-1,(2L+1))+1, (ml-1)%(2L+1)+-L
end



Base.getindex(m::Modes,inds...) = getindex(m.modes,inds...)
Base.getindex(m::Modes,ind1,ind2) = getindex(m.modes,:,:,:,modeidx(ind1,ind2,m.L))
Base.getindex(m::Modes,ind1,ind2,ind3) = getindex(m.modes,:,:,ind1,modeidx(ind2,ind3,m.L))

Base.setindex!(m::Modes,x,inds...) = setindex!(m.modes,x,inds...)
Base.setindex(m::Modes,ind1,ind2) = setindex(m.modes,:,:,:,modeidx(ind1,ind2,m.L))
Base.setindex(m::Modes,ind1,ind2,ind3) = setindex(m.modes,:,:,ind1,modeidx(ind2,ind3,m.L))

Base.size(m::Modes) = size(m.modes)
Base.size(m::Modes,d::Integer) = size(m.modes,d)
Base.axes(m::Modes,d::Integer) = axes(m.modes,d)








mutable struct GrandPropagationMatrix
    """
    Precalculates the propagation matrices for a set of distances and relative tilts to later interpolate from.
    P_d contains the spline for distance interpolation.
    P_t contains the spline for tilt interpolation.
    """
    freqs::AbstractArray{<:Real}
    thickness::Float64
    eps::Float64
    tand::Float64
    nm::Float64

    M::Int
    L::Int
    ML::Int

    P_d::ScaledInterpolation
    P_t::ScaledInterpolation
    P_disk::ScaledInterpolation

    function GrandPropagationMatrix(freqs::AbstractArray{<:Real},
            modes::Modes,
            coords::Coordinates; 
            eps=24.0,
            tand=0.,
            nm=1e30,
            thickness=1e-3)
        
        distances = range(0e-3, 10e-3, 50)
        tilts = range(deg2rad(-0.05), deg2rad(0.05), 10)

        M = modes.M; L = modes.L; ML = M*(2L+1)
        p_d = Array{ComplexF64}(undef,length(freqs),length(distances),ML,ML)
        p_t = Array{ComplexF64}(undef,length(freqs),length(tilts),length(tilts),ML,ML)
        p_disk = Array{ComplexF64}(undef,length(freqs),length(distances),ML,ML)
        bc = BSpline(Cubic(Natural(OnCell())))
        ni = NoInterp()

        # Spline for distance interpolation
        for i in eachindex(freqs), j in eachindex(distances)
            p_d[i,j,:,:] .= propagationCoeffs(freqs[i],distances[j],0.0,0.0,1.0,modes,coords)
        end
        itpP_d = interpolate(p_d,(ni,bc,ni,ni))
        sitpP_d = scale(itpP_d,1:length(freqs),distances,1:ML,1:ML)

        # Spline for disk propagation
        for i in eachindex(freqs), j in eachindex(distances)
            p_disk[i,j,:,:] .= propagationCoeffs(freqs[i],distances[j],0.0,0.0,eps,modes,coords)
        end
        itpP_disk = interpolate(p_disk,(ni,bc,ni,ni))
        sitpP_disk = scale(itpP_disk,1:length(freqs),distances,1:ML,1:ML)

        # Spline for tilt interpolation
        for i in eachindex(freqs), j in eachindex(tilts), k in eachindex(tilts)
            p_t[i,j,k,:,:] .= propagationCoeffs(freqs[i],0.0,tilts[j],tilts[k],1.0,modes,coords)
        end
        itpP_t = interpolate(p_t,(ni,bc,bc,ni,ni))
        sitpP_t = scale(itpP_t,1:length(freqs),tilts,tilts,1:ML,1:ML)

        new(freqs,thickness,eps,tand,nm,M,L,ML,sitpP_d,sitpP_t,sitpP_disk)
    end
end

const GPM = GrandPropagationMatrix

function construct_from_spline(
        gpm::GrandPropagationMatrix,
        P::ScaledInterpolation, 
        f::Real, 
        distance::Real;)
    """
    Interpolates the propagation matrix for a given frequency and distance.
    """
    mat = Array{ComplexF64}(undef,gpm.ML,gpm.ML)

    for i in 1:gpm.ML, j in 1:gpm.ML
        mat[i,j] = P(f, distance, i, j)
    end

    return mat
end

function construct_from_spline(
        gpm::GrandPropagationMatrix,
        P::ScaledInterpolation, 
        f::Real, 
        tiltx::Real, 
        tilty::Real;)
    """
    Interpolates the propagation matrix for a given frequency and tilts.
    """
    mat = Array{ComplexF64}(undef,gpm.ML,gpm.ML)

    for i in 1:gpm.ML, j in 1:gpm.ML
        mat[i,j] = P(f, tiltx, tilty, i, j)
    end

    return mat
end
