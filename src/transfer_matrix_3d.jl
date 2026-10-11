
export G, transfer_matrix_3d, transfer_matrix_3d_explicit, transfer_matrix_3d_adaptive

function G(ML::Number,
    n1::Number,
    n2::Number)
    """
    Constructs the reflection and transmission matrices for a given interface between two materials with refractive indices n1 and n2.
    """
    g = zeros(ComplexF64,2ML,2ML)

    g[1:ML,1:ML] += I(ML)*(n2+n1)/2n2
    g[ML+1:2ML,ML+1:2ML] += I(ML)*(n2+n1)/2n2

    g[ML+1:2ML,1:ML] += I(ML)*(n2-n1)/2n2
    g[1:ML,ML+1:2ML] += I(ML)*(n2-n1)/2n2

    return g
end

function transfer_matrix_3d(gpm::GrandPropagationMatrix,
        distances::Array{<:Real},
        tilts::Array{<:Real},
        ax::Vector{ComplexF64},
        freqs::AbstractArray{<:Real};
        waveguide=true)
    """
    Takes a GrandPropagationMatrix and returns the boost factor for a given set of distances, 
    relative tilts, axion modes, and frequencies. Eps, tand, nm, and thickness are optional 
    parameters for the material properties of the disks. Waveguide is an optional parameter 
    that determines whether to use the perfect waveguide approximation for the propagation 
    matrix inside the disks.
    """
    
    ML = gpm.ML

    B = zeros(ComplexF64,ML,length(freqs))

    eps  = gpm.eps*(1.0-1.0im*gpm.tand); nd = sqrt(eps); nm = complex(gpm.nm); ϵm = nm^2
    A  = 1-1/eps; A0 = 1-1/ϵm

    G0 = G(ML,nm,1) # Reflection and transmission matrices for the mirror
    Gv = G(ML,1,nd) # Reflection and transmission matrices for the disk -> vacuum interface
    Gd = G(ML,nd,1) # Reflection and transmission matrices for the vacuum -> disk interface
    
    S  = A/2
    S0 = A0/2

    T  = Matrix{ComplexF64}(I, 2*ML, 2*ML)

    Threads.@threads for j in 1:length(freqs)
        MM  = Matrix{ComplexF64}(I, 2*ML, 2*ML)*0
        tmp = Matrix{ComplexF64}(undef, 2*ML, 2*ML)
        Pvt = Matrix{ComplexF64}(I,2*ML,2*ML)
        Pvd = Matrix{ComplexF64}(I,2*ML,2*ML)
        Pv0 = Matrix{ComplexF64}(I,2*ML,2*ML)
        Pd  = Matrix{ComplexF64}(I,2*ML,2*ML)
        
        # Propagation matrix for propagation inside disks.
        Pd_ = cispi(+2*freqs[j]*nd*gpm.thickness/c0)
        Pd  = diagm([fill(conj(Pd_),ML); fill(Pd_,ML)])

        # Tilt of the first disk.
        Pv0[1:ML,1:ML] .= construct_from_spline(gpm, gpm.P_t, j,-deg2rad(tilts[1,1]),-deg2rad(tilts[1,2]))
        Pv0[ML+1:2ML,ML+1:2ML] .= inv(Pv0[1:ML,1:ML]) 

        T0 = copy(T) * Pv0

        # iterate in reverse order to sum up MM in single sweep (thx david)
        
        for i in Iterators.reverse(eachindex(distances))
            if waveguide == false
                Pd[1:ML,1:ML] .= construct_from_spline(gpm, gpm.P_disk, j, gpm.thickness); Pd[ML+1:2ML,ML+1:2ML] .= inv(Pd[1:ML,1:ML])
            end
            # Distance propagation matrix
            Pvd[1:ML,1:ML] .= construct_from_spline(gpm, gpm.P_d, j, distances[i])
            Pvd[ML+1:2ML,ML+1:2ML] .= inv(Pvd[1:ML,1:ML])
            # Tilt propagation matrix
            if tilts[i,1] == tilts[i,2] == 0.0
                Pvt .= Matrix{ComplexF64}(I,2*ML,2*ML)
            else
                Pvt[1:ML,1:ML] .= construct_from_spline(gpm, gpm.P_t, j, deg2rad(tilts[i,1]), deg2rad(tilts[i,2]))
                Pvt[ML+1:2ML,ML+1:2ML] .= inv(Pvt[1:ML,1:ML])
            end
            
            # MM += T0 * S
            axpy!(S, T0, MM)
            # T0 *= Gd * Pd
            mul!(tmp, T0, Gd)
            mul!(T0, tmp, Pd)
            # MM -= T0 * S
            axpy!(-S, T0, MM)

            # T0 *= Gv * Pvd * Pvt                      
            mul!(tmp, T0, Gv)
            mul!(T0, tmp, Pvd)
            mul!(tmp, T0, Pvt)
            mul!(T0, tmp, I)

            if i == 1
                # Reflection from the mirror at the end of the cavity
                # MM += T0 * S0                         # Construction of M
                axpy!(S0, T0, MM)
                T0 *= G0                              # T *= G0
            end
        end

        # Split the block matrix into its components

        M11 = @view MM[1:ML,1:ML]
        M12 = @view MM[1:ML,ML+1:2ML]
        M21 = @view MM[ML+1:2ML,1:ML]
        M22 = @view MM[ML+1:2ML,ML+1:2ML]

        T12 = @view T0[1:ML,ML+1:2ML]
        T22 = @view T0[ML+1:2ML,ML+1:2ML]

        # Compute the boost factor for the given frequency and store it in B

        B[:,j] = ((M11+M12) - T12*inv(T22)*(M21+M22))*ax
    end
    return B
end


function transfer_matrix_3d(gpm::GrandPropagationMatrix,
        distances::Array{<:Real},
        ax::Vector{ComplexF64},
        freqs::AbstractArray{<:Real};
        waveguide=true)
    """
    Takes a GrandPropagationMatrix and returns the boost factor for a given set of distances, 
    relative tilts, axion modes, and frequencies. Eps, tand, nm, and thickness are optional 
    parameters for the material properties of the disks. Waveguide is an optional parameter 
    that determines whether to use the perfect waveguide approximation for the propagation 
    matrix inside the disks.
    """
    
    ML = gpm.ML

    B = zeros(ComplexF64,ML,length(freqs))

    eps  = gpm.eps*(1.0-1.0im*gpm.tand); nd = sqrt(eps); nm = complex(gpm.nm); ϵm = nm^2
    A  = 1-1/eps; A0 = 1-1/ϵm

    G0 = G(ML,nm,1) # Reflection and transmission matrices for the mirror
    Gv = G(ML,1,nd) # Reflection and transmission matrices for the disk -> vacuum interface
    Gd = G(ML,nd,1) # Reflection and transmission matrices for the vacuum -> disk interface
    
    S  = A/2
    S0 = A0/2

    T  = Matrix{ComplexF64}(I, 2*ML, 2*ML)

    Threads.@threads for j in 1:length(freqs)
        MM  = Matrix{ComplexF64}(I, 2*ML, 2*ML)*0
        tmp = Matrix{ComplexF64}(undef, 2*ML, 2*ML)
        Pvd = Matrix{ComplexF64}(I,2*ML,2*ML)
        Pv0 = Matrix{ComplexF64}(I,2*ML,2*ML)
        Pd  = Matrix{ComplexF64}(I,2*ML,2*ML)
        
        # Propagation matrix for propagation inside disks.
        Pd_ = cispi(+2*freqs[j]*nd*gpm.thickness/c0)
        Pd  = diagm([fill(conj(Pd_),ML); fill(Pd_,ML)])

        T0 = copy(T)

        # iterate in reverse order to sum up MM in single sweep (thx david)
        
        for i in Iterators.reverse(eachindex(distances))
            if waveguide == false
                Pd[1:ML,1:ML] .= construct_from_spline(gpm, gpm.P_disk, j, gpm.thickness); Pd[ML+1:2ML,ML+1:2ML] .= inv(Pd[1:ML,1:ML])
            end
            # Distance propagation matrix
            Pvd[1:ML,1:ML] .= construct_from_spline(gpm, gpm.P_d, j, distances[i])
            Pvd[ML+1:2ML,ML+1:2ML] .= inv(Pvd[1:ML,1:ML])
            
            # MM += T0 * S
            axpy!(S, T0, MM)
            # T0 *= Gd * Pd
            mul!(tmp, T0, Gd)
            mul!(T0, tmp, Pd)
            # MM -= T0 * S
            axpy!(-S, T0, MM)

            # T0 *= Gv * Pvd                    
            mul!(tmp, T0, Gv)
            mul!(T0, tmp, Pvd)

            if i == 1
                # Reflection from the mirror at the end of the cavity
                # MM += T0 * S0                         # Construction of M
                axpy!(S0, T0, MM)
                T0 *= G0                              # T *= G0
            end
        end

        # Split the block matrix into its components

        M11 = @view MM[1:ML,1:ML]
        M12 = @view MM[1:ML,ML+1:2ML]
        M21 = @view MM[ML+1:2ML,1:ML]
        M22 = @view MM[ML+1:2ML,ML+1:2ML]

        T12 = @view T0[1:ML,ML+1:2ML]
        T22 = @view T0[ML+1:2ML,ML+1:2ML]

        # Compute the boost factor for the given frequency and store it in B

        B[:,j] = ((M11+M12) - T12*inv(T22)*(M21+M22))*ax
    end
    return B
end



function transfer_matrix_3d_explicit(gpm::GrandPropagationMatrix,
        distances::Array{<:Real},
        tilts::Array{<:Real},
        ax::Vector{ComplexF64},
        freqs::AbstractArray{<:Real},
        modes,
        coords;
        waveguide=true,
        decomp=true,
        )
    """
    Takes a GrandPropagationMatrix and returns the boost factor for a given set of distances, 
    relative tilts, axion modes, and frequencies. Eps, tand, nm, and thickness are optional 
    parameters for the material properties of the disks. Waveguide is an optional parameter 
    that determines whether to use the perfect waveguide approximation for the propagation 
    matrix inside the disks.
    """
    
    ML = gpm.ML

    B = zeros(ComplexF64,ML,length(freqs))

    eps  = gpm.eps*(1.0-1.0im*gpm.tand); nd = sqrt(eps); nm = complex(gpm.nm); ϵm = nm^2
    A  = 1-1/eps; A0 = 1-1/ϵm

    G0 = G(ML,nm,1) # Reflection and transmission matrices for the mirror
    Gv = G(ML,1,nd) # Reflection and transmission matrices for the disk -> vacuum interface
    Gd = G(ML,nd,1) # Reflection and transmission matrices for the vacuum -> disk interface
    
    S  = A/2
    S0 = A0/2

    T  = Matrix{ComplexF64}(I, 2*ML, 2*ML)

    Threads.@threads for j in 1:length(freqs)
        MM  = Matrix{ComplexF64}(I, 2*ML, 2*ML)*0
        tmp = Matrix{ComplexF64}(undef, 2*ML, 2*ML)
        Pvt = Matrix{ComplexF64}(I,2*ML,2*ML)
        Pvd = Matrix{ComplexF64}(I,2*ML,2*ML)
        Pv0 = Matrix{ComplexF64}(I,2*ML,2*ML)
        Pd  = Matrix{ComplexF64}(I,2*ML,2*ML)
        
        # Propagation matrix for propagation inside disks.
        Pd_ = cispi(+2*freqs[j]*nd*gpm.thickness/c0)
        Pd  = diagm([fill(conj(Pd_),ML); fill(Pd_,ML)])

        # Tilt of the first disk.
        Pv0[1:ML,1:ML] .= propagationCoeffs(freqs[j],0.0,-deg2rad(tilts[1,1]),-deg2rad(tilts[1,2]),1.0,modes,coords)
        Pv0[ML+1:2ML,ML+1:2ML] .= inv(Pv0[1:ML,1:ML]) 

        T0 = copy(T) * Pv0

        # iterate in reverse order to sum up MM in single sweep (thx david)
        
        for i in Iterators.reverse(eachindex(distances))
            if waveguide == false
                Pd[1:ML,1:ML] .= construct_from_spline(gpm, gpm.P_disk, j, gpm.thickness); Pd[ML+1:2ML,ML+1:2ML] .= inv(Pd[1:ML,1:ML])
            end
            # Distance propagation matrix
            if decomp == false
                Pvd[1:ML,1:ML] .= propagationCoeffs(freqs[j],distances[i],deg2rad(tilts[i,1]), deg2rad(tilts[i,2]),1.0,modes,coords)
                Pvd[ML+1:2ML,ML+1:2ML] .= inv(Pvd[1:ML,1:ML])
            else
                Pvd[1:ML,1:ML] .= propagationCoeffs(freqs[j],distances[i],0.0, 0.0,1.0,modes,coords)
                Pvd[ML+1:2ML,ML+1:2ML] .= inv(Pvd[1:ML,1:ML])
                if tilts[i,1] == tilts[i,2] == 0.0
                    # Tilt propagation matrix
                    Pvt .= Matrix{ComplexF64}(I,2*ML,2*ML)
                else
                    Pvt[1:ML,1:ML] .= propagationCoeffs(freqs[j],0.0,deg2rad(tilts[i,1]), deg2rad(tilts[i,2]),1.0,modes,coords)
                    Pvt[ML+1:2ML,ML+1:2ML] .= inv(Pvt[1:ML,1:ML])
                end
            end

            
            
            
            # MM += T0 * S
            axpy!(S, T0, MM)
            # T0 *= Gd * Pd
            mul!(tmp, T0, Gd)
            mul!(T0, tmp, Pd)
            # MM -= T0 * S
            axpy!(-S, T0, MM)

            # T0 *= Gv * Pvd * Pvt                      
            mul!(tmp, T0, Gv)
            mul!(T0, tmp, Pvd)
            mul!(tmp, T0, Pvt)
            mul!(T0, tmp, I)

            if i == 1
                # Reflection from the mirror at the end of the cavity
                # MM += T0 * S0                         # Construction of M
                axpy!(S0, T0, MM)
                T0 *= G0                              # T *= G0
            end
        end

        # Split the block matrix into its components

        M11 = @view MM[1:ML,1:ML]
        M12 = @view MM[1:ML,ML+1:2ML]
        M21 = @view MM[ML+1:2ML,1:ML]
        M22 = @view MM[ML+1:2ML,ML+1:2ML]

        T12 = @view T0[1:ML,ML+1:2ML]
        T22 = @view T0[ML+1:2ML,ML+1:2ML]

        # Compute the boost factor for the given frequency and store it in B

        B[:,j] = ((M11+M12) - T12*inv(T22)*(M21+M22))*ax
    end
    return B
end


function _block_from_mode_mixing(ML::Int, p::Matrix{ComplexF64})
    """
    Embeds a ML x ML mode-mixing matrix p into the 2ML x 2ML block-diagonal form
    [p 0; 0 inv(p)] used throughout the recursion for forward/backward propagation.
    """
    full = Matrix{ComplexF64}(I, 2ML, 2ML)
    full[1:ML,1:ML] .= p
    full[ML+1:2ML,ML+1:2ML] .= inv(p)
    return full
end

function _run_disk_stack(ML::Int,
        G0::Matrix{ComplexF64}, Gv::Matrix{ComplexF64}, Gd::Matrix{ComplexF64},
        S::Number, S0::Number,
        Pd::Matrix{ComplexF64},
        distances::Array{<:Real},
        Pv0_full::Matrix{ComplexF64},
        build_Pvd_full::Function)
    """
    Shared recursive disk-stack evaluation used by both the cheap (spline-factorized)
    and exact (explicit, un-factorized) passes of transfer_matrix_3d_adaptive.
    `build_Pvd_full(i)` must return the full 2ML x 2ML block mode-mixing matrix
    (distance [+ tilt]) for gap i; how it is constructed (factorized or combined)
    is the only difference between the cheap and exact passes.
    Returns (M, T22) where M = (M11+M12) - T12*inv(T22)*(M21+M22), not yet
    multiplied by the axion mode vector.
    """
    MM  = zeros(ComplexF64, 2*ML, 2*ML)
    tmp = Matrix{ComplexF64}(undef, 2*ML, 2*ML)
    T0  = copy(Pv0_full)

    for i in Iterators.reverse(eachindex(distances))
        Pvd_full = build_Pvd_full(i)

        # MM += T0 * S
        axpy!(S, T0, MM)
        # T0 *= Gd * Pd
        mul!(tmp, T0, Gd)
        mul!(T0, tmp, Pd)
        # MM -= T0 * S
        axpy!(-S, T0, MM)

        # T0 *= Gv * Pvd_full
        mul!(tmp, T0, Gv)
        mul!(T0, tmp, Pvd_full)

        if i == 1
            # Reflection from the mirror at the end of the cavity
            axpy!(S0, T0, MM)
            T0 = T0 * G0
        end
    end

    M11 = @view MM[1:ML,1:ML]
    M12 = @view MM[1:ML,ML+1:2ML]
    M21 = @view MM[ML+1:2ML,1:ML]
    M22 = @view MM[ML+1:2ML,ML+1:2ML]

    T12 = @view T0[1:ML,ML+1:2ML]
    T22 = @view T0[ML+1:2ML,ML+1:2ML]

    return ((M11+M12) - T12*inv(T22)*(M21+M22)), Matrix(T22)
end

function transfer_matrix_3d_adaptive(gpm::GrandPropagationMatrix,
        distances::Array{<:Real},
        tilts::Array{<:Real},
        ax::Vector{ComplexF64},
        freqs::AbstractArray{<:Real},
        modes,
        coords;
        waveguide=true,
        cond_threshold::Real=20.0,
        verbose::Bool=false)
    """
    Hybrid between the cheap, spline-interpolated transfer_matrix_3d and the exact,
    un-factorized transfer_matrix_3d_explicit.

    transfer_matrix_3d builds each disk's mode-mixing matrix as a product of two
    separately-interpolated matrices, Pvd*Pvt (pure distance propagation, pure tilt
    propagation). This factorization implicitly inserts a finite-mode-basis
    projection between the tilt and distance operators. The resulting defect is
    normally tiny (~0.1-3% per bounce), but for mode sets with M >= 2 it can be
    amplified by 1-2 orders of magnitude at the narrow, high-Q resonances of
    higher-order modes, because the boost factor is extracted via inv(T22), and
    T22 becomes ill-conditioned exactly at those resonances.

    This function evaluates every frequency with the cheap spline path first, and
    uses cond(T22) as a heuristic proxy for how strongly the small factorization
    defect gets amplified. Frequencies with cond(T22) > cond_threshold are
    re-evaluated with the expensive explicit per-disk propagationCoeffs calculation
    (single combined distance+tilt propagation per disk; no factorization, no
    spline).

    Caveat: cond(T22) is a useful but *imperfect* gate. It reliably flags the
    practically relevant cases (large absolute boost deviation at a resonance),
    but is not an exact bound on the output error -- some ill-conditioned but
    ultimately harmless frequencies may be recomputed needlessly (safe, just
    slower), and in principle a frequency with moderate cond(T22) could still
    carry a small residual error from the factorization. Tune cond_threshold and
    inspect `recomputed` against a trusted reference (e.g. BoostFractor) for your
    own geometry before relying on this unsupervised.

    Returns (B, recomputed) where recomputed is a BitVector flagging which
    frequencies required the expensive fallback.
    """

    ML = gpm.ML

    B = zeros(ComplexF64,ML,length(freqs))
    recomputed = falses(length(freqs))

    eps  = gpm.eps*(1.0-1.0im*gpm.tand); nd = sqrt(eps); nm = complex(gpm.nm); ϵm = nm^2
    A  = 1-1/eps; A0 = 1-1/ϵm

    G0 = G(ML,nm,1)
    Gv = G(ML,1,nd)
    Gd = G(ML,nd,1)

    S  = A/2
    S0 = A0/2

    Threads.@threads for j in 1:length(freqs)
        Pd_ = cispi(+2*freqs[j]*nd*gpm.thickness/c0)
        Pd  = diagm([fill(conj(Pd_),ML); fill(Pd_,ML)])
        if waveguide == false
            pdisk = construct_from_spline(gpm, gpm.P_disk, j, gpm.thickness)
            Pd[1:ML,1:ML] .= pdisk
            Pd[ML+1:2ML,ML+1:2ML] .= inv(pdisk)
        end

        # --- cheap, spline-factorized pass -----------------------------------
        Pv0_cheap = _block_from_mode_mixing(ML,
            construct_from_spline(gpm, gpm.P_t, j, -deg2rad(tilts[1,1]), -deg2rad(tilts[1,2])))

        build_Pvd_cheap = i -> begin
            Pd_block = _block_from_mode_mixing(ML, construct_from_spline(gpm, gpm.P_d, j, distances[i]))
            if tilts[i,1] == 0.0 && tilts[i,2] == 0.0
                Pd_block
            else
                Pt_block = _block_from_mode_mixing(ML,
                    construct_from_spline(gpm, gpm.P_t, j, deg2rad(tilts[i,1]), deg2rad(tilts[i,2])))
                Pd_block * Pt_block
            end
        end

        Bmat_cheap, T22_cheap = _run_disk_stack(ML, G0, Gv, Gd, S, S0, Pd, distances, Pv0_cheap, build_Pvd_cheap)

        if cond(T22_cheap) > cond_threshold
            recomputed[j] = true
            if verbose
                println("freq=$(freqs[j]): cond(T22)=$(cond(T22_cheap)) > $cond_threshold -> falling back to explicit calculation")
            end

            # --- expensive, exact fallback pass ------------------------------
            Pv0_exact = _block_from_mode_mixing(ML,
                propagationCoeffs(freqs[j], 0.0, -deg2rad(tilts[1,1]), -deg2rad(tilts[1,2]), 1.0, modes, coords))

            build_Pvd_exact = i -> _block_from_mode_mixing(ML,
                propagationCoeffs(freqs[j], distances[i], deg2rad(tilts[i,1]), deg2rad(tilts[i,2]), 1.0, modes, coords))

            Bmat_exact, _ = _run_disk_stack(ML, G0, Gv, Gd, S, S0, Pd, distances, Pv0_exact, build_Pvd_exact)
            B[:,j] = Bmat_exact * ax
        else
            B[:,j] = Bmat_cheap * ax
        end
    end

    return B, recomputed
end
