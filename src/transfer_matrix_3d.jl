
export G, transfer_matrix_3d

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
            Pvt[1:ML,1:ML] .= construct_from_spline(gpm, gpm.P_t, j, deg2rad(tilts[i,1]), deg2rad(tilts[i,2]))
            Pvt[ML+1:2ML,ML+1:2ML] .= inv(Pvt[1:ML,1:ML])
            
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








