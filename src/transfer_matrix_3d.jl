



function transfer_matrix_3d(::Type{Dist},distances::AbstractVector{<:Real},
        tiltsx,tiltsy,gpm::GPM,f::Int;
        eps::Real=24.,tand::Real=0.,nm::Real=1e30)
    
    ϵ  = eps*(1.0-1.0im*tand)
    nd = sqrt(ϵ); nm = complex(nm)
    ϵm = nm^2
    A  = 1-1/ϵ
    A0 = 1-1/ϵm

    P = gpm.P; ax = gpm.ax; ML = gpm.ML

    G0 = G(ML,nm,1)
    Gv = G(ML,1,nd)
    Gd = G(ML,nd,1)

    S  = A/2  * I(2ML) * diagm([ax, ax])
    S0 = A0/2 * I(2ML) * diagm([ax, ax])

    T  = Matrix{ComplexF64}(I, 2*ML, 2*ML)
    MM = Matrix{ComplexF64}(I, 2*ML, 2*ML)*0

    W  = Matrix{ComplexF64}(undef, 2*ML, 2*ML)
    TW = Matrix{ComplexF64}(undef, 2*ML, 2*ML)
    


    pd1 = cispi(-2*gpm.freqs[f]*nd*(1e-3)/c0)
    pd2 = cispi(+2*gpm.freqs[f]*nd*(1e-3)/c0)

    for ml in 1:ML
        @views copyto!(T[:,:,ml],Gd)

        s = ax(ml,f,0.,0.)
        MM[1,1,ml] = s*S; MM[2,2,ml] = s*S
    end
    
    # iterate in reverse order to sum up MM in single sweep (thx david)
    for i in Iterators.reverse(eachindex(distances))
        for ml in 1:ML
            @. T[:,1,ml] *= pd1
            @. T[:,2,ml] *= pd2                                             # T = Gd*Pd
            
            @. MM[:,:,ml] -= T[:,:,ml]*S*ax(ml,f,tiltsx[i+1],tiltsy[i+1])   # MM = Gd*Pd*S_-1
            @views mul!(W,T[:,:,ml],Gv); @views copyto!(T[:,:,ml],W)        # T *= Gd*Pd*Gv
        end

        TW .= 0.0im
        
        for ml in 1:ML
            for ml_ in 1:ML
                s = P(ml,ml_,f,distances[i],tiltsx[i+1]-tiltsx[i],tiltsy[i+1]-tiltsy[i])

                @. TW[:,1,ml] += T[:,1,ml_]*s
                @. TW[:,2,ml] += T[:,2,ml_]*conj(s)
            end

            if i > 1
                @. MM[:,:,ml] += TW[:,:,ml]*S*ax(ml,f,tiltsx[i],tiltsy[i])
                @views mul!(W,TW[:,:,ml],Gd); @views copyto!(TW[:,:,ml],W)
            else
                @. MM[:,:,ml] += TW[:,:,ml]*S0*ax(ml,f,tiltsx[i],tiltsy[i])
                @views mul!(W,TW[:,:,ml],G0); @views copyto!(TW[:,:,ml],W)
            end
        end

        copyto!(T,TW)
    end

    for ml in 1:ML
        RB[1,ml] = T[1,2,ml]/T[2,2,ml]
        RB[2,ml] = MM[1,1,ml]+MM[1,2,ml]-(MM[2,1,ml]+MM[2,2,ml])*T[1,2,ml]/T[2,2,ml]
    end

    return RB
end



# function transfer_matrix_3d(::Type{Dist},
#         distances::AbstractVector{<:Real},
#         tiltsx::AbstractVector{<:Real},
#         tiltsy::AbstractVector{<:Real},
#         ax::AbstractArray,f::Int;
#         eps::Real=24.,tand::Real=0.,nm::Real=1e30)
    
#     ϵ  = eps*(1.0-1.0im*tand)
#     nd = sqrt(ϵ); nm = complex(nm)
#     ϵm = nm^2
#     A  = 1-1/ϵ
#     A0 = 1-1/ϵm

#     P = gpm.P; M = gpm.M; L = 2modes.L+1; ML = M*L

#     Gd = ComplexF64[(1+nd)/2   (1-nd)/2;   (1-nd)/2   (1+nd)/2]
#     Gv = ComplexF64[(nd+1)/2nd (nd-1)/2nd; (nd-1)/2nd (nd+1)/2nd]
#     G0 = ComplexF64[(1+nm)/2   (1-nm)/2;   (1-nm)/2   (1+nm)/2]

#     S  = A/2; S0 = A0/2
    
#     RB = Array{ComplexF64}(undef,2,ML)
    
#     # T  = Array{ComplexF64}(undef,2,2,ML)
#     # MM = Array{ComplexF64}(undef,2,2,ML)
#     T  = zeros(ComplexF64,2,2,ML)
#     MM = zeros(ComplexF64,2,2,ML)

#     TW = Array{ComplexF64}(undef,2,2,ML)
#     W = Matrix{ComplexF64}(undef,2,2)
    


#     pd1 = cispi(-2*gpm.freqs[f]*nd*(1e-3)/c0)
#     pd2 = cispi(+2*gpm.freqs[f]*nd*(1e-3)/c0)

#     @. MM[1:4:end] = MM[4:4:end] = S

#     for ml in eachindex(ax)
#         @views copyto!(T[:,:,ml],Gd)
#         # @views copyto!(MM[:,:,ml],S)

#         @. MM[:,:,ml] *= ax[ml]
#     end
    
#     # iterate in reverse order to sum up MM in single sweep (thx david)
#     for i in Iterators.reverse(eachindex(distances))
#         for ml in eachindex(ax)
#             @. T[:,1,ml] *= pd1
#             @. T[:,2,ml] *= pd2                                # T = Gd*Pd
            
#             @. MM[:,:,ml] -= T[:,:,ml]*S#*ax[ml]
#             # @views mul!(MM[:,:,ml],T[:,:,ml],S,-ax[ml],1.)                 # MM = Gd*Pd*S_-1
#             @views mul!(W,T[:,:,ml],Gv); @views copyto!(T[:,:,ml],W)              # T *= Gd*Pd*Gv
#         end

#         TW .= 0.0im
        
#         for ml in eachindex(ax)
#             for ml_ in eachindex(ax)
#                 s = gpm.P(ml,ml_,f,distances[i],tiltsx[i+1]-tiltsx[i],tiltsy[i+1]-tiltsy[i])

#                 @. TW[:,1,ml] += T[:,1,ml_]*s
#                 @. TW[:,2,ml] += T[:,2,ml_]*conj(s)
#             end

#             if i > 1
#                 @. MM[:,:,ml] += TW[:,:,ml]*S#*ax[ml]
#                 # @views mul!(MM[:,:,ml],TW[:,:,ml],S,ax[ml],1.)
#                 @views mul!(W,TW[:,:,ml],Gd); @views copyto!(TW[:,:,ml],W)
#             else
#                 @. MM[:,:,ml] += TW[:,:,ml]*S0#*ax[ml]
#                 # @views mul!(MM[:,:,ml],TW[:,:,ml],S0,ax[ml],1.)
#                 @views mul!(W,TW[:,:,ml],G0); @views copyto!(TW[:,:,ml],W)
#             end
#         end

#         copyto!(T,TW)
#     end

#     for ml in eachindex(ax)
#         RB[1,ml] = T[1,2,ml]/T[2,2,ml]
#         RB[2,ml] = MM[1,1,ml]+MM[1,2,ml]-(MM[2,1,ml]+MM[2,2,ml])*T[1,2,ml]/T[2,2,ml]
#     end

#     return RB
# end



function transfer_matrix_3d(::Type{Dist},distances::AbstractVector{<:Real},
        tiltsx,tiltsy,gpm::GPM,freqs::AbstractVector{<:Real};
        eps::Real=24.,tand::Real=0.,nm::Real=1e30)

    RB = Array{ComplexF64}(undef,2,gpm.ML,length(freqs))
    
    for f in eachindex(freqs)
        RB[:,:,f] = transfer_matrix_3d(Dist,distances,tiltsx,tiltsy,gpm,f;
            eps=eps,tand=tand,nm=nm)
    end

    return RB
end




