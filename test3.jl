
using TransferMatrix
using Plots
using LinearAlgebra
using StaticArrays

const c0 = 299792458.


freqs = range(18e9,24e9,1_000);

M = 1; L = 1

coords = Coordinates(1,0.02; diskR=0.15);
modes = Modes(coords,M,L);


# @time gpm = GPM(freqs,distances,tilts,modes,coords);

# E0 = modes2field(ax,modes)[:,:,1]
# showField(E0,coords)

# k0 = 2π*freqs[1]/c0*sqrt(1)
# propagate!(E0,k0,coords,10e-3,deg2rad(0.5),0)
# showField(E0,coords)
# c1 = field2modes(E0,modes);


dists = [7.0]*1e-3

# dists = [
#     7.005317,
#     7.161926,
#     7.436722,
#     7.144421,
#     7.185010,
#     7.209110,
#     7.278833,
#     7.169816,
#     7.250541,
#     7.214103,
#     7.170475,
#     7.245183,
#     7.241939,
#     7.191030,
#     7.208307,
#     7.300933,
#     7.203299,
#     7.265450,
#     6.785361,
#     7.310886,
# ]*1e-3

function propagationCoeffs(freq::Real,
        distance::Real,tiltx::Real,tilty::Real,
        eps::Number,modes::Modes,coords::Coordinates)

    ML = modes.M*(2modes.L+1)
    P = Array{ComplexF64}(undef,ML,ML)

    k0 = 2π*freq/c0*sqrt(eps)

    for ml in 1:ML
        mode = copy(modes[:,:,1,ml])
        propagate!(mode,k0,coords,distance,tiltx,tilty)
        coeffs = modeDecomp(mode,modes)
        @views copyto!(P[:,ml],coeffs)
    end

    return P
end


function transfer_matrix_3d(distances,tiltx,tilty,ax,f,modes,coords; eps=24.0,tand=0.,nm=1e15,thickness=1e-3)
    M = modes.M; L = modes.L; ML = M*(2L+1)
    
    RB = Array{ComplexF64}(undef,2,ML)

    ϵ  = eps*(1.0-1.0im*tand); nd = sqrt(ϵ); nm = complex(nm); ϵm = nm^2
    A  = 1-1/ϵ; A0 = 1-1/ϵm

    Gd = SMatrix{2,2,ComplexF64}((1+nd)/2,   (1-nd)/2,   (1-nd)/2,   (1+nd)/2)
    Gv = SMatrix{2,2,ComplexF64}((nd+1)/2nd, (nd-1)/2nd, (nd-1)/2nd, (nd+1)/2nd)
    G0 = SMatrix{2,2,ComplexF64}((1+nm)/2,   (1-nm)/2,   (1-nm)/2,   (1+nm)/2)
    
    S  = SMatrix{2,2,ComplexF64}( A/2, 0.0im, 0.0im,  A/2)
    S0 = SMatrix{2,2,ComplexF64}(A0/2, 0.0im, 0.0im, A0/2)
    
    T  = [MMatrix{2,2,ComplexF64}(undef) for _ in 1:ML]
    MM = [MMatrix{2,2,ComplexF64}(undef) for _ in 1:ML]

    W  = MMatrix{2,2,ComplexF64}(undef)
    TW = [MMatrix{2,2,ComplexF64}(undef) for _ in 1:ML]



    pd1 = cispi(-2*f*nd*thickness/c0)
    pd2 = cispi(+2*f*nd*thickness/c0)

    st = propagationCoeffs(f,0,deg2rad(-tiltx),deg2rad(tilty),1.0,modes,coords)

    for ml in 1:ML
        copyto!(T[ml],Gd)
        ax_ = st*ax
        copyto!(MM[ml],S); MM .*= ax_[ml]
    end

    # iterate in reverse order to sum up MM in single sweep (thx david)
    
    for i in Iterators.reverse(eachindex(distances))
        st = propagationCoeffs(f,0,deg2rad(-tiltx),deg2rad(tilty),1.0,modes,coords)
        ax_ = st*ax
        
        for ml in 1:ML
            T[ml][:,1] .*= pd1
            T[ml][:,2] .*= pd2                     # T = Gd*Pd
            
            mul!(MM[ml],T[ml],S,-ax_[ml],1.)              # MM = Gd*Pd*S_-1
            mul!(W,T[ml],Gv); copyto!(T[ml],W)            # T *= Gd*Pd*Gv

            TW[ml] .= 0.0im
        end

        s = propagationCoeffs(f,distances[i],deg2rad(tiltx),deg2rad(tilty),1.0,modes,coords)

        for ml in 1:ML           
            for ml_ in 1:ML
                TW[ml_][:,1] .+= T[ml][:,1]*s[ml_,ml]
                TW[ml_][:,2] .+= T[ml][:,2]*conj(s[ml_,ml])
            end

            copyto!(T[ml],TW[ml])
        end
            
        for ml in 1:ML
            if i > 1
                mul!(MM[ml],T[ml],S,ax[ml],1.)
                mul!(W,T[ml],Gd); copyto!(T[ml],W)
            else
                mul!(MM[ml],T[ml],S0,ax[ml],1.)
                mul!(W,T[ml],G0); copyto!(T[ml],W)
            end
        
            RB[1,ml] = T[ml][1,2]/T[ml][2,2]
            RB[2,ml] = MM[ml][1,1]+MM[ml][1,2]-
                (MM[ml][2,1]+MM[ml][2,2])*T[ml][1,2]/T[ml][2,2]
        end
    
    end

    return RB
end

# function transfer_matrix_3d(distances::AbstractVector{<:Real},
#         tiltsx,tiltsy,ax,f::Real,modes::Modes,coords::Coordinates;
#         eps::Real=24.,tand::Real=0.,nm::Real=1e30)
    
#     ϵ  = eps*(1.0-1.0im*tand)
#     nd = sqrt(ϵ); nm = complex(nm); ϵm = nm^2

#     A  = 1-1/ϵ; A0 = 1-1/ϵm

#     Gd = ComplexF64[(1+nd)/2   (1-nd)/2;   (1-nd)/2   (1+nd)/2]
#     Gv = ComplexF64[(nd+1)/2nd (nd-1)/2nd; (nd-1)/2nd (nd+1)/2nd]
#     G0 = ComplexF64[(1+nm)/2   (1-nm)/2;   (1-nm)/2   (1+nm)/2]

#     S  = A/2; S0 = A0/2
    
#     # M = copy(S)

#     ML = modes.M*(2modes.L+1)
    
#     RB = Array{ComplexF64}(undef,2,ML)
    
#     T  = Array{ComplexF64}(undef,2,2,ML)
#     MM = zeros(ComplexF64,2,2,ML)

#     TW = Array{ComplexF64}(undef,2,2,ML)
#     W = Matrix{ComplexF64}(undef,2,2)
    
#     pd1 = cispi(-2*f*nd*(1e-3)/299792458.)
#     pd2 = cispi( 2*f*nd*(1e-3)/299792458.)

#     for ml in 1:ML
#         copyto!(T[:,:,ml],Gd)

#         MM[1,1,ml] = MM[2,2,ml] = ax[ml]*S
#     end
    
#     # iterate in reverse order to sum up MM in single sweep (thx david)
#     for i in Iterators.reverse(eachindex(distances))
#         for ml in 1:ML
#             @. T[:,1,ml] *= pd1
#             @. T[:,2,ml] *= pd2                                              # T = Gd*Pd
            
#             @. MM[:,:,ml] -= T[:,:,ml]*S*ax[ml]   # MM = Gd*Pd*S_-1
#             @views mul!(W,T[:,:,ml],Gv); @views copyto!(T[:,:,ml],W)        # T *= Gd*Pd*Gv
#         end

#         TW .= 0.0im
        
#         P = propagationCoeffs(f,distances[i],0,0,1.0,modes,coords)

#         for ml in 1:ML
#             # for ml_ in 1:ML
#             #     @. TW[:,1,ml] += T[:,1,ml_]#*P[ml,ml_]
#             #     @. TW[:,2,ml] += T[:,2,ml_]#*conj(P[ml,ml_])
#             # end

#             @. TW[:,1,ml] = cispi(-2*f*distances[i]/299792458.)*T[:,1,ml]
#             @. TW[:,2,ml] = cispi( 2*f*distances[i]/299792458.)*T[:,2,ml]

#             if i > 1
#                 @. MM[:,:,ml] += TW[:,:,ml]*S*ax[ml]
#                 @views mul!(W,TW[:,:,ml],Gd); @views copyto!(TW[:,:,ml],W)
#             else
#                 @. MM[:,:,ml] += TW[:,:,ml]*S0*ax[ml]
#                 @views mul!(W,TW[:,:,ml],G0); @views copyto!(TW[:,:,ml],W)
#             end
#         end

#         copyto!(T,TW)
#     end

#     for ml in 1:ML
#         RB[1,ml] = T[1,2,ml]/T[2,2,ml]
#         RB[2,ml] = MM[1,1,ml]+MM[1,2,ml]-(MM[2,1,ml]+MM[2,2,ml])*T[1,2,ml]/T[2,2,ml]
#     end

#     return RB
# end
# %%
ax = axionModes(coords,modes)
B = zeros(ComplexF64,M*(2L+1),length(freqs))

@time for i in eachindex(freqs)
    B[:,i] .= transfer_matrix_3d(dists,0,0,ax,freqs[i],modes,coords)[2,:]
end

# abs2.(propagationCoeffs(freqs[500],7e-3,0,0,1.0,modes,coords))
# abs2.(propagationCoeffs(freqs[500],7e-3,deg2rad(0.1),0,1.0,modes,coords))

# st = propagationCoeffs(freqs[500],0,deg2rad(0.0),deg2rad(0),1.0,modes,coords)

plot(freqs/1e9,abs2.(B)'; label=["L=-1" "L= 0" "L= 1"])

function G(ML,n1,n2)
    g = zeros(ComplexF64,2ML,2ML)

    g[1:ML,1:ML] += I(ML)*(n2+n1)/2n2
    g[ML+1:2ML,ML+1:2ML] += I(ML)*(n2+n1)/2n2

    g[ML+1:2ML,1:ML] += I(ML)*(n2-n1)/2n2
    g[1:ML,ML+1:2ML] += I(ML)*(n2-n1)/2n2

    return g
end

M = 1; L = 1

modes = Modes(coords,M,L);
ax = axionModes(coords,modes)
B = zeros(ComplexF64,M*(2L+1),length(freqs))

ML = M*(2L+1); B = zeros(ComplexF64,ML,length(freqs))
# B = zeros(ComplexF64,length(freqs))

tiltx = deg2rad(0.1)

eps = 24.; tand = 0; nm = 1e15

ϵ  = eps*(1.0-1.0im*tand); nd = sqrt(ϵ); nm = complex(nm); ϵm = nm^2
A  = 1-1/ϵ; A0 = 1-1/ϵm

G0 = G(ML,nm,1)
Gv = G(ML,1,nd)
Gd = G(ML,nd,1)

S  =  A/2*I(2ML)*diagm([ax; ax])
S0 = A0/2*I(2ML)*diagm([ax; ax])

Pv_ = zeros(ComplexF64,2ML,2ML)

# T03 = G2*P2*G1*P1*G0*P0
# MM = T13*S0-T23*S1+T33*S2
@time for i in eachindex(freqs)
    Pv_  = propagationCoeffs(freqs[i],7e-3,0,-tiltx,1.0,modes,coords)
    Pv[1:ML,1:ML] .= Pv_; Pv[ML+1:2ML,ML+1:2ML] .= inv(Pv_)
    # Pv = cispi(+2*freqs[i]*7e-3/c0)
    # Pv_ = diagm([fill(Pv,ML); fill(conj(Pv),ML)])

    Pd_ = cispi(+2*freqs[i]*nd*1e-3/c0)
    Pd = diagm([fill(conj(Pd),ML); fill(Pd,ML)])

    T33 = Array{ComplexF64}(I(2*ML))
    
    T23 = T33*Gd*Pd
    
    T13 = T23*Gv*Pv

    T = T13*G0

    MM = T13*S0 - T23*S + T33*S

    M11 = MM[1:ML,1:ML]
    M12 = MM[1:ML,ML+1:2ML]
    M21 = MM[ML+1:2ML,1:ML]
    M22 = MM[ML+1:2ML,ML+1:2ML]

    T11 = T[1:ML,1:ML]
    T12 = T[1:ML,ML+1:2ML]
    T21 = T[ML+1:2ML,1:ML]
    T22 = T[ML+1:2ML,ML+1:2ML]

    B[:,i] = ((M11+M12) - T12*inv(T22)*(M21+M22))*ones(ML)
end; plot(freqs/1e9,abs2.(B)'; label=["L=-1" "L= 0" "L= 1"])




# note: mistakenly swapping T33 for T13 in 
# MM = (T13.*S0).*ax - (T23.*S).*ax + (T33.*S).*ax
# produces current TM3d result