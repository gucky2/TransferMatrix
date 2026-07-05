
using TransferMatrix
using Plots
using LinearAlgebra
using StaticArrays
using FFTW

const c0 = 299792458.

freqs = range(21.8e9,22.3e9,300);

M = 2; L = 1

coords = Coordinates(1,0.02; diskR=0.15);
modes = Modes(coords,M,L);


# @time gpm = GPM(freqs,distances,tilts,modes,coords);

# E0 = modes2field(ax,modes)[:,:,1]
# showField(E0,coords)

# k0 = 2π*freqs[1]/c0*sqrt(1)
# propagate!(E0,k0,coords,10e-3,deg2rad(0.5),0)
# showField(E0,coords)
# c1 = field2modes(E0,modes);

dists = [
    7.005317,
    7.161926,
    7.436722,
    7.144421,
    7.185010,
    7.209110,
    7.278833,
    7.169816,
    7.250541,
    7.214103,
    7.170475,
    7.245183,
    7.241939,
    7.191030,
    7.208307,
    7.300933,
    7.203299,
    7.265450,
    6.785361,
    7.310886,
]*1e-3

using Random
Random.seed!(3)
tilts = zeros(length(dists), 2)
for i in 1:length(dists)
    tilts[i,1] = 0.01 * randn()
    tilts[i,2] = 0.01 * randn()
end

# dists = [7.0]*1e-3
# tilts = zeros(length(dists), 2)
# tilts[1,1] = 0.1
# tilts[1,2] = 0.0


function G(ML,n1,n2)
    g = zeros(ComplexF64,2ML,2ML)

    g[1:ML,1:ML] += I(ML)*(n2+n1)/2n2
    g[ML+1:2ML,ML+1:2ML] += I(ML)*(n2+n1)/2n2

    g[ML+1:2ML,1:ML] += I(ML)*(n2-n1)/2n2
    g[1:ML,ML+1:2ML] += I(ML)*(n2-n1)/2n2

    return g
end

function propagate!(E0::Matrix{ComplexF64},k0::Number,coords::Coordinates,dz::Real)
    fft!(E0)
    @. E0 *= cis(-conj(sqrt(k0^2-coords.kR)*dz))
    ifft!(E0)
    
    return
end

function propagate!(E0::Matrix{ComplexF64},k0::Number,coords::Coordinates,dz::Real,
        tiltx::Real,tilty::Real)
    if dz != 0
        propagate!(E0,k0,coords,dz)
    end
    if tiltx + tilty != tiltx - tilty
        tiltField!(E0,k0,coords,tiltx,tilty)
    end

    return
end

function tiltField!(E0::Matrix{ComplexF64},k0::Number,coords::Coordinates,
        tiltx::Real,tilty::Real)
    

    E0 .*= cis.(-k0*(tiltx*coords.R .* cos.(coords.Φ) + tilty*coords.R .* sin.(coords.Φ)))

    return
end


function propagationCoeffs(freq::Real,
        distance::Real,tiltx::Real,tilty::Real,
        eps::Number,modes::Modes,coords::Coordinates)

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


function transfer_matrix_3d(distances,tilts,ax,f,modes,coords; eps=24.0,tand=0.,nm=1e30,thickness=1e-3)
    M = modes.M; L = modes.L; ML = M*(2L+1)
    
    RB = Array{ComplexF64}(undef,2,ML)

    ϵ  = eps*(1.0-1.0im*tand); nd = sqrt(ϵ); nm = complex(nm); ϵm = nm^2
    A  = 1-1/ϵ; A0 = 1-1/ϵm

    G0 = G(ML,nm,1)
    Gv = G(ML,1,nd)
    Gd = G(ML,nd,1)
    
    S  = A/2  * I(2ML) * diagm([ax; ax])
    S0 = A0/2 * I(2ML) * diagm([ax; ax])

    T  = Matrix{ComplexF64}(I, 2*ML, 2*ML)
    MM = Matrix{ComplexF64}(I, 2*ML, 2*ML)*0

    W  = Matrix{ComplexF64}(undef, 2*ML, 2*ML)
    TW = Matrix{ComplexF64}(undef, 2*ML, 2*ML)



    Pd_ = cispi(+2*f*nd*1e-3/c0)
    Pd = diagm([fill(conj(Pd_),ML); fill(Pd_,ML)])

    Pv0_temp = propagationCoeffs(f,0.0,-deg2rad(tilts[1,1]),-deg2rad(tilts[1,2]),1.0,modes,coords)
    Pv0 = Matrix{ComplexF64}(I,2*ML,2*ML)
    Pv0[1:ML,1:ML] .= Pv0_temp; Pv0[ML+1:2ML,ML+1:2ML] .= inv(Pv0_temp)

    T0 = copy(T) * Pv0

    # iterate in reverse order to sum up MM in single sweep (thx david)
    
    for i in Iterators.reverse(eachindex(distances))
        Pvd_temp = propagationCoeffs(f,distances[i],0.0,0.0,1.0,modes,coords)
        Pvt_temp = propagationCoeffs(f,0.0,deg2rad(tilts[i,1]),deg2rad(tilts[i,2]),1.0,modes,coords)
        Pvd = Matrix{ComplexF64}(I,2*ML,2*ML)
        Pvd[1:ML,1:ML] .= Pvd_temp; Pvd[ML+1:2ML,ML+1:2ML] .= inv(Pvd_temp)
        Pvt = Matrix{ComplexF64}(I,2*ML,2*ML)
        Pvt[1:ML,1:ML] .= Pvt_temp; Pvt[ML+1:2ML,ML+1:2ML] .= inv(Pvt_temp)
    

        MM += T0 * S                              # 

        T0 *= Gd * Pd                         # T = Gd*Pd
            
        MM -= T0 * S                              # 

        T0 *= Gv * Pvd * Pvt                         # T *= Gv * Pv

        if i == 1
            MM += T0 * S0                         # 
            T0 *= G0                              # T *= G0
        end
    end

    M11 = MM[1:ML,1:ML]
    M12 = MM[1:ML,ML+1:2ML]
    M21 = MM[ML+1:2ML,1:ML]
    M22 = MM[ML+1:2ML,ML+1:2ML]

    T12 = T0[1:ML,ML+1:2ML]
    T22 = T0[ML+1:2ML,ML+1:2ML]

    RB = ((M11+M12) - T12*inv(T22)*(M21+M22))*ones(ML)

    return RB
end


ax = axionModes(coords,modes)
B = zeros(ComplexF64,M*(2L+1),length(freqs))

@time for i in eachindex(freqs)
    B[:,i] .= transfer_matrix_3d(dists,tilts,ax,freqs[i],modes,coords)
end

# abs2.(propagationCoeffs(freqs[500],7e-3,0,0,1.0,modes,coords))
# abs2.(propagationCoeffs(freqs[500],7e-3,deg2rad(0.1),0,1.0,modes,coords))

# st = propagationCoeffs(freqs[500],0,deg2rad(1),deg2rad(0),1.0,modes,coords)

graph1 = plot(freqs/1e9,abs2.(B)'; label=["M = 1, L= -1" "M = 1, L= 0" "M = 1, L= 1" "M = 2, L= -1" "M = 2, L= 0" "M = 2, L= 1"])
display(graph1)
# plot(freqs/1e9, sum(abs2.(B), dims=1)')

# %% 

# filename = "BlockMatrix_20_disk_random_tilt_2e-2degree.svg"
# filepath = joinpath(@__DIR__, "Plots_for_Meeting/")
# filepath = joinpath(filepath, filename)

# savefig(filepath)

# %%

B = zeros(ComplexF64,M*(2L+1),length(freqs)); ML = M*(2L+1)
# B = zeros(ComplexF64,length(freqs))
M = 1; L = 1

modes = Modes(coords,M,L);
ax = axionModes(coords,modes)
tiltx = deg2rad(0)
tilty = deg2rad(0.1)

eps = 24.; tand = 0; nm = 1e15
ϵ  = eps*(1.0-1.0im*tand); nd = sqrt(ϵ); nm = complex(nm); ϵm = nm^2
A  = 1-1/ϵ; A0 = 1-1/ϵm

Id = Matrix{ComplexF64}(I, ML, ML)

G0 = G(ML,nm,1)
Gv = G(ML,1,nd)
Gd = G(ML,nd,1)





S  =  A/2*I(2ML)*diagm([ax; ax])
S0 = A0/2*I(2ML)*diagm([ax; ax])

@time for i in eachindex(freqs)
    # T03 = G2*P2*G1*P1*G0*P0
    # MM = T13*S0+T23*S1+T33*S2

 
    Pv_temp = propagationCoeffs(freqs[i],7e-3,tiltx,tilty,1.0,modes,coords)
    Pv0_temp = propagationCoeffs(freqs[i],0,-tiltx,-tilty,1.0,modes,coords)
    
    Pv = Matrix{ComplexF64}(I,2*ML,2*ML)
    Pv0 = Matrix{ComplexF64}(I,2*ML,2*ML)

    Pv[1:ML,1:ML] .= Pv_temp; Pv[ML+1:2ML,ML+1:2ML] .= inv(Pv_temp)
    Pv0[1:ML,1:ML] .= Pv0_temp; Pv0[ML+1:2ML,ML+1:2ML] .= inv(Pv0_temp)

    Pd_ = cispi(+2*freqs[i]*nd*1e-3/c0)
    Pd = diagm([fill(conj(Pd_),ML); fill(Pd_,ML)])
    

    T33 = Matrix{ComplexF64}(I, 2*ML, 2*ML) * Pv0
 
    T23 = T33 * Gd * Pd
 
    T13 = T23 * Gv * Pv

    T03 = T13 * G0; T = T03

    MM = T13 * S0 + (-T23 + T33) * S

    M11 = MM[1:ML,1:ML]
    M12 = MM[1:ML,ML+1:2ML]
    M21 = MM[ML+1:2ML,1:ML]
    M22 = MM[ML+1:2ML,ML+1:2ML]

    T11 = T[1:ML,1:ML]
    T12 = T[1:ML,ML+1:2ML]
    T21 = T[ML+1:2ML,1:ML]
    T22 = T[ML+1:2ML,ML+1:2ML]

    B[:,i] = ((M11+M12) - T12*inv(T22)*(M21+M22))*ones(ML)

end; 

plot(freqs/1e9,abs2.(B)'; label=["L2=-1" "L2= 0" "L2= 1"])

# %%

# filename = "Hardcode_1Disk_no_tilt.svg"
# filepath = joinpath(@__DIR__, "Plots_for_Meeting/")
# filepath = joinpath(filepath, filename)
# savefig(filepath)


# note: mistakenly swapping T33 for T13 in 
# MM = (T13.*S0).*ax - (T23.*S).*ax + (T33.*S).*ax
# produces current TM3d result


# %%

using BoostFractor
using Distributed

function tilt!(sbdry::SetupBoundaries,deg)
    ndisk = Int((length(sbdry.distance)-2)/2)

    # tilts = deg2rad(deg)*(2*rand(2,ndisk).-1)
    tilts = deg

    fill!(sbdry.relative_tilt_x,0.); fill!(sbdry.relative_tilt_y,0.)

    sbdry.relative_tilt_x[2] = deg2rad(tilts[1,1])
    sbdry.relative_tilt_y[2] = deg2rad(tilts[1,2])
    sbdry.relative_tilt_x[end] = -deg2rad(tilts[end,1])
    sbdry.relative_tilt_y[end] = -deg2rad(tilts[end,2])

    for i in 2:ndisk
        sbdry.relative_tilt_x[2i] = deg2rad(tilts[i,1])
        sbdry.relative_tilt_y[2i] = deg2rad(tilts[i,2])
    end

    return
end

begin
    
    # Coordinate System
    dx = 0.02
    coords = SeedCoordinateSystem(X = -0.5:dx:0.5, Y = -0.5:dx:0.5)
    
    diskR = 0.15
    
    # SetupBoundaries (note that this expects the mirror to be defined explicitly as a region)
    epsilon = 24
    eps = Array{Complex{Float64}}([NaN, 1,epsilon,1,epsilon,1,
        epsilon,1,epsilon,1,epsilon,1,epsilon,1,epsilon,1,
        epsilon,1,epsilon,1,epsilon,1,epsilon,1,epsilon,1,
        epsilon,1,epsilon,1,epsilon,1,epsilon,1,epsilon,1,
        epsilon,1,epsilon,1,epsilon,1])
    distance = [0.0, 7.005317, 1.0,
                            7.161926, 1.0,
                            7.436722, 1.0,
                            7.144421, 1.0,
                            7.185010, 1.0,
                            7.209110, 1.0,
                            7.278833, 1.0,
                            7.169816, 1.0,
                            7.250541, 1.0,
                            7.214103, 1.0,
                            7.170475, 1.0,
                            7.245183, 1.0,
                            7.241939, 1.0,
                            7.191030, 1.0,
                            7.208307, 1.0,
                            7.300933, 1.0,
                            7.203299, 1.0,
                            7.265450, 1.0,
                            6.785361, 1.0,
                            7.310886, 1.0,
                            0.0]*1e-3

    sbdry = SeedSetupBoundaries(coords, diskno=20, distance=distance, epsilon=eps)

    # eps = Array{Complex{Float64}}([NaN,1,epsilon,1])
    # distance = [0.0, 7, 1.0, 0.0]*1e-3
    # sbdry = SeedSetupBoundaries(coords, diskno=1, distance=distance, epsilon=eps)
    
    # Initialize modes
    
    Mmax = 2
    Lmax = 1
    modes_BF = SeedModes(coords, ThreeDim=true, Mmax=Mmax, Lmax=Lmax, diskR=diskR)
    
    #  Mode-Vector defining beam shape to be reflected on the system
    m_reflect = zeros(Mmax*(2*Lmax+1))
    m_reflect[Lmax+1] = 1.0
end

tilt!(sbdry, tilts)

df = 0.01*1e9
# frequencies = 21.98e9:df:22.26e9
frequencies = freqs

# We will build a 3-dim array [reflection / boost factor, mode-vector, frequency ]
# The following function appends to the last dimension
zcat(args...) = cat(dims = 3, args...)

# Sweep over frequency
@time EoutModes0 = @sync @distributed (zcat) for f in frequencies    
    println("Frequency: $f")
    boost, refl = transformer(sbdry,coords,modes_BF; reflect=m_reflect, prop=propagator,diskR=diskR,f=f)
    transpose([boost  refl])
end;

#%%
graph = plot(freqs/1e9,abs2.(B)', title="20 Disk, random tilt", label=["M = 1, L= -1" "M = 1, L= 0" "M = 1, L= 1" "M = 2, L= -1" "M = 2, L= 0" "M = 2, L= 1"])
for i in 1:(modes_BF.M*(modes_BF.L*2+1))
    global graph = plot!(frequencies/1e9, abs2.(EoutModes0[1,i,:]), label="direct M=$i", linestyle=:dash)  
end

graph2 = plot(freqs/1e9, zeros(length(freqs)), title="BoostFractor - TransferMatrix", label="", color = "black")

for i in 1:(modes_BF.M*(modes_BF.L*2+1))
    global graph2 = plot!(frequencies/1e9, abs2.(EoutModes0[1,i,:])-abs2.(B[i,:]), label="M=$(div(i-1,2*Lmax+1)+1), L=$(mod(i-1,2*Lmax+1)-Lmax)", linestyle=:dash)
end
plot(graph, graph2, layout=Plots.grid(2,1, heights=[0.8, 0.2]))


# %%

filename = "Comparing_BF_TM_20_disks_random_tilt3.svg"
filepath = joinpath(@__DIR__, "Plots_for_Meeting/")
filepath = joinpath(filepath, filename)
savefig(filepath)