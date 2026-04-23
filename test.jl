
using TransferMatrix
using Plots
using BenchmarkTools


# freqs = (22:0.01:22.05)*1e9
freqs = range(21.98e9,22.26e9,50);
distances = (6.5:0.25:7.5).*1e-3
tilts = range(-deg2rad(0.01),deg2rad(0.01),11)

coords = Coordinates(1,0.02; diskR=0.15);
modes = Modes(coords,3,2);


@time gpm = GPM(freqs,distances,tilts,modes,coords);


for i in 0:5
    degmax = i*0.001
    tiltsx = [0; deg2rad(degmax)*(2*rand(length(dists)).-1)] 
    tiltsy = [0; deg2rad(degmax)*(2*rand(length(dists)).-1)] 

    @time RB = transfer_matrix_3d(Dist,dists,tiltsx,tiltsy,gpm,freqs;);

    B = [abs2.(RB[2,i,:]) for i in axes(RB,2)]
    display(plot(freqs/1e9,B; legend=false))
end



E0 = modes2field(ax,modes)[:,:,1]
showField(E0,coords)

k0 = 2π*freqs[1]/c0*sqrt(1)
propagate!(E0,k0,coords,10e-3,deg2rad(0.5),0)
showField(E0,coords)
c1 = field2modes(E0,modes);

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



# dists = [1.00334,
#         6.94754,
#         7.1766,
#         7.22788,
#         7.19717,
#         7.23776,
#         7.07746,
#         7.57173,
#         7.08019,
#         7.24657,
#         7.21708,
#         7.18317,
#         7.13025,
#         7.2198,
#         7.45585,
#         7.39873,
#         7.15403,
#         7.14252,
#         6.83105,
#         7.42282,]*1e-3


f = 22.025e9; ω = 2π*f; λ = c0/f
eps = complex(1)
k0 = 2π*f/c0*sqrt(eps)

# coords = Coordinates(1,λ/2; diskR=0.15);
coords = Coordinates(1,0.02; diskR=0.15);

m = 2; l = 0
modes = Modes(coords,m,l);


# freqs = 22.0e9

# @time p = propagationMatrix(freqs,collect(range(1e-3,10e-3,10)),1.0,modes,coords);
# @time gpm = GPM(freqs,collect(range(1e-3,10e-3,11)),modes,coords; eps=24.0);
@time gpm = GPM(freqs,collect(range(1e-3,10e-3,10)),modes,coords; eps=24.0);




# RB = transfer_matrix_3d(Dist,dists,gpm,ax;);






B = [abs2.(RB[:,2,i,1]) for i in 1:m]
display(plot(freqs/1e9,B,title="Boost 3d, m_max = $m, l_max = $l",label=["m=1" "m=2" "m=3"]))


