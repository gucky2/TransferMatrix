using TransferMatrix
using Plots

const c0 = 299792458.

freqs = range(21.98e9,22.17e9,100);

M = 2; L = 1

coords = Coordinates(1,0.02; diskR=0.15);
modes = Modes(coords,M,L);


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

ax = axionModes(coords,modes)

gpm = GrandPropagationMatrix(freqs, modes, coords)

@time B = transfer_matrix_3d(gpm,dists,tilts,ax,freqs,waveguide=true)

graph1 = plot(freqs/1e9,abs2.(B)'; label=["M = 1, L= -1" "M = 1, L= 0" "M = 1, L= 1" "M = 2, L= -1" "M = 2, L= 0" "M = 2, L= 1"])
display(graph1)