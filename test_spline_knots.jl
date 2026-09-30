using TransferMatrix
using Plots
using LinearAlgebra, SpecialFunctions, FunctionZeros, Interpolations

const c0 = 299792458.



freqs = range(21.98e9,22.17e9,100);

M = 1; L = 1

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

ax = axionModes(coords,modes)
tilts = zeros(length(dists), 2)

n2 = 5
n1_values = 5:5:100

graph1 = plot(
    xlabel="Frequency [GHz]",
    ylabel="Total boost factor |B|^2",
    title="Boost-factor convergence with distance spline knots",
    legend=:outerright,
)

for n1 in n1_values
    @time gpm = GrandPropagationMatrix(freqs, modes, coords, n1, n2)
    @time B = transfer_matrix_3d(gpm, dists, ax, freqs; waveguide=true)
    total_boostfactor = vec(sum(abs2, B; dims=1))
    plot!(graph1, freqs ./ 1e9, total_boostfactor; label="n1 = $n1")
end

display(graph1)