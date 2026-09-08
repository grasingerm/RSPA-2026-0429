using DelimitedFiles;
using Glob;
using LinearAlgebra

nrows = 257
ncols = 129

ϕm = -π / 2
ϕv = π / 2
k = 0.025

U(ks::Vector, ϕ0s::Vector, ϕs::Vector) = 0.5 * dot(ks, map(i -> (ϕs[i] - ϕ0s[i])^2, 1:length(ϕs)));
function U8(ks::Vector, ϕ0s::Vector, ϕ1, ϕ2, ϕ3)
    ϕs = [ϕ1; ϕ2; ϕ3; ϕ2; ϕ1; ϕ2; ϕ3; ϕ2]
    return U(ks, ϕ0s, ϕs)
end

function getnbrs(A, i, j)
    ni, nj = size(A)
    retval = []
    for di in -1:1, dj in -1:1
        if di == dj == 0
            continue
        end
        if 1 <= i+di <= ni && 1 <= j+dj <= nj
            val = A[i+di,j+dj]
            if !isnan(val) && !isinf(val)
                push!(retval, val)
            end
        end
    end
    return retval
end

energies = []
for (I, kmult) in enumerate([1e-2, 1, 1e2])
  energies_i = []
  for (J, f) in enumerate(readdir(glob"Phis_8-Vertex_Alpha*_SolnIdx-1.csv"))
      filter(x -> !(typeof(x) <: Real), readdlm(f, ','))
      data = map(x -> begin;
                   if x == "NaN"
                     NaN
                 else
                     x
             end; end, readdlm(f, ','));
      α0 = parse(Float64, split(split(split(f, "_")[end], '.')[1], '-')[end]) / 1000.0
      q0 = round(Int, α0*24.0/π)
      @show α = q0*π/24.0
      ks = fill(k, 8)
      ks[1:2:8] .*= sqrt(kmult)
      ks[2:2:8] ./= sqrt(kmult)
      ϕ0s = zeros(8)
      ϕ0s[1:2:8] .= ϕm
      ϕ0s[2:2:8] .= ϕv

      # nu, phi2, phi1, phi2, phi3
      υ = transpose(transpose(reshape(data[:, 1], nrows, ncols)))
      ϕ2s = transpose(transpose(reshape(data[:, 2], nrows, ncols)))
      ϕ1s = transpose(transpose(reshape(data[:, 3], nrows, ncols)))
      ϕ3s = transpose(transpose(reshape(data[:, 5], nrows, ncols)))
      for i=1:nrows, j=1:ncols
          if υ[i, j] == ϕ2s[i, j] == 0
              ϕ1s[i, j] = ϕ3s[i, j] = 0
          elseif υ[i, j] == 0
              ϕ1s[i, j] = (ϕ1s[i, j+1] + ϕ1s[i, j-1]) / 2
              ϕ3s[i, j] = (ϕ3s[i, j+1] + ϕ3s[i, j-1]) / 2
          elseif ϕ2s[i, j] == 0
              ϕ1s[i, j] = (ϕ1s[i+1, j] + ϕ1s[i-1, j]) / 2
              ϕ3s[i, j] = (ϕ3s[i+1, j] + ϕ3s[i-1, j]) / 2
          end
      end      
      Us = zeros(nrows, ncols)
      for i=1:nrows, j=1:ncols
          Us[i, j] = U8(ks, ϕ0s, ϕ1s[i, j], ϕ2s[i, j], ϕ3s[i, j])
      end
      Us = filter(!isnan, Us)
      push!(energies_i, maximum(Us))
  end
  push!(energies, maximum(energies_i))
end
@show energies

candidates = []
for (i, kmult) in enumerate([1e-2, 1, 1e2])
  for f in readdir(glob"Phis_8-Vertex_Alpha*.csv")
      println("================================================\n")
      @show f;
      @show filter(x -> !(typeof(x) <: Real), readdlm(f, ','))
      data = map(x -> begin;
                   if x == "NaN"
                     NaN
                 else
                     x
             end; end, readdlm(f, ','));
      α0 = parse(Float64, split(split(split(f, "_")[end-1], '.')[1], '-')[end]) / 1000.0
      q0 = round(Int, α0*24.0/π)
      @show α = q0*π/24.0
      ks = fill(k, 8)
      ks[1:2:8] .*= sqrt(kmult)
      ks[2:2:8] ./= sqrt(kmult)
      ϕ0s = zeros(8)
      ϕ0s[1:2:8] .= ϕm
      ϕ0s[2:2:8] .= ϕv

      # nu, phi2, phi1, phi2, phi3
      υ = transpose(transpose(reshape(data[:, 1], nrows, ncols)))
      ϕ2s = transpose(transpose(reshape(data[:, 2], nrows, ncols)))
      ϕ1s = transpose(transpose(reshape(data[:, 3], nrows, ncols)))
      ϕ3s = transpose(transpose(reshape(data[:, 5], nrows, ncols)))
      for i=1:nrows, j=1:ncols
          if υ[i, j] == ϕ2s[i, j] == 0
              ϕ1s[i, j] = ϕ3s[i, j] = 0
          elseif υ[i, j] == 0
              ϕ1s[i, j] = (ϕ1s[i, j+1] + ϕ1s[i, j-1]) / 2
              ϕ3s[i, j] = (ϕ3s[i, j+1] + ϕ3s[i, j-1]) / 2
          elseif ϕ2s[i, j] == 0
              ϕ1s[i, j] = (ϕ1s[i+1, j] + ϕ1s[i-1, j]) / 2
              ϕ3s[i, j] = (ϕ3s[i+1, j] + ϕ3s[i-1, j]) / 2
          end
      end      
      Us = zeros(nrows, ncols)
      for i=1:nrows, j=1:ncols
          Us[i, j] = U8(ks, ϕ0s, ϕ1s[i, j], ϕ2s[i, j], ϕ3s[i, j])
      end
      subcandidates = []
      for i=1:nrows, j=1:ncols
          nbrs = getnbrs(Us, i, j)
          if length(nbrs) > 0 && Us[i, j] <= minimum(nbrs)
              push!(subcandidates, (ϕ1s[i, j], ϕ2s[i, j], ϕ3s[i, j]))
          end
      end
      push!(candidates, [α, kmult, subcandidates])
  end
end
println("================================================\n")
println("CONFIGURATIONS = [")
for (i, configuration) in enumerate(candidates)
    if i < length(candidates)
        println("[", join(configuration[1:end-1], ", "), ", [", join(configuration[end], ", "), "]],")
    else
        println("[", join(configuration[1:end-1], ", "), ", [", join(configuration[end], ", "), "]]")
    end
end
println("]")
