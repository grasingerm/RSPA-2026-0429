using ScikitLearn
using Statistics
using DelimitedFiles;
using Glob;
using LinearAlgebra

@sk_import cluster: KMeans

using PythonPlot;

candidates = []

# Function to perform k-means for a range of k values
function evaluate_kmeans(X, k_range)
    nsamples = size(X, 1)
    results = Dict()
    
    for k in k_range
        if k > nsamples; continue; end
        # Create and fit K-means model
        model = KMeans(n_clusters=k, random_state=42)
        ScikitLearn.fit!(model, X)
        
        # Get cluster centers and labels
        centers = model.cluster_centers_
        labels = ScikitLearn.predict(model, X)
        
        # Calculate inertia (sum of squared distances to nearest centroid)
        inertia = model.inertia_
        
        # Store results
        results[k] = Dict(
            "centers" => centers,
            "labels" => labels,
            "inertia" => inertia
        )
    end
    
    return results
end

# Function to plot the elbow curve
function plot_elbow_curve(results, k_range, fname)
    Ks = sort(collect(keys(results)))
    inertias = [results[k]["inertia"] for k in Ks]
    
    p = plot(Ks, inertias)        
    xlabel("Number of clusters (k)")
    ylabel("Inertia")
    title("Elbow Method for Optimal k")
    savefig(fname)
end

# Function to visualize clusters for a specific k
function plot_clusters(X, results, k)
    centers = results[k]["centers"]
    labels = results[k]["labels"]
    
    # Create scatter plot of data points colored by cluster
    p = scatter(X[:, 1], X[:, 2])
    
    # Add cluster centers as 'X'
    scatter(centers[:, 1], centers[:, 2])

    return p
end

#=
rcparams = PyPlot.PyDict(PyPlot.matplotlib."rcParams");
font0 = Dict(
             "font.size" => 26,
             "axes.labelweight" => "bold",
             "axes.labelsize" => 24,
             "xtick.labelsize" => 20,
             "ytick.labelsize" => 20,
             "legend.fontsize" => 22
            );
merge!(rcparams, font0);

fontsize = 22;
#ticks = [-π, -3*π/4, -π/2, -π/4, 0, π/4, π/2, 3*π/4, π];
#ticklabels = ["\$-\\pi\$", "\$-3\\pi/4\$", "\$-\\pi/2\$","\$-\\pi/4\$","\$0\$",
#              "\$\\pi/4\$", "\$\\pi/2\$", "\$3\\pi/4\$","\$\\pi\$"];
=#
ticks2 = [-π, -π/2, 0, π/2, π];
ticklabels2 = ["\$-\\pi\$", "\$-\\pi/2\$","\$0\$",
              "\$\\pi/2\$","\$\\pi\$"];
ticks3 = [-π/2, -π/4, 0, π/4, π/2];
ticklabels3 = ["\$-\\pi/2\$", "\$-\\pi/4\$","\$0\$",
              "\$\\pi/4\$","\$\\pi/2\$"];

ϕm = -π
ϕv = π / 2
k = 0.025

U(ks::Vector, ϕ0s::Vector, ϕs::Vector) = 0.5 * dot(ks, map(i -> (ϕs[i] - ϕ0s[i])^2, 1:length(ϕs)));
function U6(ks::Vector, ϕ0s::Vector, ϕ1, ϕ2, ϕ3, ϕ4)
    ϕs = [ϕ1; ϕ2; ϕ3; ϕ4; ϕ3; ϕ2]
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
for (I, kmult) in enumerate([1e-4, 1e-2, 1, 1e2, 1e4])
  energies_i = []
  for (J, f) in enumerate(readdir(glob"Phis_6-Vertex*.csv"))
      filter(x -> !(typeof(x) <: Real), readdlm(f, ','))
      data = map(x -> begin;
                   if x == "NaN"
                     NaN
                 else
                     x
             end; end, readdlm(f, ','));
      ncols = nrows = round(Int, sqrt(size(data, 1)))
      α = parse(Float64, split(split(split(f, "_")[end], '.')[1], '-')[end]) / 1000.0
      ks = fill(k, 6)
      ϕ0s = if α <= π/3
          ks[[1, 4]] .*= sqrt(kmult)
          ks[[2, 3, 5, 6]] ./= sqrt(kmult)
          [ϕm; ϕv; ϕv; ϕm; ϕv; ϕv]
      else
          ks[[3, 5]] .*= sqrt(kmult)
          ks[[1, 2, 4, 6]] ./= sqrt(kmult)
          [ϕv; ϕv; ϕm; ϕv; ϕm; ϕv]
      end

      ϕ2s = transpose(transpose(reshape(data[:, 1], nrows, ncols)))
      ϕ3s = transpose(transpose(reshape(data[:, 2], nrows, ncols)))
      ϕ1s = transpose(transpose(reshape(data[:, 3], nrows, ncols)))
      ϕ4s = transpose(transpose(reshape(data[:, 6], nrows, ncols)))
      Us = zeros(nrows, ncols)
      for i=1:nrows, j=1:ncols
          Us[i, j] = U6(ks, ϕ0s, ϕ1s[i, j], ϕ2s[i, j], ϕ3s[i, j], ϕ4s[i, j])
      end
      Us = filter(!isnan, Us)
      push!(energies_i, maximum(Us))
  end
  push!(energies, maximum(energies_i))
end
@show energies

k_range = 2:20
for (i, kmult) in enumerate([1e-4, 1e-2, 1, 1e2, 1e4])
  #for (i, kmult) in enumerate([1e-4, 1e-2, 1, 1e2, 1e4])
  for f in readdir(glob"Phis_6-Vertex*.csv")
      println("================================================\n")
      @show f;
      @show filter(x -> !(typeof(x) <: Real), readdlm(f, ','))
      data = map(x -> begin;
                   if x == "NaN"
                     NaN
                 else
                     x
             end; end, readdlm(f, ','));
      ncols = nrows = round(Int, sqrt(size(data, 1)))
      @show α = parse(Float64, split(split(split(f, "_")[end], '.')[1], '-')[end]) / 1000.0
      ks = fill(k, 6)
      @show ϕ0s = if α <= π/3
          ks[[1, 4]] .*= sqrt(kmult)
          ks[[2, 3, 5, 6]] ./= sqrt(kmult)
          [ϕm; ϕv; ϕv; ϕm; ϕv; ϕv]
      else
          ks[[3, 5]] .*= sqrt(kmult)
          ks[[1, 2, 4, 6]] ./= sqrt(kmult)
          [ϕv; ϕv; ϕm; ϕv; ϕm; ϕv]
      end
      @show ks

      ϕ2s = transpose(transpose(reshape(data[:, 1], nrows, ncols)))
      ϕ3s = transpose(transpose(reshape(data[:, 2], nrows, ncols)))
      ϕ1s = transpose(transpose(reshape(data[:, 3], nrows, ncols)))
      ϕ4s = transpose(transpose(reshape(data[:, 6], nrows, ncols)))
      Us = zeros(nrows, ncols)
      for i=1:nrows, j=1:ncols
          Us[i, j] = U6(ks, ϕ0s, ϕ1s[i, j], ϕ2s[i, j], ϕ3s[i, j], ϕ4s[i, j])
      end
      min_xs = []
      min_ys = []
      min_Us = []
      subcandidates = []
      for i=1:nrows, j=1:ncols
          nbrs = getnbrs(Us, i, j)
          if length(nbrs) > 0 && Us[i, j] <= minimum(nbrs)
              push!(min_xs, ϕ2s[i, j])
              push!(min_ys, ϕ3s[i, j])
              push!(min_Us, Us[i, j])
              push!(subcandidates, (ϕ1s[i, j], ϕ2s[i, j], ϕ3s[i, j], ϕ4s[i, j]))
          end
      end
      prefix = split(basename(f), ".")[1] * "_U$i";
      # sort by energy
      p = sortperm(min_Us)
      min_xs .= min_xs[p]
      min_ys .= min_ys[p]
      min_Us .= min_Us[p]
      println("Local minima")
      for (x, y, U) in zip(min_xs, min_ys, min_Us)
          println("$x, $y, $U")
      end
      results = evaluate_kmeans(hcat(min_xs, min_ys), k_range)
      plot_elbow_curve(results, k_range, prefix*"_elbow-curve.png")
      for K in k_range
          if K > length(min_xs); continue; end
          fig = contourf(ϕ2s, ϕ3s, Us);
          ylabel("\$\\phi_3\$");
          xlabel("\$\\phi_2\$");
          yticks(ticks2, ticklabels2);
          xticks(ticks2, ticklabels2);
          cbar = colorbar(extend="neither");
          #cbar.set_ticklabels(ticklabels2);
          #cbar.ax.tick_params(labelsize=fontsize);
          #cbar.ax.orientation = "horizontal";
          clim(0.0, maximum(energies[i]))
          centers = results[K]["centers"]
          scatter(min_xs, min_ys)
          scatter(centers[:, 1], centers[:, 2])
          savefig(prefix * "_K-$K.pdf");
          clf();
          UUs = zeros(K)
          for j=1:K
              println("local stats, j = $j")
              @show idxs = results[K]["labels"] .== (j-1)
              @show Us_j = min_Us[idxs]
              @show UUs[j] = sum(Us_j) / length(Us_j)
              @show Umin = minimum(Us_j)
              @show Umax = maximum(Us_j)
              @show Umax - Umin
              @show Uavg = sum(Us_j) / length(Us_j)
              @show var = sum(map(u -> (u - Uavg)^2, Us_j)) / length(Us_j)
              @show std = sqrt(var)
          end
          println("global stats")
          @show Umin = minimum(UUs)
          @show Umax = maximum(UUs)
          @show Umax - Umin
          @show Umax / ks[end]
          @show Uavg = sum(UUs) / length(UUs)
          @show var = sum(map(u -> (u - Uavg)^2, UUs)) / length(UUs)
          @show std = sqrt(var)
      end
      println("================================================\n")
  end
  push!(candidates, [α, kmult, subcandidates])
end
println("================================================\n")
println("CONFIGURATIONS = [")
for configuration in candidates
    println("[", join(configuration, ", "), "]")
end
println("]")
