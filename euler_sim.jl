using LinearAlgebra
using Random
using Pkg
using Distributions
using JSON


function circle_figura(a,b,r)
    theta = LinRange(0, 2*pi, 300)
    a.+r*cos.(theta), b.+r*sin.(theta)
end

function drift_0(x,y)
    return Array{Float64}([0.0,0.0])
end


function drift_1(x,y)
    return Array{Float64}([0.0,1.0])
end

function diffusion_1(x,y)
    return Matrix{Float64}([1 0 ; 0 1])
end

function diffusion_2(x,y)
    return Matrix{Float64}([1 0 ; 0 2*sqrt(y)])
end


function stopping_condition_func(x,y)
    return(1 - (x^2+y^2) < 0)
end


function euler_sde_simulation_2D( sde_as_dict, step_size_h, max_time_steps, stopping_condition_func)
    #Given an sde_as_dict
    #a step_size_h
    #a maximum number of time steps
    #a (pointwise) stopping condition 
    #return the number of steps before the condition is satisfied 
    # and the sequence of values of the process process during those times.
    current_location = sde_as_dict["initial_location"]
    current_time_step_index = 1
    b_vector = sde_as_dict["drift"]
    B_matrix = sde_as_dict["diffusion"]
    Xs = []
    Ys = []
    while stopping_condition_func(current_location[1], current_location[2])==false && current_time_step_index < max_time_steps 
        current_time_step_index +=1
        det_increment = step_size_h * b_vector(current_location[1], current_location[2])
        mean = Array{Float64}(zeros(2))
        C = step_size_h*Matrix{Float64}(I,2,2)
        normD = MvNormal(mean, C)  
        D = rand(normD,1)
        ito_increment = B_matrix(current_location[1], current_location[2])*D
        current_location = current_location + det_increment + ito_increment
        push!(Xs,current_location[1])
        push!(Ys,current_location[2])        
    end
    return current_time_step_index, Xs, Ys
end

function euler_sde_simulation_many_paths_2D(numpaths, sde_as_dict, step_size_h, max_time_steps, stopping_condition_func)
    arrayXs = []
    arrayYs = []
    for k in 1:num_paths
        T, Xs, Ys = euler_sde_simulation_2D(sde_as_dict, step_size_h, max_time_steps, stopping_condition_func)
        push!(arrayXs, [Xs])
        push!(arrayYs, [Ys])
    end
    return arrayXs, arrayYs
end


#Figure 1:
#We specify the stochastic process:
sde_as_dict = Dict(
    "name" => "Brownian_w_drift",
    "drift" => drift_1,
    "diffusion" => diffusion_1,
    "initial_location"=>Array{Float64}([0.0,0.5])
)

#Now we carry out and plot the simulation results
using Plots
step_size_h = 0.001
max_time_steps = 20000
num_paths = 100
ArrayXs, ArrayYs = euler_sde_simulation_many_paths_2D(num_paths, sde_as_dict, step_size_h, max_time_steps, stopping_condition_func)
allpaths_Dict = Dict("XPaths"=> ArrayXs, "YPaths"=> ArrayYs)
data_filename = "euler_paths.json"
open(data_filename, "w") do f
    JSON.print(f,allpaths_Dict)    
end


using Plots
allpaths_Dict = JSON.parsefile(data_filename)
ArrayXs = allpaths_Dict["XPaths"]
ArrayYs = allpaths_Dict["YPaths"]
name = sde_as_dict["name"]
Plots.plot(title = name, dpi=1500)
Plots.plot!(circle_figura(0,0,1),seriestype = [:shape], lw=0.5, c=:blue, linecolor= :black, legend = false, aspect_ratio = 1,fillalpha=0.2)
for k in eachindex(ArrayXs)
    Xs = ArrayXs[k]
    Ys = ArrayYs[k]
    Plots.plot!(Xs,Ys, label="", alpha = 0.8, lw=1.5)
end
initial_location = sde_as_dict["initial_location"]
Plots.plot!(circle_figura(initial_location[1],initial_location[2],0.03),seriestype = [:shape], lw=0.5, c=:red, linecolor= :black, legend = false, aspect_ratio = 1)

savefig("euler"*name*"_100.png")

#Next figure:

#Figure 2:
#We specify the stochastic process:
sde_as_dict = Dict(
    "name" => "Brownian",
    "drift" => drift_0,
    "diffusion" => diffusion_1,
    "initial_location"=>Array{Float64}([0.0,0.5])
)

#Now we carry out and plot the simulation results
using Plots
step_size_h = 0.001
max_time_steps = 20000
num_paths = 1000
ArrayXs, ArrayYs = euler_sde_simulation_many_paths_2D(num_paths, sde_as_dict, step_size_h, max_time_steps, stopping_condition_func)
allpaths_Dict = Dict("XPaths"=> ArrayXs, "YPaths"=> ArrayYs)
data_filename = "euler_paths.json"
open(data_filename, "w") do f
    JSON.print(f,allpaths_Dict)    
end


using Plots
allpaths_Dict = JSON.parsefile(data_filename)
ArrayXs = allpaths_Dict["XPaths"]
ArrayYs = allpaths_Dict["YPaths"]
name = sde_as_dict["name"]
Plots.plot(title = name, dpi=1500)
Plots.plot!(circle_figura(0,0,1),seriestype = [:shape], lw=0.5, c=:blue, linecolor= :black, legend = false, aspect_ratio = 1,fillalpha=0.2)
for k in eachindex(ArrayXs)
    Xs = ArrayXs[k]
    Ys = ArrayYs[k]
    Plots.plot!(Xs,Ys, label="", alpha = 0.8, lw=1.5)
end
initial_location = sde_as_dict["initial_location"]
Plots.plot!(circle_figura(initial_location[1],initial_location[2],0.03),seriestype = [:shape], lw=0.5, c=:red, linecolor= :black, legend = false, aspect_ratio = 1)

savefig("euler"*name*"_1000.png")

#Figure 3:
#We specify the stochastic process:
sde_as_dict = Dict(
    "name" => "Bessel",
    "drift" => drift_0,
    "diffusion" => diffusion_1,
    "initial_location"=>Array{Float64}([0.0,0.5])
)

#Now we carry out and plot the simulation results
using Plots
step_size_h = 0.001
max_time_steps = 20000
num_paths = 1000
ArrayXs, ArrayYs = euler_sde_simulation_many_paths_2D(num_paths, sde_as_dict, step_size_h, max_time_steps, stopping_condition_func)
allpaths_Dict = Dict("XPaths"=> ArrayXs, "YPaths"=> ArrayYs)
data_filename = "euler_paths.json"
open(data_filename, "w") do f
    JSON.print(f,allpaths_Dict)    
end


using Plots
allpaths_Dict = JSON.parsefile(data_filename)
ArrayXs = allpaths_Dict["XPaths"]
ArrayYs = allpaths_Dict["YPaths"]
name = sde_as_dict["name"]
Plots.plot(title = name, dpi=1500)
Plots.plot!(circle_figura(0,0,1),seriestype = [:shape], lw=0.5, c=:blue, linecolor= :black, legend = false, aspect_ratio = 1,fillalpha=0.2)
for k in eachindex(ArrayXs)
    Xs = ArrayXs[k]
    Ys = ArrayYs[k]
    Plots.plot!(Xs,Ys, label="", alpha = 0.8, lw=1.5)
end
initial_location = sde_as_dict["initial_location"]
Plots.plot!(circle_figura(initial_location[1],initial_location[2],0.03),seriestype = [:shape], lw=0.5, c=:red, linecolor= :black, legend = false, aspect_ratio = 1)

savefig("euler"*name*"_1000.png")

