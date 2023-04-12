#We include the library of estimation functions
include("estimation_functions.jl")
using GLMakie

#FIGURES 1 and 2: Exit location densities for Brownian motion on the sphere

#We implement the negative_laplacian, Dynkin operator for Brownian motion in space
function negative_laplacian(p)
    #this function implements the differential operator L, in this case the laplacian
    list = []
    vars = variables(p)
    for var in vars
        append!(list, differentiate(differentiate(p,var),var))
    end
    return((-1)*sum(list))
end

#Figure 1: We fix an initial condition and Brownian motion in 2D.
n = 2 #dimension two
@polyvar x[1:n]
space_vars_vector = x
differential_operator_L = negative_laplacian #this function is defined above
initial_condition = [0.75,0.0]
#We change the degree_limit_d, to see how the CD estimation of the density changes
degree_limit_choices = [2,4,6,7]
colors = [:blue, :red, :green, :magenta]

f = Figure(resolution= (1024,768))
ax = Axis(f[1, 1],
title = "CD approximations of exit location density")


for k in 1:4
    degree_limit_d = degree_limit_choices[k]
    approximation_degree_r = degree_limit_d
    current_color = colors[k]
    normalizedChristoffel = compute_normalized_Christoffel_Function(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition,approximation_degree_r)
    thetas = range(-pi, pi, length=300)
    points = [[cos(theta),sin(theta)] for theta in thetas]
    y = [normalizedChristoffel(x=>point) for point in points]
    lines!(ax, thetas, y, linewidth=3, color = :black)    
    lines!(ax, thetas, y, linewidth=2, color = current_color)    
end
save("CD_approx.png", f)
f