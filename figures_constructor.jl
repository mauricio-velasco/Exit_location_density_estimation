#We include the library of estimation functions
include("estimation_functions.jl")
using GLMakie


#We implement the negative_laplacian, which is Dynkin operator for Brownian motion in space
function negative_laplacian(p)
    #this function implements the differential operator L, in this case the laplacian
    list = []
    vars = variables(p)
    for var in vars
        push!(list, differentiate(differentiate(p,var),var))
    end
    return((-1)*sum(list))
end

function Brownian_infinitesimal_operator(p)
    #this function implements the differential operator L for Brownian motion, which is negative ONE HALF of the laplacian
    list = []
    vars = variables(p)
    for var in vars
        push!(list, differentiate(differentiate(p,var),var))
    end
    return((-1/2)*sum(list))
end




#FIGURE 1b: Occupation densities for Brownian motion on the ball starting at [0.5,1.0]
#for various degrees of the CD kernel approx

n = 2 #dimension two
@polyvar x[1:n]
space_vars_vector = x
differential_operator_L = negative_laplacian #this function is defined above
degree_limit_choices = [2]
for degree_limit_d in degree_limit_choices
    initial_condition = [0.5,0.0]
    approximation_degree_r = degree_limit_d
    positive_offset = 2
    #We compute the normalized Christoffel Darboux function, which gives the density with respect to the equilibrium measure
    maxgap, NCD_func = compute_normalized_Christoffel_Function_occupation_measure_on_ball(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition,approximation_degree_r, positive_offset)

    #Next we want to evaluate the density with respect to the Lebesgue measure so we need to factor out the equilibrium density
    function Lebesgue_density_CD(a,b)
        #This function computes the estimated Lebesgue density of the occupation measure
        density_wrt_Equilibrium = NCD_func(x=>[b,a])
        m = 1/(pi*sqrt(1+a^2+b^2))
        return m*density_wrt_Equilibrium
    end

    #The following function sets up the evaluation
    function density_wrt_Equilibrium(; n=49)
        #This function evalutes
        X = range(-1.0,1.0,n)
        Y = range(-1.0,1.0,n)
        #Interesting notation...
        a = Lebesgue_density_CD.(X',Y)    
        return (X,Y,a)
    end

    #We evaluate the function at a large grid of points for the picture
    X,Y,Z = density_wrt_Equilibrium(; n=100)

    #contourf
    f = Figure(resolution= (1024,1024))
    ax = Axis(f[1, 1],aspect = 1)
    hm = contourf!(ax, X, Y, Z, levels=20)
    f
    save("CD_occupation_example_1b_dl"*string(degree_limit_d)*".png", f)
end
print(maxgap)
"""
#Surface
f = Figure(resolution= (1024,1024))
ax = Axis3(f[1, 1])
hm = surface!(ax, X, Y, Z)
f
f = Figure(resolution= (1024,1024))
ax = Axis3(f[1, 1])
hm = contour3d!(ax, X, Y, Z)
hm = surface!(ax, X, Y, Z, levels=10)
f

#Dropping a part
f = Figure(resolution= (1024,768))
ax = Axis(f[1, 1],aspect = 1)
hm = contourf!(ax, X, Y, Z, levels = 0.05:0.05:1, mode = :relative)
f
"""
#FIGURES 1 and 2: Exit location densities for Brownian motion on the sphere
#Figure 1: We fix an initial condition and Brownian motion in 2D.
n = 2 #dimension two
@polyvar x[1:n]
space_vars_vector = x
differential_operator_L = negative_laplacian #this function is defined above
initial_condition = [0.75,0.0]
#We change the degree_limit_d, to see how the CD estimation of the density changes
degree_limit_choices = [2,4,5,6]
colors = [:blue, :red, :green, :magenta]

f = Figure(resolution= (1024,768))
ax = Axis(f[1, 1])

for k in 1:4
    degree_limit_d = degree_limit_choices[k]
    approximation_degree_r = degree_limit_d
    current_color = colors[k]
    maxgap, normalizedChristoffel = compute_normalized_Christoffel_Function(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition,approximation_degree_r)
    thetas = range(-pi, pi, length=300)
    points = [[cos(theta),sin(theta)] for theta in thetas]
    y = [normalizedChristoffel(x=>point) for point in points]
    #lines!(ax, thetas, y, linewidth=3, color = :black)    
    lines!(ax, thetas, y, linewidth=2, color = current_color)    
end
f
save("CD_approx_example_1.png", f)

#Figure 2: We change the initial condition while keeping the total_degree fixed
n = 2 #dimension two
@polyvar x[1:n]
space_vars_vector = x
differential_operator_L = negative_laplacian #this function is defined above
degree_limit_d = 5

#We change the initial conditions with fixed degree to see how the CD estimation of the density changes
initial_condition_choices = [[0.0,0.0], [0.25,0], [0.5*cos(2pi/3), 0.5*sin(pi/3)], [0.75*cos(4pi/3), 0.75*sin(4*pi/3)]] 
colors = [:blue, :red, :green, :magenta]

f = Figure(resolution= (1024,768))
ax = Axis(f[1, 1])

for k in 1:4
    initial_condition = initial_condition_choices[k]
    approximation_degree_r = degree_limit_d
    current_color = colors[k]
    maxgap, normalizedChristoffel = compute_normalized_Christoffel_Function(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition,approximation_degree_r)
    thetas = range(-pi, pi, length=300)
    points = [[cos(theta),sin(theta)] for theta in thetas]
    y = [normalizedChristoffel(x=>point) for point in points]
    #lines!(ax, thetas, y, linewidth=3, color = :black)    
    lines!(ax, thetas, y, linewidth=2, color = current_color)    
end
save("CD_approx_example_2.png", f)
f



#FIGURES 3 and 4: Exit location densities for polynomial diffusion on the sphere
function Sq_bessel_operator(p)
    #this function implements the differential operator L, in this changes
    #for the two dimensional diffusion which is brownian in one component
    #and the squared bessel in the other
    list = []
    alpha = 1.0
    vars = variables(p)
    for var in vars
        push!(list, (-1/2)* differentiate(differentiate(p,var),var))
        push!(list, (-2)*var*differentiate(differentiate(p,var),var))
        push!(list, (-alpha)*differentiate(p,var))
    end
    return(sum(list))
end

#Figure 3: We fix an initial condition and Brownian motion in 2D.

n = 2 #dimension two
@polyvar x[1:n]
space_vars_vector = x
differential_operator_L = Sq_bessel_operator #this function is defined above
initial_condition = [0.0,0.0]
#We change the degree_limit_d, to see how the CD estimation of the density changes
#degree_limit_choices = [2,4,6,7]
degree_limit_choices = [2,4,5,6]

colors = [:blue, :red, :green, :magenta]

f = Figure(resolution= (1024,768))
ax = Axis(f[1, 1])

for k in 1:4
    degree_limit_d = degree_limit_choices[k]
    approximation_degree_r = degree_limit_d+2
    current_color = colors[k]
    maxgap, normalizedChristoffel = compute_normalized_Christoffel_Function(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition,approximation_degree_r)
    thetas = range(-pi, pi, length=300)
    points = [[cos(theta),sin(theta)] for theta in thetas]
    y = [normalizedChristoffel(x=>point) for point in points]
    #lines!(ax, thetas, y, linewidth=3, color = :black)    
    lines!(ax, thetas, y, linewidth=2, color = current_color)    
end
f
save("CD_approx_example_3.png", f)



#Figure 4: We change the initial condition while keeping the total_degree fixed

n = 2 #dimension two
@polyvar x[1:n]
space_vars_vector = x
differential_operator_L = Sq_bessel_operator #this function is defined above
degree_limit_d = 5
approximation_degree_r = degree_limit_d 


#We change the initial conditions with fixed degree to see how the CD estimation of the density changes
initial_condition_choices = [[0.0,0.0], [0,0.2], [0,0.5], [0,0.8]] 
colors = [:blue, :red, :green, :magenta]

f = Figure(resolution= (1024,768))
ax = Axis(f[1, 1])
degree_limit_d = 5

for k in 1:3
    initial_condition = initial_condition_choices[k]
    current_color = colors[k]
    maxgap, normalizedChristoffel = compute_normalized_Christoffel_Function(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition,approximation_degree_r)
    thetas = range(-pi, pi, length=300)
    points = [[cos(theta),sin(theta)] for theta in thetas]
    y = [normalizedChristoffel(x=>point) for point in points]
    #lines!(ax, thetas, y, linewidth=3, color = :black)    
    lines!(ax, thetas, y, linewidth=2, color = current_color)    
end
save("CD_approx_example_4.png", f)
f


#FIGURE 3b: Occupation densities for our polynomial diffusion on the ball starting at [0.0,0.25]
#for various degrees of the CD kernel approx

n = 2 #dimension two
@polyvar x[1:n]
space_vars_vector = x
differential_operator_L = Sq_bessel_operator #this function is defined above
degree_limit_choices = [3]
degree_limit_d = 3
for degree_limit_d in degree_limit_choices
    initial_condition = [0.0,0.25]
    approximation_degree_r = degree_limit_d+4
    positive_offset = 1
    #We compute the normalized Christoffel Darboux function, which gives the density with respect to the equilibrium measure
    maxgap, NCD_func = compute_normalized_Christoffel_Function_occupation_measure_on_ball(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition,approximation_degree_r, positive_offset)

    #Next we want to evaluate the density with respect to the Lebesgue measure so we need to factor out the equilibrium density
    function Lebesgue_density_CD(a,b)
        #This function computes the estimated Lebesgue density of the occupation measure
        density_wrt_Equilibrium = NCD_func(x=>[b,a])
        m = 1/(pi*sqrt(1+a^2+b^2))
        return m*density_wrt_Equilibrium
    end

    #The following function sets up the evaluation
    function density_wrt_Equilibrium(; n=49)
        #This function evalutes
        X = range(-1.0,1.0,n)
        Y = range(-1.0,1.0,n)
        #Interesting notation...
        a = Lebesgue_density_CD.(X',Y)    
        return (X,Y,a)
    end

    #We evaluate the function at a large grid of points for the picture
    X,Y,Z = density_wrt_Equilibrium(; n=100)

    #contourf
    f = Figure(resolution= (1024,1024))
    ax = Axis(f[1, 1],aspect = 1)
    hm = contourf!(ax, X, Y, Z, levels=20)
    f
    save("CD_occupation_example_3b_dl"*string(degree_limit_d)*".png", f)
    print("Maxgap_"*string(maxgap))    
end

g = Figure(resolution= (1024,1024))
ax = Axis3(g[1, 1])
surf = surface!(ax, X, Y, Z)
g
