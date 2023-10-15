#We include the library of estimation functions
include("estimation_functions.jl")
using GLMakie

#The figures will show the behavior of Brownian motion on the disc
function Brownian_infinitesimal_operator(p)
    #this function implements the differential operator L for Brownian motion, which is negative ONE HALF of the laplacian
    list = []
    vars = variables(p)
    for var in vars
        push!(list, differentiate(differentiate(p,var),var))
    end
    return((-1/2)*sum(list))
end


function Brownian_with_drift_infinitesimal_operator(p)
    # this function implements the differential operator L for Brownian motion with drift:
    # dX = (0,2)dt+dW(t) 
    # which is negative ONE HALF of the laplacian
    vars = variables(p)
    list = [(-2)*differentiate(p,x[2])]
    for var in vars
        push!(list, (-1/2)*differentiate(differentiate(p,var),var))
    end
    return(sum(list))
end


function Sq_bessel_infinitesimal_operator(p)
    #this function implements the differential operator L, in this changes
    #for the two dimensional diffusion which is brownian in one component
    #and the squared bessel in the other
    # dX_1 = dW_1
    # dX_2 = dt + 2 \sqrt{X_2}dW_2 
    list = [-differentiate(p,x[2])]
    push!(list, (-1/2)*differentiate(differentiate(p,x[1]),x[1]))
    push!(list, -2*x[2]*differentiate(differentiate(p,x[2]),x[2]))
    return(sum(list))
end


"""
#EXAMPLE:
n = 2 #dimension two
@polyvar x[1:n]
space_vars_vector = x
differential_operator_L = Brownian_with_drift_infinitesimal_operator #this function is defined above
target_g = x[1]^4*x[2]^4 #This is the function whose boundary moments we wish to estimate
initial_location = [0.0,0.0]
target_functions_list = [target_g]
error_tolerance = 1e-8
maximum_allowed_degree_increases = 7
lower_bounds, upper_bounds = compute_certified_exit_location_moments(space_vars_vector, target_functions_list, differential_operator_L, initial_location, error_tolerance, maximum_allowed_degree_increases)


approximation_degree_r = 8
#Next we create the optimization model
upper_or_lower = "upper_bound"
upper_bounds = compute_moments_array_stopping_time_unit_sphere(space_vars_vector, target_functions_list, differential_operator_L, initial_location, approximation_degree_r, upper_or_lower)

upper_or_lower = "lower_bound"
lower_bounds = compute_moments_array_stopping_time_unit_sphere(space_vars_vector, target_functions_list, differential_operator_L, initial_location, approximation_degree_r, upper_or_lower)

upper_bounds-lower_bounds

target_functions_list = [x[1],x[1]*x[2]]
maximum([degree(p) for p in target_functions_list])

#Desired functionality:
#(1) compute_certified_exit_location_moments function: 
    # Given a list of polynomials compute their expected values with respect to the boundary measure.
    # with error guaranteed below a given treshold

#(2) compute_certified_exit_density_estimation: 
    # Computes the normalized Christoffel Darboux kernel using certified moments.

#(3) compute_certified_occupation_moments: 

#(4) compute_certified_occupation_density_estimation: 
"""


"""
#FIGURE 1: How does the error behave as we increase the allowed degree of the sum-of-square multipliers?
#DATA CONSTRUCTION:
#iterator for the three cases:
operators_dict=Dict("Brownian" => Brownian_infinitesimal_operator, "Brownian_drift"=> Brownian_with_drift_infinitesimal_operator, "Bessel"=>Sq_bessel_infinitesimal_operator)
resultsDict = Dict()
#Computation loop
for (operator_name, infinitesimal_operator) in operators_dict
    rs_array = [j for j in range(6,16)]
    lower_bounds_array = []
    upper_bounds_array = []
    errors_array = []
    #We compute data for the Brownian motion starting at (0,0.5)
    n = 2 #dimension two
    @polyvar x[1:n] 
    space_vars_vector = x #ambient space variables
    differential_operator_L = infinitesimal_operator #this function is defined above
    target_g = x[1]^2*x[2]^4 #This is the function whose boundary moments we wish to estimate
    initial_location = [0.0,0.5]
    target_functions_list = [target_g]

    for rvalue in rs_array
        approximation_degree_r = rvalue
        upper_or_lower = "upper_bound"
        upper_bounds = compute_moments_array_stopping_time_unit_sphere(space_vars_vector, target_functions_list, differential_operator_L, initial_location, approximation_degree_r, upper_or_lower)
        upper_or_lower = "lower_bound"
        lower_bounds = compute_moments_array_stopping_time_unit_sphere(space_vars_vector, target_functions_list, differential_operator_L, initial_location, approximation_degree_r, upper_or_lower)
        push!(lower_bounds_array, lower_bounds[1])
        push!(upper_bounds_array, upper_bounds[1])
        push!(errors_array,upper_bounds[1]-lower_bounds[1])
    end
    resultsDict[operator_name] = [lower_bounds_array, upper_bounds_array, errors_array]
end
#Data gets saved:
open("moment_estimates.json", "w") do f
    JSON.print(f,resultsDict)    
end

#Individual graph (for aesthetics improvement)
j = JSON.parsefile("moment_estimates.json")
X = rs_array
operator_names = collect(keys(operators_dict))
name = operator_names[3]
Y = j[name][3]
scatter(X,log.(Y),dpi=1000, label = "Operador_")

#Automatic plotting
#Load the data (this is separated from data computation to allow for several attempts
#in trying to find the clearest plot of the data without recomputing it)
Plots.plot(dpi=1500)
using Plots
j = JSON.parsefile("moment_estimates.json")
#Iterate over the different operatores
for name in keys(operators_dict)
    X = rs_array
    Y = j[name][3]
    Plots.plot!(X,log.(Y),marker = (:circle,5),label = name,xticks=rs_array)
end
title!("Log plot of error bound for EE[x^2y^4]")
savefig("log_error.png")
#xlabel!("Approximation degree")
"""

#
#FIGURE 2: Exit Location densities:
#As before we first compute the data in a json file and then reload it for plotting.
#First, we specify scenarios in the following subroutine:
operators_dict=Dict("Brownian" => Brownian_infinitesimal_operator, "Brownian_drift"=> Brownian_with_drift_infinitesimal_operator, "Bessel"=>Sq_bessel_infinitesimal_operator)
function chosen_scenarios_dict()
    #Each scenario is specified by its name (which must coincide with those in the operators_dict above)
    #and by the sequences of approximation degrees used.
    #Other parameters, such as error_tolerance_bound
    scenarios_dict=Dict(
        "Brownian_drift"=>[6,8],
        "Bessel"=>[6,8]) 
    """scenarios_dict=Dict(
        "Brownian" => [6,8],
        "Brownian_drift"=> [6,8],
        "Bessel"=>[6,8]
        )"""         
    """scenarios_dict=Dict(
        "Brownian" => [6,8,10,12], 
        "Brownian_drift"=> [6,8])"""
    return scenarios_dict
end

function scenario_string_name(name, degree_limit_d)
    density_name = name*"_"*string(degree_limit_d)
    return density_name
end

sc_dict = chosen_scenarios_dict()
[@assert(key in keys(operators_dict)) for key in keys(sc_dict)]
#operator_names = collect(keys(operators_dict))
#name = operator_names[3]

#For computation we load the existing data file and complete it...
using JSON
data_filename = "exit_densities.json"
densities_dict = JSON.parsefile(data_filename)
#Now we do the computations.
#First we set the global variables
n = 2 #dimension two
@polyvar x[1:n]
space_vars_vector = x
initial_location = [0.0,0.5]
error_tolerance = 1e-6
#Grid in which we will sample the densities for later plotting (note that the obtained density 
#approximation is a polynomial and that this discretization is only for plotting purposes)
thetas = [t for t in range(-pi/2, 3*pi/2, length=300)]
points = [[cos(theta),sin(theta)] for theta in thetas]
#Now we iterate
Tot_maxgap = Inf #This will store the largest observed error among all moment estimations.
worst_moment_condition_number = 0
moments_condition_number_array = []
for (name, allowed_degrees_d) in sc_dict
    #Specify the relevant differential operator
    differential_operator_L = operators_dict[name] #this function is defined above
    for degree_limit_d in allowed_degrees_d
        moments_condition_number, maxgap, normalizedChristoffel = compute_certified_exit_density_estimation(degree_limit_d,space_vars_vector, differential_operator_L, initial_location, error_tolerance)
        push!(moments_condition_number_array,moments_condition_number)
        #Next we compute the approximation density at points on the circle
        density_name = scenario_string_name(name, degree_limit_d)
        y = [normalizedChristoffel(x=>point) for point in points]
        #and write the result in our densities_dict (possibly overriding previous computations)
        densities_dict[density_name] = y
        if worst_moment_condition_number < moments_condition_number
            worst_moment_condition_number = moments_condition_number
        end
        if maxgap < Tot_maxgap
            Tot_maxgap = maxgap
        end    
    end
end
print("Largest observed gap is "*string(Tot_maxgap)*"\n")
print("Worst moment condition number is "*string(worst_moment_condition_number)*"\n")
print(string(moments_condition_number_array))
@assert(Tot_maxgap < error_tolerance)
#Finally, we write the data to disk
open(data_filename, "w") do f
    JSON.print(f,densities_dict)    
end

"""
#degree_limit_choices = [2,4,5,6]
degree_limit_choices = [2,4,5,6]
length(degree_limit_choices)
colors = [:blue, :red, :green, :magenta] Should we choose colors?
"""

#Now we plot the result
using Plots
densities_dict = JSON.parsefile(data_filename)

#We specify the figures we wish to plot via a dictionary specified in the following function,
function chosen_pictures_dict()
    #Each picture is specified by its filename 
    #and a dictionary of parameters specifying the operator name and the approximation_degrees we wish to plot.
    pictures_dict=Dict(
        "Exit_location_density_1" => Dict("name"=>"Brownian", "desired_degrees"=>[4,6,8]), 
        "Exit_location_density_2" => Dict("name"=>"Brownian_drift", "desired_degrees"=> [6,8]),
        "Exit_location_density_3" => Dict("name"=>"Bessel", "desired_degrees"=> [6,8])
        )
    #This function also verifies that the desired data exists in our json file with data
    for (key,value) in pictures_dict
        @assert(value["name"] in keys(operators_dict))
        for k in value["desired_degrees"]
            @assert(scenario_string_name(value["name"], k) in keys(densities_dict), scenario_string_name(value["name"], k)*" not present in computed densities.")
        end
    end
    return pictures_dict
end
#We verify that we have data for the desired scenariosi
pc_dict = chosen_pictures_dict()
xlabels_array = [round(x,sigdigits=2) for x in range(-pi/2, 3*pi/2, length=10)]
using Plots
for (picture_filename, specification_dict) in pc_dict
    Plots.plot(title = "Exit location densities", dpi=1500, xticks=xlabels_array)
    name = specification_dict["name"]
    for k in specification_dict["desired_degrees"]
        data_label = scenario_string_name(name, k) 
        y = densities_dict[data_label]
        y = Vector{Float32}(y)
        Plots.plot!(thetas, y, label = data_label, linewidth = 2)
    end
    savefig(picture_filename*".png")
end


"""
#Trying out rotation substitution
rotation_angle = pi/4
alpha = rotation_angle
# apply this rotation for every polynomial p
p = 1.0*x[3]^0
q = subs(p,x[1]=>cos(alpha)*x[1]-sin(alpha)*x[2],x[2]=>sin(alpha)*x[1]+cos(alpha)*x[2])
"""

n = 2 #dimension two
@polyvar x[1:n]
space_vars_vector = x
degree_limit_d = 4
create_monomial_basis_on_sphere(space_vars_vector, degree_limit_d)

typeof(1.0*x[1]^0+0.0)

basis_as_List = create_monomial_basis_on_sphere(space_vars_vector, degree_limit_d)
alpha = rotation_angle
rotated_basis_as_List = [subs(p,x[1]=>cos(alpha)*x[1]-sin(alpha)*x[2],x[2]=>sin(alpha)*x[1]+cos(alpha)*x[2]) for p in basis_as_List]
basis_as_List = rotated_basis_as_List
q = basis_as_List[1]
M = monomials(q^2)

p = x[1]+im*x[2]
q = x[1]-im*x[2]
k=3
h = p^2
(p^k+q^k)/2
real(h.a)
h.x
Polynomial(real(h.a),h.x)



MonomialVector(p.variables,Z)


for k in 1:degree_limit_d
    print(k)
    cos_part = real((p^k+q^k)/2)
    #push!(basis_as_List, cos_part)
    sin_part = real((p^k+q^k)/(2*im))
    #push!(basis_as_List, sin_part)
end
