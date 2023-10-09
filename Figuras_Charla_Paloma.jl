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
    # dX = (0,1)dt+dW(t) 
    # which is negative ONE HALF of the laplacian
    vars = variables(p)
    list = [(-1)*differentiate(p,x[2])]
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
    vars = variables(p)
    for var in vars
        push!(list, (-1/2)* differentiate(differentiate(p,var),var))
        push!(list, (-2)*var*differentiate(differentiate(p,var),var))
    end
    return(sum(list))
end



#EXAMPLE:
n = 2 #dimension two
@polyvar x[1:n]
space_vars_vector = x
differential_operator_L = Brownian_with_drift_infinitesimal_operator #this function is defined above
target_g = x[1]^4*x[2]^4 #This is the function whose boundary moments we wish to estimate
initial_location = [0.0,0.0]
target_functions_list = [target_g]
error_tolerance = 1e-8
maximum_allowed_degree_increases = 1
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



