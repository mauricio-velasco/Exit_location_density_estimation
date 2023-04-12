using DynamicPolynomials
using CSDP
using SumOfSquares
using JuMP

function create_model_moments_stopping_time_unit_sphere(space_vars_vector, approximation_degree_r, initial_condition, target_g, differential_operator_L, upper_or_lower)
    #Given:
    # ambient space variables space_vars_vector
    # an approximation degree r
    # an initial location vector (y_1,...,y_n)
    # a target polynomial on the ambient space variables g
    # an implementation of the differential operator L
    # a string choice of either upper_bound or lower_bound
    # Return a PolyJump model which gives the desired upper and lower bounds
    # for the moments of g according to the exit-location-density

    x = space_vars_vector
    n = length(x)
    r = approximation_degree_r
    @assert(length(initial_condition) == n)
    @assert(upper_or_lower in ["upper_bound", "lower_bound"])
    #Now we can build the model
    solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true)
    model = SOSModel(solver);
    allmons = DynamicPolynomials.monomials(x, 0:r)
    #We make our main variable
    @variable(model, V, Poly(allmons))
    #And define the objective function
    if upper_or_lower == "lower_bound"
        @objective(model, Max, V(x=>initial_condition))
    else
        @objective(model, Min, V(x=>initial_condition))
    end
    #Next we build the constraints
    #Our first constraint requires two sums of squares
    @variable(model, sigma1, Poly(allmons))
    @constraint(model, sigma1 in SOSCone())
    
    allmons_lower_deg = DynamicPolynomials.monomials(x, 0:r-2)
    @variable(model, sigma2, Poly(allmons_lower_deg))
    @constraint(model, sigma2 in SOSCone())
    f1 = 1-sum([x[k]^2 for k in 1:n])
    #Create our constraint in the interior of the ball
    if upper_or_lower == "lower_bound"
        @constraint(model, (-1)*differential_operator_L(V) == sigma1 + sigma2 * f1)
    else
        @constraint(model, differential_operator_L(V) == sigma1 + sigma2 * f1)
    end
    #Our second constraint requires three sums of squares tau
    @variable(model, tau1, Poly(allmons))
    @constraint(model, tau1 in SOSCone())
    
    allmons_lower_deg = DynamicPolynomials.monomials(x, 0:r-2)
    @variable(model, tau2, Poly(allmons_lower_deg))
    @constraint(model, tau2 in SOSCone())
    @variable(model, tau3, Poly(allmons_lower_deg))
    @constraint(model, tau3 in SOSCone())


    f1 = 1-sum([x[k]^2 for k in 1:n])
    f2 = (-1)*f1
    #Create our second constraint in the boundary of the ball
    if upper_or_lower == "lower_bound"
        @constraint(model, target_g-V == tau1 + tau2 * f1 + tau3 * f2)
    else
        @constraint(model, V-target_g == tau1 + tau2 * f1 + tau3 * f2)
    end
    return model
end

function compute_moments_array_stopping_time_unit_sphere(space_vars_vector, target_functions_list, differential_operator_L, initial_condition, approximation_degree_r, upper_or_lower)
    #Given:
    # ambient space variables space_vars_vector
    # a list of target polynomials in the ambient space variables target_functions_list
    # an implementation of the differential operator L of the process
    # an initial location vector (y_1,...,y_n) of the process
    # an approximation degree r
    # a string choice of either upper_bound or lower_bound
    # Return a vector of bounds for the moments of the functions g in the target_functions_list according to the exit location measure

    bounds = []
    for target_g in target_functions_list
        println(target_g)
        model = create_model_moments_stopping_time_unit_sphere(space_vars_vector, approximation_degree_r, initial_condition,target_g,differential_operator_L,upper_or_lower)
        JuMP.optimize!(model)
        println(JuMP.primal_status(model))
        value = objective_value(model)
        println(value)
        println("________________________________________")
        push!(bounds, value)        
    end
    return bounds
end

function create_monomial_basis_on_sphere(space_vars_vector, degree_limit_d)
    # Creates a list of monomials in the variables of space_vars_vector of degree at most degree_limit_d
    # which a basis for the coordinate ring of a sphere in degrees at most degree_limit_d
 
    n = length(space_vars_vector)
    d = degree_limit_d
    x = space_vars_vector
    z = x[1:n-1]
    mons_no_last = DynamicPolynomials.monomials(z,0:d)
    mons_no_last_one_less_degree = DynamicPolynomials.monomials(z,0:d-1)
    mons_with_last_once = [x[n]*m for m in mons_no_last_one_less_degree]
    basis_as_List = vcat(mons_with_last_once, mons_no_last)
    l = length(basis_as_List)
    @assert(l==binomial(n+d,d)-binomial(n+d-2,d-2))
    return basis_as_List
end


function compute_moment_matrix(degree_limit_d, space_vars_vector,  differential_operator_L, initial_condition, approximation_degree_r)
    #Given:
    # a degree_limit_d
    # a vector of ambient space variables space_vars_vector
    # an implementation of the differential operator of the process
    # an initial condition
    # an approximation degree r
    
    #Computes: 
    # the PSD matrix M of moments E_{\nu}[f_if_j], whose entries are 
    # obtained by averaging the lower and upper bounds obtained from our optimization
    # and the maxgap, equal to the largest gap between any lower and upper bound
    #for the matrix entries.
    
    n = length(space_vars_vector)
    d = degree_limit_d
    basis_as_List = create_monomial_basis_on_sphere(space_vars_vector, degree_limit_d)    
    l = length(basis_as_List)
    M = zeros(l,l)
    #N = Array{Monomial{true}}(undef,l,l)
    maxgap = 0.0
    for i in 1:l
        for j in i:l
                g_1 = basis_as_List[i]
                g_2 = basis_as_List[j]
                if j==i
                    target_functions_list = [(g_1*g_2)/2]
                else
                    target_functions_list = [g_1*g_2]
                end
                approximation_degree_r = maximum([approximation_degree_r, maximum([degree(p) for p in target_functions_list])])                
                upper_or_lower = "lower_bound"
                lower_bounds = compute_moments_array_stopping_time_unit_sphere(space_vars_vector, target_functions_list, differential_operator_L, initial_condition, approximation_degree_r, upper_or_lower)
                upper_or_lower = "upper_bound"
                upper_bounds = compute_moments_array_stopping_time_unit_sphere(space_vars_vector, target_functions_list, differential_operator_L, initial_condition, approximation_degree_r, upper_or_lower)
                M[i,j] = (lower_bounds[1] + upper_bounds[1])/2
                gap = abs(upper_bounds[1]-lower_bounds[1])
                if maxgap <  gap
                    maxgap = gap
                end
        end
    end
    M = M+transpose(M)
    return maxgap, M        
end

function compute_normalized_Christoffel_Function(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition,approximation_degree_r)
    # Given:
    # a degree_limit_d of functions which we want to estimate
    # a vector of ambient space variables space_vars_vector
    # an implementation of the differential operator of the process
    # an initial location for the process.
    
    # Computes the normalized Christoffel function (which, under suitable conditions converges to the exit time density)
    # The resulting function can be evaluated via: normalizedChristoffel(x=>[1.0,0.0])
    # It also returns the maxgap between obtained upper and lower bounds 
    # giving qualitative guarantees for the moment estimation.
    maxgap, M = compute_moment_matrix(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition,approximation_degree_r)
    basis = create_monomial_basis_on_sphere(space_vars_vector,degree_limit_d)
    l = length(basis)
    A = inv(M)
    K = transpose(basis)*A*basis
    normalizedChristoffel = l/K
    return maxgap, normalizedChristoffel    
end
