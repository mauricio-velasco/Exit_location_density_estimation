using DynamicPolynomials
using CSDP
using SumOfSquares
using JuMP
using LinearAlgebra
import MultivariatePolynomials.degree
import Base.real

function real(p)
    #returns the real part of a mult
    coeffs = Array{Float64}(real(p.a))
    return DynamicPolynomials.Polynomial(coeffs,p.x)
end


function degree(q)
    #Extends the functionality of MultivariatePolynomials
    D = [degree(m) for m in monomials(q)]
    return maximum(D)
end
    
function create_model_moments_stopping_time_unit_sphere_v2(space_vars_vector, approximation_degree_r, initial_condition, target_g, differential_operator_L, upper_or_lower)
    # Given:
    # ambient space variables space_vars_vector
    # an approximation degree r (which must be at least two)
    # an initial location vector (y_1,...,y_n)
    # a target polynomial on the ambient space variables g
    # an implementation of the differential operator L
    # a string choice of either upper_bound or lower_bound
    # Return a PolyJump model which gives the desired upper and lower bounds
    # for the moments of g according to the exit-location-density

    x = space_vars_vector
    n = length(x)
    r = approximation_degree_r
    @assert(r>=2)
    @assert(length(initial_condition) == n)
    @assert(upper_or_lower in ["upper_bound", "lower_bound"])
    #Now we can build the model
    solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true)
    model = SOSModel(solver);
    allmons = DynamicPolynomials.monomials(x, 0:r)
    #We make our main variable
    @variable(model, V, SumOfSquares.Poly(allmons))#this says that the variable is of class Poly as defined in the sum of squares package
    #And define the objective function
    if upper_or_lower == "lower_bound"
        @objective(model, Max, V(x=>initial_condition))
    else
        @objective(model, Min, V(x=>initial_condition))
    end
    #Next we build the constraints
    #Our first constraint requires two sums of squares
    @variable(model, sigma1, SumOfSquares.Poly(allmons))
    @constraint(model, sigma1 in SOSCone())
    
    allmons_lower_deg = DynamicPolynomials.monomials(x, 0:r-2)
    @variable(model, sigma2, SumOfSquares.Poly(allmons_lower_deg))
    @constraint(model, sigma2 in SOSCone())
    f1 = 1-sum([x[k]^2 for k in 1:n])
    #Create our constraint in the interior of the ball
    if upper_or_lower == "lower_bound"
        @constraint(model, (-1)*differential_operator_L(V) == sigma1 + sigma2 * f1)
    else
        @constraint(model, differential_operator_L(V) == sigma1 + sigma2 * f1)
    end
    #Our second constraint requires three sums of squares tau
    @variable(model, tau1, SumOfSquares.Poly(allmons))
    @constraint(model, tau1 in SOSCone())

    allmons_lower_deg = DynamicPolynomials.monomials(x, 0:r-2)
    @variable(model, gen_pol_lower_degree, SumOfSquares.Poly(allmons_lower_deg))

    f1 = 1-sum([x[k]^2 for k in 1:n])
    #Create our second constraint in the boundary of the ball
    if upper_or_lower == "lower_bound"
        @constraint(model, target_g-V == tau1 + gen_pol_lower_degree * f1 )
    else
        @constraint(model, V-target_g == tau1 + gen_pol_lower_degree * f1)
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
        model = create_model_moments_stopping_time_unit_sphere_v2(space_vars_vector, approximation_degree_r, initial_condition,target_g,differential_operator_L,upper_or_lower)
        JuMP.optimize!(model)
        println(JuMP.primal_status(model))
        value = objective_value(model)
        println(value)
        println("____________________________________________________________________________________")
        push!(bounds, value)        
    end
    return bounds
end

function compute_certified_exit_location_moments(space_vars_vector, target_functions_list, differential_operator_L, initial_location, error_tolerance, maximum_allowed_degree_increases = 40)
    # USEFUL FUNCTION 1:
    # Given:
    # ambient space variables space_vars_vector
    # a list of target polynomials in the ambient space variables called target_functions_list
    # an implementation of the differential operator L of the process
    # an initial location vector (y_1,...,y_n) of the process
    # an error_tolerance_bound
    # an integer maximum_allowed_degree_increases (by default 5)
    # Return vectors of lower_bounds and upper_boundsfor the moments of the target_functions_list 
    # with respect to the exit location measure from the unit sphere.

    x1 = space_vars_vector[1]#used later for hack to make sure degree is defined for constant polynomials.
    lower_bounds = []
    upper_bounds = []
    for target_g in target_functions_list
        local_target_functions_list = [target_g*x1^0]
        certified_result = false
        approximation_degree_r = max(2,degree(target_g*x1^0)) # Hack to make sure the degree is defined, must be at least two
        increases = 0

        while !certified_result
            upper_or_lower = "upper_bound"
            upper_bound = compute_moments_array_stopping_time_unit_sphere(space_vars_vector, local_target_functions_list, differential_operator_L, initial_location, approximation_degree_r, upper_or_lower)
            upper_bound = upper_bound[1]

            upper_or_lower = "lower_bound"
            lower_bound = compute_moments_array_stopping_time_unit_sphere(space_vars_vector, local_target_functions_list, differential_operator_L, initial_location, approximation_degree_r, upper_or_lower)    
            lower_bound = lower_bound[1]

            if abs(upper_bound-lower_bound) < error_tolerance
                push!(lower_bounds,lower_bound)
                push!(upper_bounds,upper_bound)
                certified_result = true
            else
                increases += 1
                approximation_degree_r += 1
                @assert increases <= maximum_allowed_degree_increases "Unable to achieve the desired error bound. Please increase the allowed approximation degree for function "*string(target_g)*" beyond "*string(approximation_degree_r)
            end
        end
    end
    return upper_bounds, lower_bounds
end


function create_monomial_basis_on_sphere(space_vars_vector, degree_limit_d)
    #Temporary attempt for building Tchebyshev basis in 2D
    # Creates a list of monomials in the variables of space_vars_vector of degree at most degree_limit_d
    # which a basis for the coordinate ring of a sphere in degrees at most degree_limit_d
    n = length(space_vars_vector)
    @assert(n==2)#this temporary function works only for 2D
    d = degree_limit_d
    x = space_vars_vector
    z = x[1:n-1]
    basis_as_List = [1.0*x[1]^0+0.0]
    p=x[1]+im*x[2]
    q=x[1]+im*x[2]

    for k in 1:degree_limit_d
        cos_part = real((p^k+q^k)/2)
        push!(basis_as_List, cos_part)
        sin_part = real((p^k+q^k)/(2*im))
        push!(basis_as_List, sin_part)
    end
    l = length(basis_as_List)
    @assert(l==binomial(n+d,d)-binomial(n+d-2,d-2))
    return basis_as_List
end



function good_create_monomial_basis_on_sphere(space_vars_vector, degree_limit_d, rotation_angle=pi/4)
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
    if n>=2
        #We rotate the first two components of every element of the basis by pi/4 whenever the dimension allows
        #This improves the condition number of the moment matrix in our examples
        alpha = rotation_angle
        rotated_basis_as_List = [subs(p,x[1]=>cos(alpha)*x[1]-sin(alpha)*x[2],x[2]=>sin(alpha)*x[1]+cos(alpha)*x[2]) for p in basis_as_List]
        basis_as_List = rotated_basis_as_List
    end
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
    # a basis f_i for polynomials of degree at most degree_limit_d
    # the PSD matrix M of moments E_{\nu}[f_if_j], whose entries are 
    # obtained by averaging the lower and upper bounds resulting from our optimization
    # and the maxgap, equal to the largest gap between any lower and upper bound
    # for the matrix entries.
    
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


function compute_certified_moment_matrix(degree_limit_d, space_vars_vector,  differential_operator_L, initial_condition, required_error_bound)
    #Given:
    # a degree_limit_d
    # a vector of ambient space variables space_vars_vector
    # an implementation of the differential operator of the process
    # an initial condition
    # a required error bound
    
    #Computes: 
    # a basis f_i for polynomials of degree at most degree_limit_d
    # the PSD matrix M of moments E_{\nu}[f_if_j], whose entries are 
    # obtained by averaging the lower and upper bounds resulting from our optimization
    # and the maxgap, equal to the largest gap between any lower and upper bound
    # for the matrix entries.
    
    n = length(space_vars_vector)
    d = degree_limit_d
    basis_as_List = create_monomial_basis_on_sphere(space_vars_vector, degree_limit_d)    
    l = length(basis_as_List)
    M = zeros(l,l)
    #N = Array{Monomial{true}}(undef,l,l)
    maxgap = 0.0
    for i in 1:l
        for j in i:l
                print( "Calculando item "*string(i)*","*string(j)*" de "*string(l)*" cuadrado.\n")
                g_1 = basis_as_List[i]
                g_2 = basis_as_List[j]
                if j==i
                    target_functions_list = [(g_1*g_2)/2]
                else
                    target_functions_list = [g_1*g_2]
                end
                error_tolerance = required_error_bound                
                lower_bounds, upper_bounds = compute_certified_exit_location_moments(space_vars_vector, target_functions_list, differential_operator_L, initial_location, error_tolerance)
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
    
    # Computes the normalized Christoffel function ON THE UNIT SPHERE (which, under suitable conditions converges to the exit time density).
    # The resulting function can be evaluated via: normalizedChristoffel(x=>[1.0,0.0])
    # It also returns the maxgap between obtained upper and lower bounds 
    # giving qualitative guarantees for the moment estimation.
    maxgap, M = compute_moment_matrix(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition, approximation_degree_r)
    basis = create_monomial_basis_on_sphere(space_vars_vector,degree_limit_d)
    M = Matrix{Float64}(M)
    l = length(basis)
    A = inv(M)
    K = transpose(basis)*A*basis
    normalizedChristoffel = l/(K*2*pi)
    return maxgap, normalizedChristoffel    
end


function compute_certified_exit_density_estimation(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition, required_error_bound)
    #USEFUL FUNCTION 2:
    # Given:
    # a degree_limit_d of functions whose moments we want to estimate
    # a vector of ambient space variables space_vars_vector
    # an implementation of the differential operator of the process
    # an initial location for the process.
    # A required error bound. The algorithm will continue to compute until this error bound is met
    
    # Computes the normalized Christoffel function (which, under suitable conditions converges to the exit time density) on the unit sphere
    # The resulting function can be evaluated via the output with the syntax: normalizedChristoffel(x=>[1.0,0.0])
    # It also returns the maxgap between obtained upper and lower bounds 
    # giving qualitative guarantees for the moment estimation.
    maxgap, M = compute_certified_moment_matrix(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition, required_error_bound)
    basis = create_monomial_basis_on_sphere(space_vars_vector, degree_limit_d)
    M = Matrix{Float64}(M)
    l = length(basis)
    moments_condition_number = cond(M)
    A = inv(M)
    K = transpose(basis)*A*basis
    normalizedChristoffel = l/(K*2*pi)
    return moments_condition_number, maxgap, normalizedChristoffel    
end



#The following functions can be used to estimate the density of the occupation measure 
#of our diffusion in the unit ball.



function polynomial_to_coefficient_vector(target_polynomial, monomial_basis)
    #Given:
    #An array of space variables
    #A target polynomial in the space variables
    #A monomial_basis
    #We compute the representation of the target polynomial in the monomial_basis
    #We first check whether the operation is possible
    required_mons = DynamicPolynomials.monomials(target_polynomial)
    for mon in required_mons
        @assert(mon in monomial_basis)
    end
    result = zeros(0)
    for basis_element in monomial_basis
        push!(result,DynamicPolynomials.coefficient(target_polynomial,basis_element))    
    end
return result
end

function coefficient_vector_to_polynomial(coefficient_vector, monomial_basis)
    #Given:
    #A coefficient vector
    #A monomial basis
    #We compute the polynomial described by the vector
    summands = []
    for k in eachindex(monomial_basis)
        coeff = coefficient_vector[k]
        monom = monomial_basis[k]
        push!(summands,coeff*monom)
    end
return sum(summands)
end


function differential_operator_as_matrix(differential_operator_L, source_monomial_basis, target_monomial_basis)
    #Given a source and target monomial basis, write the differential operator as a matrix in this space.
    #The existence of a coefficient representation first verifies that the required monomials are in the target 
    #basis ensuring the operator is well defined.
    ResultColumns = []
    for source_basic_monomial in source_monomial_basis
        transformed_monomial = differential_operator_L(source_basic_monomial)
        push!(ResultColumns, polynomial_to_coefficient_vector(transformed_monomial, target_monomial_basis))
    end
    return reduce(hcat,ResultColumns)
end


function find_an_inverse_image_polynomial_under_differential_operator(differential_operator_L, source_monomial_basis, target_monomial_basis, target_polynomial_g)
    #Finds a polynomial w with Lw = target_polynomial_g (the sizes of spaces are encoded by source and target monomial bases)
    M = differential_operator_as_matrix(differential_operator_L, source_monomial_basis, target_monomial_basis)
    r = polynomial_to_coefficient_vector(target_polynomial_g,target_monomial_basis)
    # Now we want to find a polynomial whose Lf is p:
    wanted_coefficients = M\r
    wanted_polynomial = coefficient_vector_to_polynomial(wanted_coefficients, source_monomial_basis)
    return wanted_polynomial    
end


function create_monomial_basis_on_ball(space_vars_vector, degree_limit_d)
    return DynamicPolynomials.monomials(space_vars_vector, 0:degree_limit_d)
end


function compute_bound_for_given_occupation_moment_on_ball(space_vars_vector, degree_limit_d, target_polynomial_g,positive_offset, differential_operator_L, initial_condition, approximation_degree_r, upper_or_lower)
    #First, we find an inverse image under the infinitesimal generator
    x = space_vars_vector
    target_monomial_basis = create_monomial_basis_on_ball(space_vars_vector, degree_limit_d)
    source_monomial_basis = create_monomial_basis_on_ball(space_vars_vector, degree_limit_d + positive_offset)
    wanted_polynomial_p = find_an_inverse_image_polynomial_under_differential_operator( differential_operator_L, source_monomial_basis, target_monomial_basis,target_polynomial_g)
    # Then we estimate the bound using the Kolmogorov equation:
    # <target_polynomial_g, mu> = wanted_polynomial(initial_condition) - <wanted_polynomial_p,nu> 
    if upper_or_lower == "upper_bound"
        upper_or_lower_in_boundary = "lower_bound"
        target_functions_list = [wanted_polynomial_p]
        lower_bound_boundary = compute_moments_array_stopping_time_unit_sphere(space_vars_vector, target_functions_list, differential_operator_L, initial_condition, approximation_degree_r, upper_or_lower_in_boundary)        
        bound = wanted_polynomial_p(x=> initial_condition)-lower_bound_boundary[1]
    elseif upper_or_lower == "lower_bound"                
        upper_or_lower_in_boundary = "upper_bound"
        target_functions_list = [wanted_polynomial_p]
        upper_bound_boundary = compute_moments_array_stopping_time_unit_sphere(space_vars_vector, target_functions_list, differential_operator_L, initial_condition, approximation_degree_r, upper_or_lower_in_boundary)
        bound = wanted_polynomial_p(x=> initial_condition)-upper_bound_boundary[1]
    end
    return bound
end


function compute_occupation_moment_matrix_on_ball(degree_limit_d, space_vars_vector,  differential_operator_L, positive_offset, initial_condition, approximation_degree_r)
    #Given:
    # a degree_limit_d
    # a vector of ambient space variables space_vars_vector
    # an implementation of the differential operator of the process
    # a positive offset for the process (we look for polynomials f of degree at most positive_offset+degree_limit_d with given Lf)
    # an initial condition
    # an approximation degree r
    
    #Computes: 
    # the PSD matrix M of moments E_{\mu}[f_if_j], whose entries are 
    # obtained by averaging the lower and upper bounds from our optimization routine for the moments of \nu
    # applied to the function h obtained as inverse image of f_if_j under the differential operator in degree
    # below source_degree plus an integer called positive_offset.
    # The maxgap, which is also returned, is equal to the largest gap between any lower and upper bound
    # for the matrix entries.
    
    target_basis = create_monomial_basis_on_ball(space_vars_vector, degree_limit_d)   
    l = length(target_basis)
    M = zeros(l,l)
    maxgap = 0.0

    for i in 1:l
        for j in i:l
                g_1 = target_basis[i]
                g_2 = target_basis[j]
                if j==i
                    target_polynomial_g = (g_1*g_2)/2
                else
                    target_polynomial_g = g_1*g_2
                end
                approximation_degree_r_boundary = maximum([approximation_degree_r, degree(target_polynomial_g)+positive_offset])
                #TODO: add positive offset to previous line                
                degree_limit_d_boundary = degree(target_polynomial_g)
                upper_or_lower = "lower_bound"
                lower_bound = compute_bound_for_given_occupation_moment_on_ball(space_vars_vector, degree_limit_d_boundary, target_polynomial_g,positive_offset, differential_operator_L, initial_condition, approximation_degree_r_boundary, upper_or_lower)
                upper_or_lower = "upper_bound"
                upper_bound = compute_bound_for_given_occupation_moment_on_ball(space_vars_vector, degree_limit_d_boundary, target_polynomial_g,positive_offset, differential_operator_L, initial_condition, approximation_degree_r_boundary, upper_or_lower)

                M[i,j] = (lower_bound + upper_bound)/2
                gap = abs(upper_bound-lower_bound)
                if maxgap <  gap
                    maxgap = gap
                end
        end
    end
    M = M+transpose(M)
    return maxgap, M, target_basis        
end


function compute_normalized_Christoffel_Function_occupation_measure_on_ball(degree_limit_d,space_vars_vector, differential_operator_L, initial_condition,approximation_degree_r,positive_offset)
    x = space_vars_vector
    maxgap, M, target_basis = compute_occupation_moment_matrix_on_ball(degree_limit_d,space_vars_vector,differential_operator_L,positive_offset, initial_condition, approximation_degree_r)
    l = length(target_basis)
    A = inv(M)
    K = transpose(target_basis)*A*target_basis
    normalizedChristoffel = l/K
    return maxgap, normalizedChristoffel    
end

