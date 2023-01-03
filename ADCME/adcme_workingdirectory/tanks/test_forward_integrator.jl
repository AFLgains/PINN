#=
test_forward_integrator:
- Julia version: 
- Author: Ric Porteous
- Date: 2023-01-01
=#

n = 10
U_IN = zeros(2, n)
U_IN[1,:] = ones(n)
println(U_IN)

function forward_time(i, h_arr)
    delt = 0.0001
    # An explicit integrator that uses forward in time calcualate the next time step
    A = Array([-1 0.0;1 -0.1])
    h = h_arr

    h_next = h
    n_steps = 1.0 / delt
    for steps in 1:n_steps
        h_next = h_next + delt*(A*h_next .+ U_IN[:,i])
        println(h_next)
    end

    h_arr = h_next
    i + 1, h_arr
end

h_arr = [0; 0]
j = 1
j,h_arr = forward_time(j,h_arr)
println("end",h_arr)
