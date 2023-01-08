#=
test_a_matrix:
- Julia version: 
- Author: Ric Porteous
- Date: 2023-01-06
=#
using LinearAlgebra
using PyPlot
using ADCME

function create_A(n_col,del_s)
    A = zeros(n_col,n_col)

    for i in 5:n_col
        A[i, i-4] = -0.5
        A[i, i-3] = 1
        A[i, i-2] = 0
        A[i, i-1] = -1
        A[i, i] = 0.5
    end

    A[1,1] = 1

    #A[2,1] = -3/2
    #A[2,2] = 2
    #A[2,3] = -0.5
    A[2,:] = 1

    A[3,1] = 1
    A[3,2] = -2
    A[3,3] = 1

    A[4,2] = -1
    A[4,3] = 3
    A[4,4] = -3
    A[4,5] = 1

    return A
end


function create_As(n_col,del_s)
    A_s = zeros(n_col,n_col)

    for i in 5:n_col
        A_s[i, i-4] = -0.5
        A_s[i, i-3] = 1
        A_s[i, i-2] = 0
        A_s[i, i-1] = -1
        A_s[i, i] = 0.5
    end

    A_s[1,1] = 1

    A_s[2,1] = -3/2
    A_s[2,2] = 2
    A_s[2,3] = -0.5

    A_s[3,:] = ones(n_col)

    #A_s[3,1] = 1
    #A_s[3,2] = -2
    #A_s[3,3] = 1

    A_s[4,2] = -5/2
    A_s[4,3] = 9
    A_s[4,4] = -12
    A_s[4,5] = 7
    A_s[4,6] = -3/2

    return A_s
end

n_col = 51
speed_col = LinRange(0,1,n_col)|> collect
del_s = speed_col[2] - speed_col[1]
guess_speed = speed_tf_guess(speed_col)
A_eq = create_As(n_col,del_s,guess_speed)
config_s,theta_s = speed_approximater_inits()
speed_error_rhs = speed_error_est(speed_col[1:n_col-1],config_s,theta_s)
rhs_speed = [speed_error_rhs; constant(zeros(1,1)) ]
speed_tf_error = A_eq \ rhs_speed
speed_tf_estimate = guess_speed*(1.+speed_tf_error)
speed_tf = speed_est(speed,speed_col,speed_tf_estimate)

sess = Session(); init(sess)
println(run(sess,speed_tf_estimate))
println(run(sess,rhs_third_d))
#println(A_eq)
#BFGS!(sess,loss)

plot(speed_col,run(sess,speed_tf_estimate),"ko")
plot(speed_col,speed_col.^2 .*(1.0 .-speed_col.^2))
savefig("results/test.png")
