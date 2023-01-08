#=
test_a_matrix:
- Julia version: 
- Author: Ric Porteous
- Date: 2023-01-06
=#
using LinearAlgebra
using ADCME
using PyPlot
using CSV
using DataFrames
using Debugger

function load_data(file_name)
    df = CSV.read(file_name, DataFrame)
    return df,length(df[:,1])
end

function process_to_arrays(df)
    time = df.time |> collect
    temperature = df.temperature |> collect
    h1 = df.h1 |> collect
    h2 = df.h2 |> collect
    Uin = df.Uin |> collect
    speed = df.speed |> collect
    friction = df.friction |> collect
    return time, temperature, h1, h2, Uin, speed, friction
end

function speed_approximater_inits()
    config = [10,10,1]
    theta = Variable(fc_init([1,config...]))
    return config, theta
end

function speed_tf_guess(s)
    #guess = -0.44*s.^2 .+ 0.66.*s
    guess = s.^2 .*(1.0 .- s.^2)
    return guess # Initial guess at what it's supposed to be
    #return 0*s
end

function speed_error_est(a, config,theta)
    error =  0.1*tanh(fc(a,config,theta))
    return error
end

function speed_est(speed,speed_col,speed_tf_estimate)
    return interp1(speed_col,squeeze(speed_tf_estimate),speed)
end

function create_As(n_col,del_s,g)
    A_s = zeros(n_col,n_col)
    for i in 1:n_col
        A_s[i, i] = 1
    end
    A_s[n_col,:] = g.*ones(n_col)
    return A_s
end


TANK_DATA = "two_tanks_non_linear_example.csv"

# Load data
df,n = load_data(TANK_DATA)
time, temperature, h1, h2, Uin, speed, true_k = process_to_arrays(df)

# Configure approximators

n_col = 11
speed_col = LinRange(0,0.95,n_col)|> collect # Don't go to 1 otherwise matrix becomes less than full rank'
del_s = speed_col[2] - speed_col[1]
guess_speed_tf = speed_tf_guess(speed_col)
A_eq = create_As(n_col,del_s,guess_speed_tf)
config_s,theta_s = speed_approximater_inits()
speed_error_rhs = speed_error_est(speed_col[1:n_col-1],config_s,theta_s)
#speed_error_rhs = Variable(zeros(n_col-1,1))
rhs_speed = [reshape(speed_error_rhs,:,1); constant(zeros(1,1)) ]
speed_tf_error = A_eq \ rhs_speed
speed_tf_estimate = guess_speed_tf.*(1.0 .+squeeze(speed_tf_error))
speed_tf = speed_est(speed,speed_col,speed_tf_estimate)

sess = Session(); init(sess)
println(run(sess,speed_tf_estimate))
println(speed_tf)
#println(speed)

#println(run(sess,rhs_third_d))
#println(rank(A_eq))
#println(A_eq)
#BFGS!(sess,loss)

plot(speed_col,run(sess,speed_tf_estimate),"ko")
plot(speed_col,speed_col.^2 .*(1.0 .-speed_col.^2))
savefig("results/test2.png")
