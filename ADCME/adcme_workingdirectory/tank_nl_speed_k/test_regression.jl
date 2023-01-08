#=
test_regression:
- Julia version: 
- Author: Ric Porteous
- Date: 2023-01-07
=#
using ADCME
using LinearAlgebra
using CSV
using DataFrames
using Debugger
using PyPlot

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

function create_regression_coeffs(n_coefficients)
    regression_matrix = zeros(n,n_coefficients)
    for  i in 1:n_coefficients
        regression_matrix[:,i] = speed.^i
    end
    return regression_matrix
end

TANK_DATA = "two_tanks_non_linear_example.csv"
MAX_SPEED_INFLUENCE = 0.25

# Load data
df,n = load_data(TANK_DATA)
time, temperature, h1, h2, Uin, speed, true_k = process_to_arrays(df)

n_coefficients = 8
regression_matrix = create_regression_coeffs(n_coefficients)
speed_coeff_positive = -tf.exp(Variable(ones(n_coefficients-2,1)))
speed_coeff_unconstrained = Variable(ones(2,1))
speed_coeff = [speed_coeff_unconstrained;speed_coeff_positive]
speed_predictions = regression_matrix*speed_coeff

# Perform the optimisation
sess = Session(); init(sess)
println(run(sess,speed_predictions))