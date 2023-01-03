#=
twotanks_nl_inference_penalty:
- Julia version: 
- Author: Ric Porteous
- Date: 2023-01-03
=#
using LinearAlgebra
using ADCME
using PyPlot
using CSV
using DataFrames
using Debugger

TANK_DATA = "two_tanks_non_linear_example.csv"

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

function calc_dt(time)
    dt = time[2] - time[1]
    return dt
end

function friction_approximater_inits()
    config = [10,10,1]
    theta = Variable(fc_init([1,config...]))
    return config, theta
end

function friction_est(a, config,theta)
    nn = squeeze(fc(a,config,theta)^2) # 1 for stablity
    return nn
end


function eta_est()
    eta = Variable(1.0)
    return eta
end

function visualise_result_friction_temp(a, true_k, config, theta, sess)
    a_range = LinRange(-60,20,101) |> collect
    est_k = run(sess,friction_est(a_range, config,theta)) # 1 for stablity
    figure(figsize=(10,4))
    plot(a,true_k,"k.")
    plot(a_range,est_k,"r-")
    xlabel("a"); ylabel("K")
    savefig("results/pcl_bck_lin_crank/a vs k.png")
end

function visualise_result_friction_time(time,true_k,est_k)
    figure(figsize=(10,4))
    plot(time,true_k,"k.")
    plot(time,est_k,"r-")
    xlabel("time"); ylabel("k")
    ylim((0,2))
    savefig("results/pcl_bck_lin_crank/time vs k.png")

    figure(figsize=(10,4))
    plot(est_k,true_k,"k.")
    xlabel("est_k"); ylabel("true_k")
    savefig("results/pcl_bck_lin_crank/est_k vs true_k.png")
end

function visualise_result_state_time(time,true_h1,true_h2,h1_est,h2_est)
    figure(figsize=(10,4))
    plot(time,true_h1,"k.")
    plot(time,h1_est,"r-")
    xlabel("time"); ylabel("h1")
    savefig("results/pcl_bck_lin_crank/time vs h1.png")

    figure(figsize=(10,4))
    plot(time,true_h2,"k.")
    plot(time,h2_est,"r-")
    xlabel("time"); ylabel("h2")
    savefig("results/pcl_bck_lin_crank/time vs h2.png")
end

df,n = load_data(TANK_DATA)
time, temperature, h1, h2, Uin, speed, true_k = process_to_arrays(df)
dt = calc_dt(time)

config,theta = friction_approximater_inits()
K_approx = friction_est(temperature,config,theta)
eta_approx = eta_est()

U_IN = constant(Uin)
S_IN = constant(speed)
Y_IN = constant(h1 - h2)

function condition(i, h_arr)
    i <= n
end

function backward_linearised_crank_nicolson(i, h_arr)

    # An explicit integrator that uses forward in time calcualate the next time step

    h = read(h_arr, i - 1)

    # Forward pass
    Fim1 = [(-K_approx[i-1]*(S_IN[i-1]*S_IN[i-1]*(1.0-S_IN[i-1]*S_IN[i-1]))*h[1]*h[1] + U_IN[i-1]); (K_approx[i-1]*(S_IN[i-1]*S_IN[i-1]*(1.0-S_IN[i-1]*S_IN[i-1]))*h[1]*h[1] - eta_approx^2 *h[2])]

    # Jacobian
    a = -2*K_approx[i-1]*(S_IN[i-1]*S_IN[i-1]*(1.0-S_IN[i-1]*S_IN[i-1]))*h[1]
    b = 0
    c = 2*K_approx[i-1]*(S_IN[i-1]*S_IN[i-1]*(1.0-S_IN[i-1]*S_IN[i-1]))*h[1]
    d = -eta_approx^2
    J = tensor([a b; c d])

    # Right hand side LHS*H_next = RHS
    rhs = (I - dt/2*J)*h + dt*Fim1
    lhs = I - dt/2*J

    # Solve
    h_next = lhs \ rhs

    h_arr = write(h_arr, i, h_next)
    i + 1, h_arr
end


h_arr = TensorArray(n)
h_arr = write(h_arr, 1, zeros(2))
i = constant(2, dtype = Int32)
_, h = while_loop(condition, backward_linearised_crank_nicolson, [i, h_arr])
h_est = set_shape(ADCME.stack(h), (n, 2))

# Compute the loss function
loss = sum(((h_est[:,1]-h_est[:,2]) - (h1-h2) )^2)

# Perform the optimisation
sess = Session(); init(sess)

optimisation_time = @elapsed begin
    BFGS!(sess, loss,1000)
end

# Visualise
visualise_result_friction_temp(temperature, true_k, config, theta, sess)
visualise_result_friction_time(time, true_k, run(sess,K_approx))
visualise_result_state_time(time,h1,h2,run(sess,h_est[:,1]),run(sess,h_est[:,2]))
println(run(sess,eta_approx))

