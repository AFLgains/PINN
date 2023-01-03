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

function create_res(h1,h2,K_approx,eta_approx,speed, Uin,dt)
    n = length(h1)
    speed_correction = speed.*speed.*(1.0.-speed.*speed)
    res_h1 = (h1[2:end] - h1[1:end-1])/dt + K_approx[2:end].*speed_correction[2:end].*h1[2:end].*h1[2:end] - Uin[2:end]
    res_h2 = (h2[2:end] - h2[1:end-1])/dt - K_approx[2:end].*speed_correction[2:end].*h1[2:end].*h1[2:end] + eta_approx^2 *h2[2:end]
    return res_h1, res_h2
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
    savefig("results/pen_y/a vs k.png")
end

function visualise_result_friction_time(time,true_k,est_k)
    figure(figsize=(10,4))
    plot(time,true_k,"k.")
    plot(time,est_k,"r-")
    xlabel("time"); ylabel("k")
    ylim((0,2))
    savefig("results/pen_y/time vs k.png")

    figure(figsize=(10,4))
    plot(est_k,true_k,"k.")
    xlabel("est_k"); ylabel("true_k")
    savefig("results/pen_y/est_k vs true_k.png")
end

function visualise_result_state_time(time,true_h1,true_h2,h1_est,h2_est)
    figure(figsize=(10,4))
    plot(time,true_h1,"k.")
    plot(time,h1_est,"r-")
    xlabel("time"); ylabel("h1")
    savefig("results/pen_y/time vs h1.png")

    figure(figsize=(10,4))
    plot(time,true_h2,"k.")
    plot(time,h2_est,"r-")
    xlabel("time"); ylabel("h2")
    savefig("results/pen_y/time vs h2.png")
end

df,n = load_data(TANK_DATA)
time, temperature, h1, h2, Uin, speed, true_k = process_to_arrays(df)
dt = calc_dt(time)

config,theta = friction_approximater_inits()
K_approx = friction_est(temperature,config,theta)
h1_est = Variable(ones(n))
h2_est = Variable(ones(n))
eta_approx = eta_est()

# Create rest
res1,res2 = create_res(h1_est,h2_est,K_approx,eta_approx,speed,Uin,dt)

# Calculate the residual
loss = sqrt(sum(res1^2) + sum(res2^2) + sum(((h1_est - h2_est) - (h1 - h2))^2) )

# Session + initialisation
sess = Session(); init(sess)

# Optimise!
#BFGS!(sess, loss)

learning_rate = 1e-2
opt = AdamOptimizer(learning_rate).minimize(loss)
sess = Session(); init(sess)
for i = 1:100000
    _, l = run(sess, [opt, loss])
    @info "Iteration $i, loss = $l"
end

# Visualise
visualise_result_friction_temp(temperature, true_k, config, theta, sess)
visualise_result_friction_time(time, true_k, run(sess,K_approx))
visualise_result_state_time(time,h1,h2,run(sess,h1_est),run(sess,h2_est))
println(run(sess,eta_approx))

