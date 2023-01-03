#=
In this script we'll be inferring how temperature and friction are related using the
PCL technique''
car_inference_rm:
- Julia version:
- Author: Ric Porteous
- Date: 2022-12-27
=#
using LinearAlgebra
using ADCME
using PyPlot
using CSV
using DataFrames
using Debugger
using ControlSystems

INTEGRATOR_METHOD = "backward_nr" # backward,backward_nr, forward, crank, , luenburge, kalman (not implemented)
TANK_DATA = "two_tanks_example.csv"

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
    friction = df.friction |> collect
    return time, temperature, h1, h2, Uin, friction
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
    eta = Variable(0.2)
    return eta
end

function visualise_result_friction_temp(a, true_k, config, theta, sess)
    a_range = LinRange(-40,40,101) |> collect
    est_k = run(sess,friction_est(a_range, config,theta)) # 1 for stablity
    figure(figsize=(10,4))
    plot(a,true_k,"k.")
    plot(a_range,est_k,"r-")
    xlabel("a"); ylabel("K")
    savefig(INTEGRATOR_METHOD* "_temperature vs k_pcl.png")
end

function visualise_result_friction_time(time,true_k,est_k)
    figure(figsize=(10,4))
    plot(time,true_k,"k.")
    plot(time,est_k,"r-")
    xlabel("time"); ylabel("k")
    savefig(INTEGRATOR_METHOD* "time vs k_pcl.png")

    figure(figsize=(10,4))
    plot(est_k,true_k,"k.")
    xlabel("est_k"); ylabel("true_k")
    savefig(INTEGRATOR_METHOD* "est_k vs true_k_pcl.png")
end

function visualise_result_h_time(time,true_h1,true_h2,h_est)
    figure(figsize=(10,4))
    plot(time,true_h1,"k.")
    plot(time,h_est[:,1],"r-")
    xlabel("time"); ylabel("h1")
    savefig(INTEGRATOR_METHOD* "time vs h1_pcl.png")

    figure(figsize=(10,4))
    plot(time,true_h2,"k.")
    plot(time,h_est[:,2],"r-")
    xlabel("time"); ylabel("h2")
    savefig(INTEGRATOR_METHOD* "time vs h2_pcl.png")
end


df,n = load_data(TANK_DATA)
time, temperature, h1, h2, Uin, true_k = process_to_arrays(df)
dt = calc_dt(time)

config,theta = friction_approximater_inits()
K_approx = friction_est(temperature,config,theta)
eta_approx = eta_est()


U_IN = zeros(n, 2)
U_IN[:,1] = Uin
U_IN = constant(U_IN)
Y_IN = constant(h1 - h2)



# Compute the loss function
loss = sum(((h_est[:,1]-h_est[:,2]) - (h1-h2) )^2)

# Perform the optimisation
sess = Session(); init(sess)

optimisation_time = @elapsed begin
    BFGS!(sess, loss, 1000)
end
#learning_rate = 1e-2
#opt = AdamOptimizer(learning_rate).minimize(loss)
#sess = Session(); init(sess)
#for i = 1:300
#    _, l = run(sess, [opt, loss])
#    @info "Iteration $i, loss = $l"
#end

# Visualise
visualise_result_friction_temp(temperature, true_k, config, theta, sess)
visualise_result_friction_time(time, true_k, run(sess,K_approx))
visualise_result_h_time(time,h1,h2,run(sess,h_est))

# Record results
k_error = mean((true_k - run(sess,K_approx)).^2)
open( INTEGRATOR_METHOD * ".txt","a") do io
   println(io, "integrator,total_loss,computation_time,k_error")
   println(io, INTEGRATOR_METHOD,",",run(sess,loss),",",optimisation_time,",",k_error)
end