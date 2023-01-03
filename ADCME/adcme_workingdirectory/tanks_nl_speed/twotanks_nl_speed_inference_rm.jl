# Two tanks, using residual minimisation
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

function create_res(h1,h2,K_approx,eta_approx,speed_tf, Uin,dt)
    n = length(h1)
    res_h1 = (h1[2:end] - h1[1:end-1])/dt + K_approx[2:end].*speed_tf[2:end].*h1[2:end].*h1[2:end] - Uin[2:end]
    res_h2 = (h2[2:end] - h2[1:end-1])/dt - K_approx[2:end].*speed_tf[2:end].*h1[2:end].*h1[2:end] + eta_approx^2 *h2[2:end]
    return res_h1, res_h2
end


function speed_approximater_inits()
    config = [5,5,1]
    theta = Variable(fc_init([1,config...]))
    return config, theta
end

function speed_est(a, config,theta)
    nn = squeeze(fc(a,config,theta)^2) # square for stablity
    return nn
end


function eta_est()
    eta = Variable(1.0)
    return eta
end

function visualise_result_tf_speed(true_speed,config, theta, sess)

    figure(figsize=(10,4))

    speed_range = LinRange(0,1,101) |> collect
    est_speed = run(sess,speed_est(speed_range, config,theta)) # 1 for stablity
    plot(speed_range,est_speed,"r-")

    plot(true_speed,true_speed.^2 .*(1.0.-true_speed.^2),"k.")

    xlabel("speed"); ylabel("tf")
    savefig("results/rm/speed_tf.png")
end

function visualise_result_speed_time(time,speed,est_speed_tf)
    figure(figsize=(10,4))
    plot(time,speed.^2 .*(1.0 .- speed.^2),"k.")
    plot(time,est_speed_tf,"r-")
    xlabel("time"); ylabel("k")
    ylim((0,2))
    savefig("results/rm/time vs speed_tf.png")

end

df,n = load_data(TANK_DATA)
time, temperature, h1, h2, Uin, speed, true_k = process_to_arrays(df)
dt = calc_dt(time)

config,theta = speed_approximater_inits()
speed_tf = speed_est(speed,config,theta)
eta_approx = eta_est()

res1,res2 = create_res(h1,h2,true_k,eta_approx,speed_tf,Uin,dt)

# Calculate the residual
loss = sum(res1^2) + sum(res2^2)

# Session + initialisation
sess = Session(); init(sess)

# Optimise!
BFGS!(sess, loss)

#learning_rate = 1e-2
#opt = AdamOptimizer(learning_rate).minimize(loss)
#sess = Session(); init(sess)
#for i = 1:10
#    _, l = run(sess, [opt, loss])
#    @info "Iteration $i, loss = $l"
#end

# Visualise
visualise_result_tf_speed(speed,config, theta, sess)
visualise_result_speed_time(time,speed,run(sess,speed_tf))
println(run(sess,eta_approx))

