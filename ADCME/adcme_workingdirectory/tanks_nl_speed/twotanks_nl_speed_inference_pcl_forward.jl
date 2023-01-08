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

function speed_approximater_inits()
    config = [10,10,1]
    theta = Variable(fc_init([1,config...]))
    return config, theta
end

function speed_est(a, config,theta)
    nn =  squeeze(fc(a,config,theta))
    return nn
end

function speed_tf_guess(speed)
    return (-0.44*speed.^2 .+ 0.66.*speed).^0.5 # Initial guess at what it's supposed to be
    #return speed.*speed.*(1.0 .- speed.*speed)
end

function eta_est()
    eta = Variable(1.0)
    return eta
end

function visualise_result_tf_speed(true_speed,config, theta, sess)

    figure(figsize=(10,4))

    speed_range = LinRange(0.2,1,101) |> collect
    speed_error = run(sess,speed_est(speed_range, config,theta))
    speed_base = speed_tf_guess(speed_range)
    est_speed = (speed_error + speed_base).^2
    plot(speed_range,est_speed,"r-")
    plot(true_speed,true_speed.^2 .*(1.0.-true_speed.^2),"k.")

    xlabel("speed"); ylabel("tf")
    savefig("results/fwd/speed_tf.png")
end

function visualise_result_speed_time(time,speed,est_speed_tf)
    figure(figsize=(10,4))
    plot(time,speed.^2 .*(1.0 .- speed.^2),"k.")
    plot(time,est_speed_tf,"r-")
    xlabel("time"); ylabel("k")
    ylim((0,2))
    savefig("results/fwd/time vs speed_tf.png")

end

function visualise_result_state_time(time,true_h1,true_h2,h1_est,h2_est)
    figure(figsize=(10,4))
    plot(time,true_h1,"k.")
    plot(time,h1_est,"r-")
    xlabel("time"); ylabel("h1")
    savefig("results/fwd/time vs h1.png")

    figure(figsize=(10,4))
    plot(time,true_h2,"k.")
    plot(time,h2_est,"r-")
    xlabel("time"); ylabel("h2")
    savefig("results/fwd/time vs h2.png")
end

df,n = load_data(TANK_DATA)
time, temperature, h1, h2, Uin, speed, true_k = process_to_arrays(df)
dt = calc_dt(time)

config,theta = speed_approximater_inits()
speed_tf_error = speed_est(speed,config,theta)
eta_approx = eta_est()
speed_tf = (speed_tf_error + speed_tf_guess(speed))^2

U_IN = constant(Uin)
K_TRUE = constant(true_k)
Y_IN = constant(h1 - h2)

function condition(i, h_arr)
    i <= n
end

function euler_step(i,h,delt)
    update = h + [(-K_TRUE[i]*speed_tf[i]*h[1]*h[1] + U_IN[i])*delt; (K_TRUE[i]*speed_tf[i]*h[1]*h[1] - eta_approx^2 *h[2])*delt]
    return update

end

function forward_time(i, h_arr)
    delt = 0.1
    # An explicit integrator that uses forward in time calcualate the next time step

    h = read(h_arr, i - 1)
    h_next = h
    n_steps = 1.0 / delt
    h_next = euler_step(i-1,h_next,delt)
    h_next = euler_step(i-1,h_next,delt)
    h_next = euler_step(i-1,h_next,delt)
    h_next = euler_step(i-1,h_next,delt)
    h_next = euler_step(i-1,h_next,delt)
    h_next = euler_step(i-1,h_next,delt)
    h_next = euler_step(i-1,h_next,delt)
    h_next = euler_step(i-1,h_next,delt)
    h_next = euler_step(i-1,h_next,delt)
    h_next = euler_step(i-1,h_next,delt)
    h_arr = write(h_arr, i, h_next)
    i + 1, h_arr
end


h_arr = TensorArray(n)
h_arr = write(h_arr, 1, zeros(2))
i = constant(2, dtype = Int32)
_, h = while_loop(condition, forward_time, [i, h_arr])
h_est = set_shape(ADCME.stack(h), (n, 2))

# Compute the loss function
loss = sum(((h_est[:,1]-h_est[:,2]) - (h1-h2) )^2)

# Perform the optimisation
sess = Session(); init(sess)

optimisation_time = @elapsed begin
    BFGS!(sess, loss,100)
end

# Visualise
visualise_result_tf_speed(speed,config, theta, sess)
visualise_result_speed_time(time,speed,run(sess,speed_tf))
visualise_result_state_time(time,h1,h2,run(sess,h_est[:,1]),run(sess,h_est[:,2]))
println(run(sess,eta_approx))

