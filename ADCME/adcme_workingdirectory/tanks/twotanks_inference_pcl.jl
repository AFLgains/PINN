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

print(Y_IN)

function condition(i, h_arr)
    i <= n
end


function backward_time(i, h_arr)
    # An implicit integrator that uses backward in time calcualate the next time step
    A = tensor([1.0+dt*K_approx[i] 0.0;-K_approx[i] 1.0+dt*eta_approx^2])
    h = read(h_arr, i - 1)
    rhs = h + dt*U_IN[i]
    h_next = A \ rhs
    h_arr = write(h_arr, i, h_next)
    i + 1, h_arr
end

function forward_time(i, h_arr)
    delt = 0.1
    # An explicit integrator that uses forward in time calcualate the next time step
    A = tensor([-K_approx[i-1] 0.0;K_approx[i-1] -eta_approx^2])
    h = read(h_arr, i - 1)

    h_next = h
    n_steps = 1.0 / delt
    for steps in 1:n_steps
        h_next = h_next + delt*(A*h_next + U_IN[i-1])
    end

    h_arr = write(h_arr, i, h_next)
    i + 1, h_arr
end

function crank_nicolson(i,h_arr)
    Ai = tensor([-K_approx[i] 0.0;K_approx[i] -eta_approx^2])
    Aim1 = tensor([-K_approx[i-1] 0.0;K_approx[i-1] -eta_approx^2])
    hm1 = read(h_arr, i - 1)
    rhs = (I + dt/2*Aim1)*hm1 + dt/2*(U_IN[i] + U_IN[i-1])
    h_next = (I - dt/2*Ai) \ rhs
    h_arr = write(h_arr, i, h_next)
    i + 1, h_arr
end



function luenberg_state_estimator(i, h_arr)

    #  A L observer will use feedback from the data to self correct it's state estimation
    y_true = Y_IN[i-1] # The current observation


    delt = 0.5
    arr = [-K_approx[i-1] 0.0;K_approx[i-1] -eta_approx^2]
    A = tensor(arr)
    h_est = read(h_arr, i - 1)

    L = [0; -0.9]

    h_now = h_est
    n_steps = 1 / delt
    for steps in 1:n_steps
        h_predict = h_now + delt*(A*h_now + U_IN[i-1])
        y_pred = h_predict[1] - h_predict[2]
        h_now = h_predict - L*(y_pred - y_true)
    end

    h_arr = write(h_arr, i, h_now)
    i + 1, h_arr
end

h_arr = TensorArray(n)
h_arr = write(h_arr, 1, zeros(2))
i = constant(2, dtype = Int32)

if INTEGRATOR_METHOD == "backward"

    println("Using ",INTEGRATOR_METHOD)
    _, h = while_loop(condition, backward_time, [i, h_arr])

elseif INTEGRATOR_METHOD == "forward"

    println("Using ",INTEGRATOR_METHOD)
    _, h = while_loop(condition, forward_time, [i, h_arr])

elseif INTEGRATOR_METHOD == "crank"

    println("Using ",INTEGRATOR_METHOD)
    _, h = while_loop(condition, crank_nicolson, [i, h_arr])

elseif INTEGRATOR_METHOD == "runge2"

    println("Using ",INTEGRATOR_METHOD)
    _, h = while_loop(condition, runge_second_order, [i, h_arr])

elseif INTEGRATOR_METHOD == "luenberg"

    println("Using ",INTEGRATOR_METHOD)
    _, h = while_loop(condition, luenberg_state_estimator, [i, h_arr])

elseif INTEGRATOR_METHOD == "backward_nr"

    println("Using ",INTEGRATOR_METHOD)
    _, h = while_loop(condition, backward_newton_raphson, [i, h_arr])

else
    println("Integration  method not recognised")
    println("Using Crank nicolson")
    _, h = while_loop(condition, crank_nicolson, [i, h_arr])

end


h_est = set_shape(ADCME.stack(h), (n, 2))

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