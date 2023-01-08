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
MAX_SPEED_INFLUENCE = 0.25

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

# Initialisation
function speed_approximater_inits()
    config = [10,10,1]
    theta = Variable(fc_init([1,config...]))
    return config, theta
end

function friction_approximater_inits()
    config = [10,10,1]
    theta = Variable(fc_init([1,config...]))
    return config, theta
end


# Friction estiamtion
function friction_est(a, config,theta)
    nn = squeeze(2*sigmoid(fc(a,config,theta) ) )
    return nn
end

function speed_3d_est(a, config,theta)
    nn = squeeze(-50*sigmoid(fc(a,config,theta) ) )
    return nn
end


# Speed estimation
function speed_est(speed,speed_col,speed_tf_estimate)
    return interp1(speed_col,squeeze(speed_tf_estimate),speed)
end

# Eta estimation
function eta_est()
    eta =tf.exp(Variable(1.0))
    return eta
end

# Functions to visualise the results
function visualise_result_tf_speed(true_speed,speed_col,speed_est)

    figure(figsize=(10,4))

    plot(speed_col,speed_est,"r-")
    plot(true_speed,true_speed.^2 .*(1.0.-true_speed.^2),"k.")
    plot(speed_col,speed_col.^2 .*(1.0.-speed_col.^2),"r--")

    xlabel("speed"); ylabel("tf")
    savefig("results/physics_speed/speed_tf.png")
end

function visualise_result_friction_temp(temp, true_k, config, theta, sess)
    a_range = LinRange(-60,20,101) |> collect
    est_k = run(sess,friction_est(a_range, config,theta)) # 1 for stablity
    figure(figsize=(10,4))
    plot(temp,true_k,"k.")
    plot(a_range,est_k,"r-")
    xlabel("a"); ylabel("K")
    savefig("results/physics_speed/temp vs k.png")
end

function visualise_result_speed_time(time,speed,est_speed_tf)
    figure(figsize=(10,4))
    plot(time,speed.^2 .*(1.0 .- speed.^2),"k.")
    plot(time,est_speed_tf,"r-")
    xlabel("time"); ylabel("k")
    ylim((0,2))
    savefig("results/physics_speed/time vs speed_tf.png")
end

function visualise_result_friction_time(time,true_k,est_k)
    figure(figsize=(10,4))
    plot(time,true_k,"k.")
    plot(time,est_k,"r-")
    xlabel("time"); ylabel("k")
    ylim((0,2))
    savefig("results/physics_speed/time vs k.png")

    figure(figsize=(10,4))
    plot(est_k,true_k,"k.")
    xlabel("est_k"); ylabel("true_k")
    savefig("results/physics_speed/est_k vs true_k.png")
end

function visualise_result_state_time(time,true_h1,true_h2,h1_est,h2_est)
    figure(figsize=(10,4))
    plot(time,true_h1,"k.")
    plot(time,h1_est,"r-")
    xlabel("time"); ylabel("h1")
    savefig("results/physics_speed/time vs h1.png")

    figure(figsize=(10,4))
    plot(time,true_h2,"k.")
    plot(time,h2_est,"r-")
    xlabel("time"); ylabel("h2")
    savefig("results/physics_speed/time vs h2.png")
end

function visualise_result_derivative(speed,derivative)
    figure(figsize=(10,4))
    plot(speed,derivative,"r--")
    plot(speed,-24*speed,"r-")
    xlabel("speed"); ylabel("derivative")
    savefig("results/physics_speed/speed vs derivative.png")
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

    A_s[4,2] = -5/2
    A_s[4,3] = 9
    A_s[4,4] = -12
    A_s[4,5] = 7
    A_s[4,6] = -3/2

    return A_s
end

# Load data
df,n = load_data(TANK_DATA)
time, temperature, h1, h2, Uin, speed, true_k = process_to_arrays(df)
dt = calc_dt(time)


# We will constrain a the function of speed to be
# pinned at 0,0, with 0 derivatice at 0 and a positive double derivative at 0

n_col = 51
speed_col = LinRange(0,1,n_col)|> collect
del_s = speed_col[2] - speed_col[1]
A_eq = create_As(n_col,del_s)
rhs_second_d = constant(2.0/15.0/del_s)
#config_s, theta_s = speed_approximater_inits()
#speed_coeffs_rhs = speed_3d_est(speed_col[2:n_col-2], config_s,theta_s)
speed_coeffs_rhs = Variable(sqrt.(20.0*speed_col[2:n_col-2]))
rhs_third_d = -(speed_coeffs_rhs)^2*del_s^3
rhs_speed = [constant(zeros(2,1));reshape(rhs_second_d,:,1); reshape(rhs_third_d,:,1) ]
speed_tf_estimate = A_eq \ rhs_speed
speed_tf = speed_est(speed,speed_col,speed_tf_estimate)


config_f,theta_f = friction_approximater_inits()
K_approx = friction_est(temperature,config_f,theta_f)

eta_approx = eta_est()

# Constants
U_IN = constant(Uin)
Y_IN = constant(h1 - h2)

# Main loop
function condition(i, h_arr)
    i <= n
end

function backward_linearised(i, h_arr)

    # An explicit integrator that uses forward in time calcualate the next time step

    h = read(h_arr, i - 1)

    # Forward pass
    Fim1 = [(-K_approx[i-1]*speed_tf[i-1]*h[1]*h[1] + U_IN[i-1]); (K_approx[i-1]*speed_tf[i-1]*h[1]*h[1] - eta_approx *h[2])]

    # Jacobian
    a = -2*K_approx[i-1]*speed_tf[i-1]*h[1]
    b = 0
    c = 2*K_approx[i-1]*speed_tf[i-1]*h[1]
    d = -eta_approx
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
_, h = while_loop(condition, backward_linearised, [i, h_arr])
h_est = set_shape(ADCME.stack(h), (n, 2))

# Compute the loss function
loss = sum(((h_est[:,1]-h_est[:,2]) - (h1-h2) )^2)
#loss =  sum((h_est[:,1]-h1)^2) + sum((h_est[:,2]-h2)^2)

# Perform the optimisation
sess = Session(); init(sess)
optimisation_time = @elapsed begin
    BFGS!(sess, loss,1000)
end

# Visualise
visualise_result_tf_speed(speed,speed_col,run(sess,squeeze(speed_tf_estimate) ) )
visualise_result_speed_time(time,speed,run(sess,speed_tf))
visualise_result_friction_temp(temperature, true_k, config_f, theta_f, sess)
visualise_result_friction_time(time, true_k, run(sess,K_approx))
visualise_result_state_time(time,h1,h2,run(sess,h_est[:,1]),run(sess,h_est[:,2]))
visualise_result_derivative(speed_col[2:n_col-2],run(sess,rhs_third_d/del_s^3))

println(run(sess,eta_approx))
#println(run(sess,rhs_speed))
#println(run(sess,A_speed*speed_tf_estimate[2:end-1]))
