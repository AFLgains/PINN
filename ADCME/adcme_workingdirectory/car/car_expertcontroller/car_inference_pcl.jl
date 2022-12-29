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

CAR_DATA = "car_example.csv"
M = 500 # kg # car is half a ton
AERO_COEFF = 50/450
Vc = 80.0/3.6
Uc = 250.0

function load_data(file_name)
    df = CSV.read(file_name, DataFrame)
    return df,length(df[:,1])
end

function process_to_arrays(df)
    time = df.time |> collect
    temperature = df.temperature |> collect
    velocity = df.velocity |> collect
    force = df.force |> collect
    friction = df.friction_0 |> collect
    return time, temperature, velocity, force, friction
end

function calc_constants(time)
    Tc = length(time)
    dt = (time[2] - time[1]) / Tc
    return dt,Tc
end

function normalise_data(velocity,time,force,Tc)
    vn = velocity / Vc
    tn = time / Tc
    fn = force / Uc
    return vn, tn, fn
end

function create_A(n)
    A = diagm(-1 => [-0.5*ones(n-2);-1], 0 =>[-1;zeros(n-2);1], 1 =>[1;0.5*ones(n-2)] )
    return A
end


function friction_approximater_inits()
    config = [10,10,1]
    theta = Variable(fc_init([1,config...]))
    return config, theta
end

function friction_est(temp, config,theta)
    nn = squeeze(fc(temp,config,theta)^2) # 1 for stablity
    return nn
end


function residual_and_jac(theta,vn)
    # Create a nn

    K_approx = friction_est(temperature[2:end], config, theta)
    u_full = [constant(zeros(1));vn]
    res = M*Vc / Tc / dt * (u_full[2:end] - u_full[1:end-1]) .- (Uc*fn[2:end] .- (K_approx.*(1.0 .- tf.exp(-Vc*u_full[2:end])) .+ AERO_COEFF*Vc^2*u_full[2:end].*u_full[2:end] ))
    #res = M*Vc / Tc / dt*(u_full[2:end] - u_full[1:end-1]) - K_approx.*(1.0 .- tf.exp(-Vc*u_full[2:end])) .+ AERO_COEFF*Vc^2*u_full[2:end].*u_full[2:end]
    J = gradients(res,vn)
    return res, J
end

function visualise_result_friction_temp(temperature, true_friction, config, theta, sess)
    temp = LinRange(-40,40,81) |> collect
    est_friction = run(sess,squeeze(fc(temp,config,theta)^2)) # 1 for stablity
    figure(figsize=(10,4))
    plot(temperature,true_friction,"k.")
    plot(temp,est_friction,"r-")
    xlabel("Temperature"); ylabel("Friction")
    savefig("Temperature vs Friction_pcl.png")
end

function visualise_result_friction_time(time,true_friction, est_friction)
    figure(figsize=(10,4))
    plot(time,true_friction,"k.")
    plot(time,est_friction,"r-")
    xlabel("time"); ylabel("Friction")
    savefig("time vs Friction_pcl.png")
end


function visualise_result_vel_time(time,true_velocity,est_vel)
    figure(figsize=(10,4))
    plot(time,true_velocity,"k.")
    plot(time[2:end],est_vel,"r-")
    xlabel("time"); ylabel("Velocity")
    savefig("time vs velocity_pcl.png")
end



df,n = load_data(CAR_DATA)
time, temperature, velocity, force, true_friction = process_to_arrays(df)
dt,Tc = calc_constants(time)
true_vn, tn, fn = normalise_data(velocity,time,force,Tc)
A = create_A(n)
config,theta = friction_approximater_inits()

ADCME.options.newton_raphson.rtol = 1e-4 # relative tolerance
ADCME.options.newton_raphson.tol = 1e-4 # absolute tolerance
ADCME.options.newton_raphson.verbose = true # print details in newton_raphson

vn_est = newton_raphson_with_grad(residual_and_jac, constant(zeros(n-1)),theta) # will solve for x such that f(theta,x) = 0
residual = vn_est - true_vn[2:end]

# Calculate the residual
loss = sum(residual^2)

# Session + initialisation
sess = Session(); init(sess)

println(run(sess,loss))

# Optimise!
BFGS!(sess, loss,400)

#learning_rate = 1e-2
#opt = AdamOptimizer(learning_rate).minimize(loss)
#sess = Session(); init(sess)
#for i = 1:10000
#    _, l = run(sess, [opt, loss])
#    @info "Iteration $i, loss = $l"
#end

# Visualise
visualise_result_friction_temp(temperature, true_friction, config, theta, sess)
est_friction = run(sess,squeeze(friction_est(temperature, config,theta)))
visualise_result_friction_time(time, true_friction, est_friction)
visualise_result_vel_time(time,true_vn,run(sess,vn_est))

