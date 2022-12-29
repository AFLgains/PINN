#=
In this script we'll be inferring how temperature and friction are related using the
residual minmisation technique''
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

CAR_DATA = "car_example_mpc.csv"
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

function visualise_result_friction_temp(temperature, true_friction, config, theta, sess)
    temp = LinRange(-40,40,81) |> collect
    est_friction = run(sess,squeeze(fc(temp,config,theta)^2)) # 1 for stablity
    figure(figsize=(10,4))
    plot(temperature,true_friction,"k.")
    plot(temp,est_friction,"r-")
    xlabel("Temperature"); ylabel("Friction")
    savefig("Temperature vs Friction.png")
end

function visualise_result_friction_time(time,true_friction,est_friction)
    figure(figsize=(10,4))
    plot(time,true_friction,"k.")
    plot(time,est_friction,"r-")
    xlabel("time"); ylabel("Friction")
    savefig("time vs Friction.png")
end


function main()
    df,n = load_data(CAR_DATA)
    time, temperature, velocity, force, true_friction = process_to_arrays(df)
    dt,Tc = calc_constants(time)
    vn, tn, fn = normalise_data(velocity,time,force,Tc)
    A = create_A(n)
    config,theta = friction_approximater_inits()
    K_approx = friction_est(temperature,config,theta)


    # Calculate the residual
    res = M*Vc / Tc / dt * A * vn .- (Uc*fn .- (K_approx.*(1.0 .- tf.exp(-Vc*vn)) .+ AERO_COEFF*Vc^2*vn.^2 ))
    loss = sqrt(sum(res^2))

    # Session + initialisation
    sess = Session(); init(sess)

    # Optimise!
    BFGS!(sess, loss, 1000)

    #learning_rate = 1e-2
    #opt = AdamOptimizer(learning_rate).minimize(loss)
    #sess = Session(); init(sess)
    #for i = 1:10000
    #    _, l = run(sess, [opt, loss])
    #    @info "Iteration $i, loss = $l"
    #end

    # Visualise
    visualise_result_friction_temp(temperature, true_friction, config, theta, sess)
    visualise_result_friction_time(time, true_friction, run(sess,K_approx))

end

main()