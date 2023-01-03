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

SAG_DATA = "simple_saglike_ode.csv"


function load_data(file_name)
    df = CSV.read(file_name, DataFrame)
    return df,length(df[:,1])
end

function process_to_arrays(df)
    time = df.time |> collect
    x_r = df.x_r |> collect
    x_w = df.x_w |> collect
    u_r = df.u_r |> collect
    u_w = df.u_w |> collect
    true_k = df.K_ext |> collect
    a = df.a |> collect
    return time, x_r, x_w, u_r, u_w, true_k, a
end

function calc_dt(time)
    dt = time[2] - time[1]
    return dt
end


function create_A(K_approx,dt,n)
    #mask = [0;ones(n-1)]
    A = spdiag(n,
    -1 => -ones(n-1)/dt,
    0 =>K_approx[1:end] .+ 1/dt*ones(n)
    )
    return A
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

function visualise_result_friction_temp(a, true_k, config, theta, sess)
    a_range = LinRange(-0.8,0.8,101) |> collect
    est_k = run(sess,friction_est(a_range, config,theta)) # 1 for stablity
    figure(figsize=(10,4))
    plot(a,true_k,"k.")
    plot(a_range,est_k,"r-")
    xlabel("a"); ylabel("K")
    savefig("a vs k_pcl.png")
end

function visualise_result_friction_time(time,true_k,est_k)
    figure(figsize=(10,4))
    plot(time,true_k,"k.")
    plot(time,est_k,"r-")
    xlabel("time"); ylabel("k")
    savefig("time vs k_pcl.png")

    figure(figsize=(10,4))
    plot(est_k,true_k,"k.")
    xlabel("est_k"); ylabel("true_k")
    savefig("est_k vs true_k_pcl.png")
end

function visualise_result_xr_time(time,true_xr,xr_est)
    figure(figsize=(10,4))
    plot(time,true_xr,"k.")
    plot(time,xr_est,"r-")
    xlabel("time"); ylabel("xr")
    savefig("time vs x_r_pcl.png")
end

function solve_xr_smf(A,u_r) # Sparse matrix factorisation
    res = A \ u_r
    return res[2:end]
end

function residual_and_jac(theta,x_r)
    # Create a nn

    K_approx = friction_est(a[2:end], config, theta)
    x_full = [constant(zeros(1));x_r]
    res = K_approx.*x_full[2:end] .- u_r[2:end] .+ (x_full[2:end] .- x_full[1:end-1])/dt
    J = gradients(res,x_r)
    return res, J
end



df,n = load_data(SAG_DATA)
time, x_r, x_w, u_r, u_w, true_k, a = process_to_arrays(df)
dt = calc_dt(time)

config,theta = friction_approximater_inits()
K_approx = friction_est(a,config,theta)
A = create_A(K_approx,dt,n)

ADCME.options.newton_raphson.rtol = 1e-4 # relative tolerance
ADCME.options.newton_raphson.tol = 1e-4 # absolute tolerance
ADCME.options.newton_raphson.verbose = true # print details in newton_raphson

xr_est = solve_xr_smf(A,u_r)
#xr_est = newton_raphson_with_grad(residual_and_jac, constant(zeros(n-1)),theta) # will solve for x such that f(theta,x) = 0
residual = xr_est- x_r[2:end]

# Calculate the residual
loss = sum(residual^2)

# Session + initialisation
sess = Session(); init(sess)

# Optimise!
BFGS!(sess, loss, 15000)

#learning_rate = 1e-2
#opt = AdamOptimizer(learning_rate).minimize(loss)
#sess = Session(); init(sess)
#for i = 1:20000
#    _, l = run(sess, [opt, loss])
#    @info "Iteration $i, loss = $l"
#end

# Visualise
visualise_result_friction_temp(a, true_k, config, theta, sess)
visualise_result_friction_time(time, true_k, run(sess,K_approx))
visualise_result_xr_time(time[2:end],x_r[2:end],run(sess,xr_est))

