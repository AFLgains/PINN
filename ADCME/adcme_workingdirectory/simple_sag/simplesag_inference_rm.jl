
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

function create_res(x_r,K_approx,u_r,dt)
    n = length(x_r)
    res = (x_r[2:end] - x_r[1:end-1])/dt + K_approx[2:end].*x_r[2:end] - u_r[2:end]
    return res
end


function friction_approximater_inits()
    config = [10,10,1]
    theta = Variable(fc_init([1,config...]))
    return config, theta
end

function friction_est(a, config,theta)
    nn = squeeze(fc(a,config,theta)) # 1 for stablity
    return nn
end

function visualise_result_friction_temp(a, true_k, config, theta, sess)
    a_range = LinRange(-1,1,101) |> collect
    est_k = run(sess,friction_est(a_range, config,theta)) # 1 for stablity
    figure(figsize=(10,4))
    plot(a,true_k,"k.")
    plot(a_range,est_k,"r-")
    xlabel("a"); ylabel("K")
    savefig("a vs k.png")
end

function visualise_result_friction_time(time,true_k,est_k)
    figure(figsize=(10,4))
    plot(time,true_k,"k.")
    plot(time,est_k,"r-")
    xlabel("time"); ylabel("k")
    savefig("time vs k.png")


        figure(figsize=(10,4))
    plot(est_k,true_k,"k.")
    xlabel("est_k"); ylabel("true_k")
    savefig("est_k vs true_k.png")
end


function main()
    df,n = load_data(SAG_DATA)
    time, x_r, x_w, u_r, u_w, true_k, a = process_to_arrays(df)
    dt = calc_dt(time)

    config,theta = friction_approximater_inits()
    K_approx = friction_est(a,config,theta)

    res = create_res(x_r,K_approx,u_r,dt)

    # Calculate the residual
    loss = sqrt(sum(res^2))

    # Session + initialisation
    sess = Session(); init(sess)

    # Optimise!
    #BFGS!(sess, loss, 10)

    learning_rate = 1e-2
    opt = AdamOptimizer(learning_rate).minimize(loss)
    sess = Session(); init(sess)
    for i = 1:5000
        _, l = run(sess, [opt, loss])
        @info "Iteration $i, loss = $l"
    end

    # Visualise
    visualise_result_friction_temp(a, true_k, config, theta, sess)
    visualise_result_friction_time(time, true_k, run(sess,K_approx))

end

main()