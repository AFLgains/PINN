using LinearAlgebra
using ADCME
using PyPlot

close("all")

n = 101
h = 1.0  / (n-1)
x = LinRange(0,1,n) |> collect
psi = @. (1 - 2*x)*(-100*x^2*(2*x - 2) - 200*x*(1 - x)^2)/(100*x^2*(1 - x)^2 + 1)^2 - 2 - 2/(100*x^2*(1 - x)^2 + 1)
psi = psi[2:end-1]

noise = 0.1
u_obs = (x.*(1.0.-x))
u_obs = u_obs.*(noise*randn(length(x)).+1.0)
u_obs = u_obs[2:end-1]


function residual_and_jac(theta,x)
    # x is our input, u
    # Create a nn
    nn = squeeze(fc(reshape(x,:,1),[20,20,20,1],theta)) + 1.0

    # Measure the gradient of the nn
    grad_nn = tf.gradients(nn,x)[1]

    # pad the U
    u_full = vector(2:n-1,x,n) # Will basically pad the ends with zeros

    # Calculate the residual
    res = grad_nn.*((u_full[3:end] - u_full[1:end-2])/(2*h))^2 + nn.*(u_full[3:end] + u_full[1:end-2] - 2*u_full[2:end-1])/h^2 - psi
    #res = -grad_nn.*(u_full[3:end] + u_full[1:end-2] - 2*u_full[2:end-1])/ h^2 + u_full[2:end-1]
    # Calduate the jacobian
    J = gradients(res,x)
    return res, J
end

theta = Variable(fc_init([1,20,20,20,1]))
ADCME.options.newton_raphson.rtol = 1e-4 # relative tolerance
ADCME.options.newton_raphson.tol = 1e-4 # absolute tolerance
ADCME.options.newton_raphson.verbose = true # print details in newton_raphson
u_est = newton_raphson_with_grad(residual_and_jac, constant(zeros(n-2)),theta) # will solve for x such that f(theta,x) = 0

n_freq = 1

residual = u_est[1:n_freq:end] - u_obs[1:n_freq:end]
loss = sum(residual^2)

# Optimize
sess = Session(); init(sess)
BFGS!(sess, loss)

# visualise
u_test = LinRange(0,0.25,100) |> collect
X_pred = squeeze(fc(u_test, [20,20,20,1], theta)) + 1.0 # The modelled b(u)

figure(figsize=(10,4))
subplot(121)
true_result = @. 1/(1+100*u_test^2) + 1
plot(u_test, true_result,label = "reference")
plot(u_test, run(sess, X_pred), "-o", markersize=5., label="Estimated")
legend(); xlabel("\$u\$"); ylabel("\$b(u)\$"); grid("on")

subplot(122)
plot(x[2:end-1], u_obs,label = "reference")
plot(x[2:end-1], run(sess, u_est), "o", markersize=5., label="Estimated")
plot(x[2:end-1][1:n_freq:end],run(sess, u_est[1:n_freq:end]),"x", markersize=10., label="Data")
savefig("function_inverse_using_NR.png")
