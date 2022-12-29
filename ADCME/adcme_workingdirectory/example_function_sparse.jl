using LinearAlgebra
using ADCME
using PyPlot

close("all")

n = 101
h = 1.0  / (n-1)
x = LinRange(0,1,n) |> collect

# The correct answer
u = sin.(pi*x)

# The forcing function
f= @. (1+u^2)/(1+2u^2) * π^2 * u + u

# IN this case we don't have lots of U's. So we are going to
# Simulate U as part of our solution. Let X be our solution
function residual_and_jac(theta,x)
    # Create a nn
    nn = squeeze(fc(reshape(x,:,1),[20,20,1],theta)) + 1.0
    u_full = vector(2:n-1,x,n) # Will basically pad the ends with zeros
    res = -nn.*(u_full[3:end] + u_full[1:end-2] - 2*u_full[2:end-1])/ h^2 + u_full[2:end-1] - f[2:end-1]
    J = gradients(res,x)
    res, J
end

theta = Variable(fc_init([1,20,20,1]))
ADCME.options.newton_raphson.rtol = 1e-4 # relative tolerance
ADCME.options.newton_raphson.tol = 1e-4 # absolute tolerance
ADCME.options.newton_raphson.verbose = true # print details in newton_raphson
u_est = newton_raphson_with_grad(residual_and_jac, constant(zeros(n-2)),theta) # will solve for x such that f(theta,x) = 0
residual = u_est[1:5:end] - u[2:end-1][1:5:end]
loss = sum(residual^2)

# Optimize
sess = Session(); init(sess)
BFGS!(sess, loss)

# visualise
u_test = reshape(x,:,1)
b = squeeze(fc(u_test, [20,20,1], theta)) + 1.0 # The modelled b(u)

figure(figsize=(10,4))
subplot(121)
true_result = @. (1+x^2)/(1+2*x^2)
plot(x, true_result,label = "reference")
plot(u_test, run(sess, b), "o", markersize=5., label="Estimated")
legend(); xlabel("\$u\$"); ylabel("\$b(u)\$"); grid("on")

subplot(122)
plot(x, u,label = "reference")
plot(x[2:end-1], run(sess, u_est), "o", markersize=5., label="Estimated")
plot(x[2:end-1][1:5:end],run(sess, u_est[1:5:end]),"x", markersize=10., label="Data")
savefig("example_function_sparse.png")
