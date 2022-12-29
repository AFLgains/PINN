using LinearAlgebra
using ADCME
using PyPlot

close("all")

n = 101
h = 1.0  / (n-1)
x = LinRange(0,1,n) |> collect
psi = @. (1 - 2*x)*(-100*x^2*(2*x - 2) - 200*x*(1 - x)^2)/(100*x^2*(1 - x)^2 + 1)^2 - 2 - 2/(100*x^2*(1 - x)^2 + 1)


noise = 0.0001
u_obs = (x.*(1.0.-x))
u_obs = u_obs.*(noise*randn(length(x)).+1.0)
u_obs = constant(u_obs)

theta = Variable(fc_init([1,20,20,20,1]))

# x is our input, u
# Create a nn
nn = squeeze(fc(u_obs,[20,20,20,1],theta)) + 1.0

# Measure the gradient of the nn
grad_nn = tf.gradients(nn,u_obs)[1]
# Calculate the residual
res =  grad_nn[2:end-1].*(u_obs[3:end] - u_obs[1:end-2])^2 / 4 / h^2 + nn[2:end-1].*(u_obs[3:end] + u_obs[1:end-2] - 2*u_obs[2:end-1])/h^2 - psi[2:end-1]
loss = sum(res^2)

# Optimize
sess = Session(); init(sess)
ADCME.BFGS!(sess, loss, 20000)
ADCME.BFGS!(sess, loss, 20000)
ADCME.BFGS!(sess, loss, 20000)
ADCME.BFGS!(sess, loss, 20000)
# visualise
u_test = LinRange(0,0.25,100) |> collect
u_test_sparse = LinRange(0,0.25,10) |> collect
X_pred = squeeze(fc(u_test_sparse, [20,20,20,1], theta)) + 1.0 # The modelled b(u)

figure(figsize=(10,4))
true_result = @. 1/(1+100*u_test^2) + 1
plot(u_test, true_result,label = "reference")
plot(u_test_sparse, run(sess, X_pred), "-o", markersize=5., label="Estimated")
legend(); xlabel("\$u\$"); ylabel("\$b(u)\$"); grid("on")
savefig("function_inverse_using_RM.png")
