using LinearAlgebra
using ADCME
using PyPlot

close("all")

n = 101
h = 1.0  / (n-1)
x = LinRange(0,1,n) |> collect

# The correct answer
u = sin.(pi*x)
f= @. (1+u^2)/(1+2u^2) * π^2 * u + u

# Create the fully connected neural network layer
b = squeeze(fc(u[2:end-1],[20,20,1]))

# Residual
residual = -b.*(u[3:end] + u[1:end-2] - 2*u[2:end-1])/ h^2 + u[2:end-1] - f[2:end-1]
loss = sum(residual^2)

# Now solve using LBGFS
# Optimise
sess = Session(); init(sess)
BFGS!(sess,loss)

# Plot results
true_result = @. (1+x^2)/(1+2*x^2)
plot(x, true_result,label = "reference")
plot(u[2:end-1], run(sess, b), "o", markersize=5., label="Estimated")
legend(); xlabel("\$u\$"); ylabel("\$b(u)\$"); grid("on")
savefig("example_function_output.png")
#println("Estimated b = ",run(sess,b))