using LinearAlgebra
using ADCME
using PyPlot

close("all")

n = 101
h = 1/(n-1)
x = LinRange(0,1,n) |> collect

# Create the fully connected neural network layer
X = fc(x,[20,20,20,1])^2

# Create the A Matrix as a function of the X (have to use spdiag, if the variables are part of the matrix
A = spdiag(n-2,
0=>(2*X[2:end-1]+X[1:end-2]+X[3:end])/2.0,
-1=>-(X[3:end-1] + X[2:end-2])/2.0,
1=>-(X[2:end-2] + X[3:end-1])/2.0)

# Create the right hand side
psi = @. 2*(x^2 -x*(2*x-1)+1)/(x^2 + 1)^2
b = psi*h^2

#solve for u using a linear solver
u_est = A\b[2:end-1] # A linear solver

# Create the loss function
u_true = @. x*(1−x)
res = (u_est - u_true[2:end-1])
loss = sum(res^2)


# Optimise
sess = Session(); init(sess)
BFGS!(sess,loss)

# Plot results
true_result = @. 1/(1+x^2)
plot(x, true_result,label = "reference")
plot(x, run(sess, X), "o", markersize=5., label="Estimated")
legend(); xlabel("\$u\$"); ylabel("\$b(u)\$"); grid("on")
savefig("function_inverse.png")