using LinearAlgebra
using ADCME

n = 101
h = 1.0  / (n-1)
x = LinRange(0,1,n)[2:end-1]

# This is the variaible we want to optimise for
b = Variable(10.0)

# Constructing our computational graph
A = diagm(0=>2 / h^2*ones(n-2),-1=>-1/h^2*ones(n-3),1=>-1/h^2*ones(n-3))
B = (b*A + I)
f = @. 4*(2+x-x^2)
u = B\f
ue = u[div(n+1,2)] # Take the 51st value

# Constructing the loss function
loss = (ue - 1.0)^2

# Optimise
sess = Session(); init(sess)
BFGS!(sess,loss)

# Print results
println("Estimated b = ",run(sess,b))