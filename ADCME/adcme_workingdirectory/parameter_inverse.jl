using LinearAlgebra
using ADCME

n = 100
h = 1/n
X0 = Variable(10.0)


A = X0*diagm(0=>2/h^2*ones(n-1),-1=>-1/h^2*ones(n-2),1=>-1/h^2*ones(n-2))
psi = 2*ones(n-1)
u = A\psi # A linear solver
loss = (u[50] - 0.25)^2

# Optimise
sess = Session(); init(sess)
BFGS!(sess,loss)



# Print results
println("Estimated b = ",run(sess,X0))