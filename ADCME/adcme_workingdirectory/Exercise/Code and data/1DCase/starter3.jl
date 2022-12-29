# Instruction: Complete ____ and add additional codes at your will
using ADCME 
using PyPlot
using DelimitedFiles

m = 50
n = 50
dt = 1 / m
dx = 1 / n
F = zeros(m + 1, n)
xi = LinRange(0, 1, n + 1)[1:end - 1]
f = (x, t)->exp(-50(x - 0.5)^2)
for k = 1:m + 1
    t = (k - 1) * dt
    F[k,:] = dt * f.(xi, t)
end

xi_input = Array(reshape(xi, :, 1))
# TODO: Construct a neural network that maps xi_input to output
config = [20,20,20,1]
theta = Variable(fc_init([1,config...]))
κ = squeeze(fc(xi_input,config,theta))+ 1.0
# For squeeze: see the ADCME doc 
# https://kailaix.github.io/ADCME.jl/dev/tu_basic/
# and the corresponding function in TF
# https://www.tensorflow.org/api_docs/python/tf/squeeze

lamda = dt/dx^2*κ
mask = [2;ones(n-2)]
A = spdiag(n, -1 => -lamda[2:end], 0 => 1+2*lamda, 1 =>-lamda[1:end-1].*mask)

function condition(i, u_arr)
    i <= m + 1
end

function body(i, u_arr)
    u = read(u_arr, i - 1)
    rhs = u + F[i]
    u_next = A \ rhs
    u_arr = write(u_arr, i, u_next)
    i + 1, u_arr
end

F = constant(F)
u_arr = TensorArray(m + 1)
u_arr = write(u_arr, 1, zeros(n))
i = constant(2, dtype = Int32)
_, u = while_loop(condition, body, [i, u_arr])
u = set_shape(stack(u), (m + 1, n))

uc = readdlm("data_pcl.txt")
uc_wnoise = uc .* (1.0 .+ 0.1*randn(length(uc[:,1]),length(uc[1,:]) ))

# we magnify the loss function by 1e10 so that the optimizer does not stop too early. 
loss = sum((uc_wnoise - u[:,1:25])^2)

sess = Session(); init(sess)
BFGS!(sess, loss)

κval = run(sess, κ); plot(xi, κval)
xlabel("\$x\$"); ylabel("\$\\kappa\$")
savefig("ex3_reference_uc_w_lots_noise.png")