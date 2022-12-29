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

a = Variable(1.0)
b = Variable(1.0)

# TODO: Construct κ using `a` and `b`
κ = a + b*xi

# TODO: Construct `A` using `spdiag`
#= 
Hint: the following syntax might be useful 
∘ Concatenate two tensors: [o1;o2]
∘ Elementwise multiplication: o1 .* o2 
∘ Indexing: o1[1], o1[1:3], o1[3:end-1] =#
lamda = dt/dx^2*κ
mask = [2;ones(n-2)]
A = spdiag(n, -1 => -lamda[2:end], 0 => 1+2*lamda, 1 =>-lamda[1:end-1].*mask)


function condition(i, u_arr)
    i <= m + 1
end

function body(i, u_arr)
    u = read(u_arr, i - 1)
    # TODO: Compute u_next using u and F[i]
    rhs =  u + F[i,:]
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

# readdlm reads a matrix from file using "Delimited Files"
# https://docs.julialang.org/en/v1/stdlib/DelimitedFiles/
uc = readdlm("data.txt")[:]

# TODO: Formulate the loss function
res = squeeze(u[:,1]) - uc
loss = sum(res^2)

sess = Session(); init(sess)
BFGS!(sess, loss)
@show run(sess, [a, b])