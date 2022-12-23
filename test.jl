using ADCME

a = Variable(2.0)

function f(t, y, var)
    [var*(y[2]-y[1]);
    y[1]*(27-y[3])-y[2];
    y[1]*y[2]-8/3*y[3]]
end

x0 = [1., 0., 0.]

forward = rk4(f, 30.0, 3000, x0, a)
sess = Session(); init(sess)
res_ = run(sess, forward)

b = Variable(4.)
forward_hyp = rk4(f, 30.0, 3000, x0, b)

loss = sum((res_ - forward_hyp)^2) 
sess = Session(); init(sess)
run(sess, loss)

sess = Session(); init(sess)
BFGS!(sess, loss)
b_ = run(sess, [b])


opt = AdamOptimizer(learning_rate=0.1).minimize(loss)
sess = Session(); init(sess)
for i = 1:30
    run(sess, opt)
    b_, loss_ = run(sess, [b, loss])
    @info i, b_, loss_
end


sess = Session(); init(sess)
run(sess, loss)
sess = Session(); init(sess)
BFGS!(sess, loss)
b_ = run(sess, [b])