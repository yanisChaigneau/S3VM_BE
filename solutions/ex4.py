def logistic_loss(y, t):
    gamma = 20
    loss = 1/gamma*np.log(1+np.exp(gamma*(1-y*t)))
    return loss

def exp_loss(y, t):
    s = 3
    loss = np.exp(-3*t**2)
    return loss

t1 = np.linspace(0,2,20)
t2 = np.linspace(-3, 3, 60)
y = 1

f, axs = plt.subplots(1,2,figsize=(15,6))
plt.subplot(1,2,1)
plt.plot(t1, logistic_loss(y, t1), label='Modified Logistic Loss')
plt.plot(t1, [hinge_loss(y, t) for t in t1], label='Hinge loss')
plt.xlabel('t')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t2, exp_loss(y, t2), label='Exponential loss')
plt.plot(t2, [effective_hinge_loss(y, t) for t in t2], label='Effective Hinge Loss')
plt.xlabel('t')
plt.legend()

fig.tight_layout()