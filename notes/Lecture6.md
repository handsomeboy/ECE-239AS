### Lecture 6

#### Neural Nets Overview

- First layer: $h_1 = f(W_1 x + b)$ and so on
- Last layer is just a linear layer
- If the activation is just the identity, devolves into a linear classifier



#### Sigmoid Activation

- $\sigma(x) = \frac{1}{1 + \exp(-x)}$
- Derivative: $\sigma(x)(1 - \sigma(x))$
- In general, we want gradients that are large enough and predictable enough to use in the learning algorithm, but sigmoid gradients are small for large x and small x
- Sigmoid gradient is maximized when $x = 0$ with value $0.25$. 
- Pros:
  - around x = 0, the unit behaves linearly -> good learniing speed
  - everywhere differentiable
- Cons:
  - it is saturating
  - If we have $f = \sigma(w^Tx + b)$ and define $z = w^tx + b$
  - Then we can compute $\frac{df}{dw}$ using the chain rule:
    - $\frac{df}{dw} = \frac{df}{dz} \frac{dz}{dw}$
    - illustrates the saturating gradient problem
  - Something about the activations being positive or negative -> zigzagging during GD
  - Not zero centered -> **zig-zagging gradients**

#### Tanh activation

- $tanh(x) = 2\sigma(x) - 1$
- Zero-centered: no zig zagging during gradient descent
- At tails of $x$, gradient can still vanish

#### ReLu

- $\max(0,x)$
- zero for x < 0, positive for x > 0
- Derivative: 1 if x > 0 else 0
  - Not differentiable at x = 0, so we can define a value here (between 0 and 1)
- Pros:
  - learning converges after (AlexNet paper said ReLus trained several times faster)
  - When unit is active, it behaves linearly
  - No saturation, larger gradients
  - Learning doesn't happen when input $x$ is less than or equal to 0
    - "dying ReLu problem"
      - can resolve by using Leaky or max out

#### Softplus

- $\log(1 + \exp(z))$ -> "smooth" version of ReLu
  - does not have sharp change/0 graident problem

#### Leaky ReLu

- $f(x) = max(\alpha x, x)$
- PreLu: can tune alpha as a parameter on our data
- ELU: $f(x) = max(\alpha(\exp(x)-1),x)$
- Maxout unit:  $f(x) = \max(w_1x + b_1, w_2x + b_2)$





#### Picking a Loss Function

- MSE: $(y^i - \sigma(z^i))$
- when $z$ is large and negative or large and positive, then the gradients are practically zero
  - learning will not occur
- If you write out the derivatives, then it's a function of $\sigma(z)(1 - \sigma(z))$
- CE:
- Better cost function: when z is very negative, we'll have large gradients, grads will only go to 0 when $z$ is large (and y is 1)
- TODO: need to form this a lot better



### #### Training with Gradient Descent

- Pet peeve: we train with gradient descent, not with backpropagation
- ​

