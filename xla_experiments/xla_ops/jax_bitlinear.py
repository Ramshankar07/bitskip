"""JAX/XLA BitLinear implementation."""

jax_available = False
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    jax_available = True
except ImportError:
    pass


if jax_available:
    class JaxBitLinear:
        def __init__(self, in_features: int, out_features: int, activation_bits: int = 8):
            self.in_features = in_features
            self.out_features = out_features
            self.activation_bits = activation_bits
            key = jax.random.PRNGKey(0)
            self.weight = jax.random.normal(key, (out_features, in_features)) * 0.02
        
        @staticmethod
        @jit
        def forward(x, weight, activation_bits=8):
            x_scale = jnp.abs(x).max(axis=-1, keepdims=True).clip(min=1e-6)
            max_val = (1 << (activation_bits - 1)) - 1
            x_q = jnp.round(x * max_val / x_scale).clip(-max_val, max_val) * x_scale / max_val
            
            w_scale = jnp.abs(weight).mean().clip(min=1e-6)
            w_q = jnp.where(weight > 0.5 * w_scale, w_scale, 
                   jnp.where(weight < -0.5 * w_scale, -w_scale, 0.0))
            
            return jax.nn.relu(x_q @ w_q.T) ** 2
        
        def __call__(self, x):
            return self.forward(x, self.weight, self.activation_bits)
    
    
    class JaxHBitLinear:
        def __init__(self, in_features: int, out_features: int, activation_bits: int = 4):
            self.in_features = in_features
            self.out_features = out_features
            self.activation_bits = activation_bits
            key = jax.random.PRNGKey(0)
            self.weight = jax.random.normal(key, (out_features, in_features)) * 0.02
        
        @staticmethod
        @jit
        def hadamard(x):
            n = x.shape[-1]
            h = 1
            y = x
            while h < n:
                y = y.reshape(-1, n // (2 * h), 2, h)
                a, b = y[..., 0, :], y[..., 1, :]
                y = jnp.stack([a + b, a - b], axis=-2).reshape(-1, n)
                h *= 2
            return y.reshape(x.shape)
        
        @staticmethod
        @jit
        def forward(x, weight, activation_bits=4):
            x_ln = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
            
            x_scale = jnp.abs(x_ln).max(axis=-1, keepdims=True).clip(min=1e-6)
            max_val = (1 << (activation_bits - 1)) - 1
            x_q = jnp.round(x_ln * max_val / x_scale).clip(-max_val, max_val) * x_scale / max_val
            
            x_h = JaxHBitLinear.hadamard(x_q)
            
            w_scale = jnp.abs(weight).mean().clip(min=1e-6)
            w_q = jnp.where(weight > 0.5 * w_scale, w_scale,
                   jnp.where(weight < -0.5 * w_scale, -w_scale, 0.0))
            
            out = x_h @ w_q.T
            return JaxHBitLinear.hadamard(out)
        
        def __call__(self, x):
            return self.forward(x, self.weight, self.activation_bits)

else:
    JaxBitLinear = None
    JaxHBitLinear = None

