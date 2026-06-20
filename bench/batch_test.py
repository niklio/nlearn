import sys, jax, jax.numpy as jnp, optax
from jax import random
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import nlearn.train as T

bs = int(sys.argv[1])
key = random.PRNGKey(0)
params = T.init_model(key)
schedule = optax.constant_schedule(1e-3)
opt = optax.adam(learning_rate=schedule)
opt_state = opt.init(params)
step_fn = T.make_train_step(opt)
batch = jnp.ones((bs, 513), dtype=jnp.int32)
_, _, loss = step_fn(params, opt_state, batch)
print(float(loss))
