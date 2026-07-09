#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -o jobout/smoke_gpu_%j.out
# Purpose: verify ics-env jax 0.9.1 + cuda12 plugin sees the GPU on a MIG
# slice, and that blackjax + jax_flows run there.
module load python/3.11.5 gcc cuda/12.6
source "$HOME/ics-env/bin/activate"
python -u - <<'EOF'
import jax, jax.numpy as jnp, jax.random as jr
print("backend:", jax.default_backend(), "devices:", jax.devices())
x = jr.normal(jr.key(0), (2048, 2048))
print("matmul ok:", float(jnp.trace(x @ x.T)))
import blackjax
mala = blackjax.mala(lambda x: -0.5 * jnp.sum(x**2), 0.1)
st = mala.init(jnp.zeros(4))
st, _ = jax.jit(mala.step)(jr.key(1), st)
print("blackjax on gpu ok")
from jax_flows import TimeConditionedMLP, cfm_sample
import numpy as np
m = TimeConditionedMLP(hidden_dims=(64, 64), output_dim=2)
params = m.init(jr.key(2), jnp.ones((1, 2)), jnp.ones((1,)))["params"]
s = cfm_sample(m, params, jr.key(3), (256, 2), n_steps=50, solver="heun")
print("cfm_sample heun ok:", np.asarray(s).shape, "finite:", bool(jnp.isfinite(s).all()))
print("SMOKE-PASS")
EOF
