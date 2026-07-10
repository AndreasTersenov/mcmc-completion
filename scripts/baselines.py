"""Baselines 2-4 (frozen list) on the pre-registered 12-target subset
(each family x d in {2,4}, idx 0; d<=4 = the model's operating regime and
banana-d>=8 is excluded by the audit). B1 (untrained floor) runs separately
via eval_full on an init checkpoint.
- B2 bespoke per-target FM at 10 H100-minutes (300k steps, calibrated)
- B3 energy-MLP fit on the context (x,E) + MALA on the fitted energy
- B4 blackjax MCLMC at wall-clock matched to ICS inference (measured in-job)
Metrics per baseline: SW2^2 vs exact, mode_recovery; B2 also certificate.
Writes results/baselines.json."""
import json, os, pickle, sys, time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jax_flows import TimeConditionedMLP, cfm_loss, cfm_sample
import blackjax
from ics.cfm import cnf_logpdf, cond_cfm_sample
from ics.certificate import snis_certificate
from ics.context import Context, whiten_invert
from ics.eval import mode_recovery
from ics.models import ICSModel
from ics.train import load_checkpoint
from ics.zoo import logpdf, sample_x
from stage0.sliced_w2 import sliced_w2_squared

N_EVAL = 4096
SUBSET = [(f, d, 0) for f in ("gmm", "dwell", "funnel", "warp", "banana", "funnelmix")
          for d in (2, 4)]

es = pickle.load(open(os.path.join(os.environ["SCRATCH"], "ics-zoo", "eval",
                                   "eval_set.pkl"), "rb"))
rows_by_key = {(r["family"], r["d"], r["idx"]): r for r in es}

def sw2_and_modes(target, x_gen, seed):
    fresh = np.asarray(sample_x(jr.key(seed), target, 2 * N_EVAL), np.float64)
    sw2 = float(sliced_w2_squared(np.asarray(x_gen, np.float64), fresh,
                                  n_proj=128, rng=np.random.default_rng(seed)))
    return sw2, mode_recovery(target, np.asarray(x_gen))

def b2_bespoke(target, ctx, seed):
    d = target.d
    x_tr = ((sample_x(jr.key(seed), target, 200_000) - ctx.mu) / ctx.sigma
            ).astype(jnp.float32)
    m = TimeConditionedMLP(hidden_dims=(256, 256, 256), output_dim=d)
    p = m.init(jr.key(seed+1), jnp.ones((1, d), jnp.float32), jnp.ones((1,), jnp.float32))["params"]
    tx = optax.adam(optax.cosine_decay_schedule(2e-3, 300_000)); o = tx.init(p)
    @jax.jit
    def st(p, o, k):
        kb, kl = jr.split(k)
        idx = jr.randint(kb, (512,), 0, x_tr.shape[0])
        l, g = jax.value_and_grad(cfm_loss)(p, x_tr[idx], kl, m)
        up, o = tx.update(g, o); return optax.apply_updates(p, up), o, l
    for k in jr.split(jr.key(seed+2), 300_000):
        p, o, _ = st(p, o, k)
    p64 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), p)
    sW = cfm_sample(m, p64, jr.key(seed+3), (2*N_EVAL, d), n_steps=100, solver="heun")
    x_gen = ctx.mu + ctx.sigma * sW
    def vf(x, t):
        return m.apply({"params": p64}, x, jnp.full((x.shape[0],), t, dtype=x.dtype))
    logq = cnf_logpdf(vf, sW, n_steps=100)
    logp = logpdf(target, x_gen) + jnp.log(ctx.sigma).sum()
    cert = snis_certificate(np.asarray(logp), np.asarray(logq))
    sw2, mr = sw2_and_modes(target, x_gen, seed+4)
    return dict(sw2=sw2, mode_recovery=mr, **cert)

def b3_energy_mala(target, ctx, seed):
    d = target.d
    xs, E = ctx.x_raw.astype(jnp.float32), ctx.energy.astype(jnp.float32)
    mu, sd = E.mean(), E.std() + 1e-8
    m = TimeConditionedMLP(hidden_dims=(128, 128), output_dim=1)  # t unused
    p = m.init(jr.key(seed), jnp.ones((1, d), jnp.float32), jnp.ones((1,), jnp.float32))["params"]
    tx = optax.adam(1e-3); o = tx.init(p)
    @jax.jit
    def st(p, o, k):
        def loss(p):
            pred = m.apply({"params": p}, xs, jnp.zeros(xs.shape[0], jnp.float32))[:, 0]
            return jnp.mean((pred - (E - mu) / sd) ** 2)
        l, g = jax.value_and_grad(loss)(p)
        up, o = tx.update(g, o); return optax.apply_updates(p, up), o, l
    for k in jr.split(jr.key(seed+1), 3000):
        p, o, _ = st(p, o, k)
    def ld(x):  # fitted log-density (unnormalized)
        e = m.apply({"params": p}, x[None].astype(jnp.float32),
                    jnp.zeros(1, jnp.float32))[0, 0]
        return -(e * sd + mu).astype(jnp.float64)
    mala = blackjax.mala(ld, 0.05)
    keys = jr.split(jr.key(seed+2), 4)
    chains = []
    for c in range(4):
        st_ = mala.init(5.0 * jr.normal(jr.fold_in(jr.key(seed+3), c), (d,)))
        def step(s, k): s, _ = mala.step(k, s); return s, s.position
        _, pos = jax.lax.scan(step, st_, jr.split(keys[c], 3000))
        chains.append(np.asarray(pos[1000::2]))
    x_gen = np.concatenate(chains)[:2*N_EVAL]
    sw2, mr = sw2_and_modes(target, x_gen, seed+5)
    return dict(sw2=sw2, mode_recovery=mr)

def b4_mclmc(target, ctx, seed, budget_s):
    d = target.d
    ld = lambda x: logpdf(target, x[None, :])[0]
    key = jr.key(seed)
    state = blackjax.mcmc.mclmc.init(position=jnp.zeros(d), logdensity_fn=ld, rng_key=key)
    kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=ld, integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        inverse_mass_matrix=inverse_mass_matrix)
    t0 = time.time()
    (state_t, params_t), _ = blackjax.adaptation.mclmc_adaptation.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel, num_steps=2000, state=state, rng_key=jr.fold_in(key, 1))
    kern = kernel(params_t.inverse_mass_matrix if hasattr(params_t, "inverse_mass_matrix") else jnp.ones(d))
    samples = []
    st_ = state_t
    while time.time() - t0 < budget_s and len(samples) < 4 * N_EVAL:
        def step(s, k):
            s, _ = kern(k, s, params_t.L, params_t.step_size)
            return s, s.position
        st_, pos = jax.lax.scan(step, st_, jr.split(jr.fold_in(key, len(samples)+2), 512))
        samples.append(np.asarray(pos))
    x_gen = np.concatenate(samples)[-2*N_EVAL:]
    sw2, mr = sw2_and_modes(target, x_gen, seed+6)
    return dict(sw2=sw2, mode_recovery=mr, seconds=time.time()-t0)

def main():
    # measure ICS inference wall-clock for B4 matching
    model = ICSModel(n_attn=2)
    params = jax.tree_util.tree_map(jnp.asarray, load_checkpoint(
        os.path.join(os.environ["SCRATCH"], "ics-zoo", "ckpt_train4.pkl"))["params"])
    r0 = rows_by_key[("gmm", 2, 0)]
    ctx0 = Context(**{k: jnp.asarray(v) for k, v in r0["context_t5"]._asdict().items()})
    t0 = time.time()
    _ = np.asarray(cond_cfm_sample(model, params, ctx0.tokens.astype(jnp.float64),
                                   jr.key(0), n=2*N_EVAL, n_steps=100))
    ics_seconds = time.time() - t0
    print(f"ICS inference wall-clock (2N samples): {ics_seconds:.1f}s", flush=True)

    out = {"ics_seconds": ics_seconds, "rows": []}
    for n, (f, d, i) in enumerate(SUBSET):
        row = rows_by_key[(f, d, i)]
        t = row["target"]
        ctx = Context(**{k: jnp.asarray(v) for k, v in row["context_t5"]._asdict().items()})
        rec = dict(family=f, d=d, idx=i)
        t1 = time.time()
        rec["b2"] = b2_bespoke(t, ctx, 50_000 + 100*n)
        rec["b3"] = b3_energy_mala(t, ctx, 60_000 + 100*n)
        rec["b4"] = b4_mclmc(t, ctx, 70_000 + 100*n, budget_s=max(ics_seconds, 5.0))
        rec["seconds"] = round(time.time() - t1, 1)
        out["rows"].append(rec)
        print(f"[{n+1}/12] {f}-d{d}: b2 sw2={rec['b2']['sw2']:.3f} ess={rec['b2']['ess_frac_2n']:.3f} | "
              f"b3 sw2={rec['b3']['sw2']:.3f} | b4 sw2={rec['b4']['sw2']:.3f} [{rec['seconds']}s]", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
                                     "baselines.json"), "w"), indent=2)
    print("BASELINES-DONE")

if __name__ == "__main__":
    main()
