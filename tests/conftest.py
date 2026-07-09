import jax

# Certificate math needs f64; training code opts into f32 explicitly.
jax.config.update("jax_enable_x64", True)
