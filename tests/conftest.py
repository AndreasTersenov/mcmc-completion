import jax

# Certificate math needs f64; training code opts into f32 explicitly.
jax.config.update("jax_enable_x64", True)

import zlib


def dseed(*parts):
    """Deterministic seed from labels (builtin hash() is salted per process —
    tests seeded with it are nondeterministic; same lesson as run_m2.dseed)."""
    return zlib.crc32("|".join(map(str, parts)).encode())
