import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from remarc.agents.tianshou_agent import train_wf_landscapes
from remarc.core.hyperparameters import Presets as P

EPOCHS = 5 


def time_rl(N, num_drugs):
    p = P(
        state_shape=(2**N,),
        num_actions=num_drugs,
        lr=1e-4,
        epochs=EPOCHS,
        train_steps_per_epoch=2000,
        test_episodes=5,
        batch_size=64,
        buffer_size=20000,
        dataset="synthetic",
        gen_per_step=10,
    )
    
    object.__setattr__(p, "num_drugs", num_drugs)

    start = time.perf_counter()
    train_wf_landscapes(p, signature=f"runtime_N{N}_D{num_drugs}")
    return time.perf_counter() - start