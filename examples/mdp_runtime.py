import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from remarc.agents.shepherd_eval import ShepherdMDP
from remarc.core.landscapes import Landscape
from remarc.envs.wright_fisher_env import WrightFisherEnv

L = 3   


def time_mdp(N, num_drugs):
    env = WrightFisherEnv(
        seq_length=N,
        gen_per_step=10,
        landscape_list=[Landscape(N, sigma=0.5) for _ in range(num_drugs)],
        num_drugs=num_drugs,
    )

    start = time.perf_counter()
    ShepherdMDP.from_env(env, L=L).solve()
    return time.perf_counter() - start