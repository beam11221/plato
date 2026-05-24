# GPU-Tier-Aware Sleep Times for FedAsync

Simulates hardware heterogeneity across 100 Jetson clients (Xavier / Orin / Thor)
by computing each client's training wall-time from its GPU TFLOPS rating.

## File map

| File | Role |
|------|------|
| `gpu_tier_sleep.py` | Formula module — `TierSpec`, `ClientSpec`, `sleep_time_seconds()` |
| `fedasync_gpu_tier.py` | Entry point — injects sleep times, adds per-round log callback |
| `fedasync_CIFAR10_resnet18_100client_gpu_tier.toml` | Run config (ResNet-18, 50 rounds, Dirichlet α=0.5) |
| `tiers.csv` | Hardware specs (do not modify schema) |
| `clients.csv` | Client → tier mapping (do not modify schema) |

PLATO core files are **not modified**.

## How it works

```
fedasync_gpu_tier.py (main)
  ├── gpu_tier_sleep.get_computer()          # loads tiers.csv + clients.csv once
  ├── compute sleep_time_seconds(cid, …)     # per-call formula, no caching
  ├── Config.client_sleep_times = np.array(…) # inject into PLATO's existing path
  └── simple.Client(trainer_callbacks=[GpuTierLogCallback])
        └── on every training epoch:
              ComposableTrainer.simulate_sleep_time()   # reads Config.client_sleep_times
                └── time.sleep(sleep_seconds)           # real wall-clock delay
              GpuTierLogCallback.on_train_epoch_end()   # logs tier/size/time
```

PLATO's `speed_simulation = true` + `sleep_simulation = false` path performs the
actual `time.sleep()` so that `simulate_wall_time = true` on the server captures
realistic staleness.

## Formula

```
work_flops = W_MODEL_FLOPS × dataset_size × epochs
sleep_time = work_flops / (gpu_tflops_fp16 × 1e12 × eta_nominal)
if noise_sigma > 0:
    sleep_time *= (1 + N(0, noise_sigma))
```

`W_MODEL_FLOPS = 1.1e9` is defined as a named constant at the top of
`gpu_tier_sleep.py`.

### Current values (partition_size=500, epochs=1)

| Tier   | TFLOPS | sleep_time (s) |
|--------|--------|----------------|
| Xavier | 10     | 0.0550         |
| Orin   | 78     | 0.0071         |
| Thor   | 150    | 0.0037         |

Ratios: 15 : 1.9 : 1 (Xavier : Orin : Thor), matching the TFLOPS inverse ratio.

## How to change eta_nominal or noise_sigma

Edit **tiers.csv only** — no code changes needed:

```
tier_id,…,eta_nominal,noise_sigma,…
xavier,…,0.8,0.0,…    ← Xavier is 80 % efficient → longer sleep
orin,…,1.0,0.1,…      ← Orin gets 10 % multiplicative noise per round
thor,…,1.0,0.0,…
```

`eta_nominal < 1` scales the effective TFLOPS down (longer sleep).  
`noise_sigma > 0` injects `N(0, sigma)` multiplicative noise each call.

## Enabling per-round noise (future)

When `noise_sigma > 0`, each call to `sleep_time_seconds()` returns a different
value. The current wiring precomputes all sleep times once at startup:

```python
# fedasync_gpu_tier.py
Config.client_sleep_times = np.array([computer.sleep_time_seconds(...) ...])
```

This is exact when noise_sigma = 0. To get fresh noise each round, switch to a
trainer subclass that calls `gpu_tier_sleep.sleep_time_seconds()` inside
`simulate_sleep_time()` — one override, no PLATO core changes:

```python
# fedasync_trainer.py  (create when needed)
from plato.trainers.basic import Trainer
import gpu_tier_sleep, time
from plato.config import Config

class GpuTierTrainer(Trainer):
    def simulate_sleep_time(self):
        if not (hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation):
            return
        dataset_size = ...  # from self.sampler / self.trainset
        sleep_seconds = gpu_tier_sleep.sleep_time_seconds(
            self.client_id, dataset_size, epochs=1
        )
        if sleep_seconds > 0 and not getattr(Config().clients, "sleep_simulation", False):
            time.sleep(sleep_seconds)
```

Then pass `trainer=GpuTierTrainer` to `simple.Client(...)`.

## Run

```bash
cd /workspace/plato/examples/async/fedasync
python fedasync_gpu_tier.py -c fedasync_CIFAR10_resnet18_100client_gpu_tier.toml
```

## Verification

Grep the run log for `[GpuTierSleep]`:

```
[GpuTierSleep] round=3 client_id=12 tier_id=xavier dataset_size=500 sleep_time=0.055000 s
[GpuTierSleep] round=3 client_id=63 tier_id=orin   dataset_size=500 sleep_time=0.007051 s
[GpuTierSleep] round=3 client_id=91 tier_id=thor   dataset_size=500 sleep_time=0.003667 s
```

- Thor clients have the shortest sleep, Xavier the longest.
- Ratios ≈ 1 : 7.8 : 15 (matching TFLOPS ratios).
- The slowest-selected client per round dominates wall-clock (FedAsync per-update path).
