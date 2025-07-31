import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDelta

pyro.set_rng_seed(0)

# 参数设定
T = 6               # 时间长度
L = 4               # z 的长度
N = 5               # 每个 z[l] 的取值空间（0 ~ N-1）
num_eta = L + 1     # η_t ∈ {0, 1, ..., L}

# 先验定义
p_eta0 = torch.ones(num_eta) / num_eta
transition_probs = torch.ones(num_eta, num_eta) / num_eta  # η_t | η_{t−1}
# s_t | η_t=0 ~ Uniform(0, N-1)
uniform_s_dist = dist.Categorical(torch.ones(N) / N)

# 观测值（整数）0 ~ N-1
s_obs = torch.tensor([1, 3, 0, 2, 2, 4])  # shape: (T,)

def model(s=None):
    # z: 每个 z[l] ∈ {0, ..., N-1}
    z = []
    for l in range(L):
        z_l = pyro.sample(f"z_{l}", dist.Categorical(torch.ones(N) / N))
        z.append(z_l)
    z = torch.stack(z)

    eta_prev = pyro.sample("eta_0", dist.Categorical(p_eta0))
    for t in range(T):
        eta_t = pyro.sample(f"eta_{t}", dist.Categorical(transition_probs[eta_prev]),
                            infer={"enumerate": "parallel"})

        if s is not None:
            if isinstance(eta_t, torch.Tensor) and eta_t.dim() == 0:
                eta_val = eta_t.item()
                if eta_val == 0:
                    pyro.sample(f"s_{t}", uniform_s_dist, obs=s[t])
                else:
                    pyro.sample(f"s_{t}", dist.Delta(z[eta_val - 1]), obs=s[t])
            else:
                # 枚举版本：用 mixture mask over eta
                probs = []
                for eta_val in range(num_eta):
                    if eta_val == 0:
                        probs.append(uniform_s_dist.log_prob(s[t]))
                    else:
                        probs.append(dist.Delta(z[eta_val - 1]).log_prob(s[t]))
                logp = torch.stack(probs)[eta_t]
                pyro.factor(f"logp_s_{t}", logp)
        eta_prev = eta_t

# guide: MAP 对 z 进行估计
guide = AutoDelta(poutine.block(model, expose=[f"z_{l}" for l in range(L)]))

# 推断器配置
optimizer = Adam({"lr": 0.05})
elbo = TraceEnum_ELBO(max_plate_nesting=1)
svi = SVI(model, guide, optimizer, elbo)

# 执行推断
for step in range(500):
    loss = svi.step(s_obs)
    if step % 100 == 0:
        print(f"[step {step}] ELBO: {loss:.2f}")

# 后验估计结果
z_map = torch.stack([guide()[f"z_{l}"] for l in range(L)])
print("MAP estimate of z:", z_map)
