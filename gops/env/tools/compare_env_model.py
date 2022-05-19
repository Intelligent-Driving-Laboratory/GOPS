import torch
import numpy as np
import matplotlib.pyplot as plt


def compare(env, model, *, name=""):
    s = env.reset()

    state_buffer = []
    state_buffer2 = []

    s_torch = torch.as_tensor(s, dtype=torch.float32).reshape(1, -1)

    state_buffer.append(s)
    state_buffer2.append(s_torch)

    r_buffer = []
    r_buffer2 = []

    for i in range(100):
        a = env.action_space.sample()

        s_torch, r, d, info = model.forward(
            s_torch, torch.as_tensor(a, dtype=torch.float32).reshape(1, -1), torch.as_tensor([False], dtype=torch.bool)
        )

        state_buffer2.append(s_torch)
        r_buffer2.append(r.item())

        s, r, d, _ = env.step(a)

        r_buffer.append(r)
        state_buffer.append(s)


    state_stack = np.stack(state_buffer)
    state_stack2 = torch.cat(state_buffer2).detach().numpy()

    r_stack = np.stack(r_buffer)
    r_stack2 = np.stack(r_buffer2)

    plt.figure()
    plt.plot(state_stack)
    plt.plot(state_stack2, "--")
    plt.title("state")
    plt.show()

    plt.figure()

    plt.plot(r_stack)
    plt.plot(r_stack2, "--")
    plt.title("reward")
    plt.show()
