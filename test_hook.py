import torch 
import torch.nn as nn

class LipsNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.register_full_backward_hook(backward_hook)
        self.reg_loss = 0
        self.opt = torch.optim.Adam()

        self.num_forward = 0

    def forward(self, x):
        if self.train:
            self.reg_loss += K**2
            self.num_forward += 1
        return x


def backward_hook(module, grad_input, grad_output):
    module.num_forward -= 1
    if module.num_forward == 1:
        module.reg_loss.backward()
        module.opt.step()
        module.opt.zero_grad()
    return grad_input


if __name__ == "__main__":
    model = LipsNet()
    input = torch.ones(5, requires_grad=True)
    loss = model(input).sum()
    loss.backward() # [command 1] not running, nothing output


