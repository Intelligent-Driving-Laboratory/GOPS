import torch 
import torch.nn as nn

class LipsNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, x):
        return x
    
    def backward(self, ):
        print("module's backward call") # [command 1]
        return super().backward()

    def parameter_group():
        return {k:para1, k:para2}
        return {para2, para1}
        # return {para1:lr1, para2:lr2}


if __name__ == "__main__":
    model = LipsNet()
    input = torch.ones(5, requires_grad=True)
    loss = model(input).sum()
    loss.backward() # [command 1] not running, nothing output


