import torch 
import torch.nn as nn 

from model import Generator, Descriminator


device = torch.device("cuda")

random_style = torch.randn(1, 1, 128)
random_style = random_style.to(device)
'''
print("Loading generator...")
g = Generator()
g.to(device)
'''
print("Loading Descriminator")
d = Descriminator()
d.to(device)

'''
print("generator forward pass...")
out, latents = g(random_style)
'''
out = torch.randn(1, 256, 2**16)
out = out.to(device)
print("d forward pass.....")
label = d(out)
print(label)

