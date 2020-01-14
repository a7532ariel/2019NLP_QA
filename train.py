from model import MLP
from model import LinearProject
from torch.utils.data.dataset import Dataset
import argparse
import torch
import numpy as np

class customDataset(Dataset):
    
    def __init__(self, layer=1, data_path="total_train.npy"):
        print("Loading from: " + data_path)
        self.layer = layer
        self.data = np.load(data_path)
        print(self.data.shape) 
 
    def __getitem__(self, index):
        ### data, label
        return self.data[self.layer][index], self.data[self.layer-1][index]
 
    def __len__(self):
        return self.data[self.layer].shape[0]

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", default = "./models/")
parser.add_argument("--layer", type = int, required=True)
parser.add_argument("--model", type = str, default = "linear")
parser.add_argument("--loss", type = str, default = "cosine")
parser.add_argument("--epoch", type = int, default = 8)
parser.add_argument("--lr", type = float, default = 0.0005)

args = parser.parse_args()
models = {"mlp": MLP(), "linear": LinearProject()}
losses = {"mse": torch.nn.MSELoss(), "cosine": torch.nn.CosineEmbeddingLoss()}
dataset = customDataset(args.layer)
model = models[args.model.lower()].cuda()
loss_func  = losses[args.loss.lower()]
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr) 
total_iter = len(dataloader)
print("Start Training, Layer: %s, Model: %s, Loss: %s" % (args.layer, args.model.lower(), args.loss.lower()))
for e in range(args.epoch):
    print("Epoch: ",e)
    iters  = 0
    n_data = 0
    total_loss = 0.0

    for data, label in dataloader:
        iters += 1
        optimizer.zero_grad()
        data = data.cuda()
        label = label.cuda()
        output = model(data)
        if(args.loss == "cosine"):
            loss = loss_func(output, label, torch.ones(1).cuda())
        elif(args.loss == "mse"):
            loss = loss_func(output, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_data += data.shape[0]
        print("iter: %4f/%4f, avg_loss: %4f, Loss: %4f" % (iters, total_iter, total_loss/n_data,  loss.item()), end = '\r' )

torch.save(model.state_dict(), args.output_path + str(args.model.lower())+ "_" + str(args.loss.lower()) + "_" + str(args.layer) + ".pkl")
        
