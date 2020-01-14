from model import MLP
from model import LinearProject
from torch.utils.data.dataset import Dataset
import argparse
import torch
import numpy as np
neighbor = False
class customDataset(Dataset):
    
    def __init__(self, layer=1, data_path="total_dev.npy", len_path="len_dev.npy"):
        print("Loading from: " + data_path)
        print("Loading from: " + len_path)
        self.layer = layer
        self.data = np.load(data_path)
        self.len_data = np.load(len_path)
        self.pos = 0
        print(self.data.shape) 
 
    def __getitem__(self, index):
        ### data, label
        return self.data[self.layer][index], self.data[self.layer-1][index]
 
    def __len__(self):
        return self.data[self.layer].shape[0]

    def get_test_data(self, index):
        data, label = self.data[self.layer][self.pos:self.pos+index], self.data[self.layer-1][self.pos:self.pos+index]
        self.pos += index
        return data, label
    
    def get_all_data(self, index):
        data = self.data[:,self.pos:self.pos+index]
        assert(data.shape[0] == 13)
        self.pos += index
        return data

def get_nearest_neighbor(list1, list2):
    output = []
    for word in list1:
        min_dis = None
        index = None
        for i, label_word in enumerate(list2):
            dis = np.mean((word-label_word) ** 2)    
            if((min_dis == None) or min_dis > dis):
                   min_dis = dis
                   index = i
        output.append(label[index])

    return np.array(output)

parser = argparse.ArgumentParser()
parser.add_argument("--state_dict", default = "./models/linear_mse")
parser.add_argument("--layer", type = int, required=True)
#parser.add_argument("--model_layer", type = int, required=True)
parser.add_argument("--model", type = str, default = "linear")
parser.add_argument("--loss", type = str, default = "mse")
parser.add_argument("--number", type = int)
args = parser.parse_args()

models = {"mlp": MLP(), "linear": LinearProject()}
dataset = customDataset(args.layer)
#model = models[args.model.lower()].cuda()
all_model = {}
### WARNING need to use for loop for model init, if you use [MLP()] * number, only ONE model will be constructed!!!
for i in range(1,args.number+1):
    all_model[i] = LinearProject().cuda()
 
### load model
for i in range(args.number,0,-1):
    print("Using model: %s" % (args.state_dict + "_" + str(i) + ".pkl"))
    all_model[i].load_state_dict(torch.load(args.state_dict + "_" + str(i) + ".pkl"))

#print("Using model: %s" % (args.state_dict + "_" + str(args.model_layer) + ".pkl"))
#state_dict = torch.load(args.state_dict + "_" + str(args.model_layer) + ".pkl")
#model.load_state_dict(state_dict)
#model = model.cuda()

#print("Start Testing, Layer: %s, Model: %s" % (args.layer, args.model.lower()))
iters  = 0
n_data = 0
total_correct = 0
cand = None
for length in dataset.len_data:
   iters += 1
   #data, label = dataset.get_test_data(length)
   data = dataset.get_all_data(length)
   data = torch.tensor(data).cuda() # keep
   n_data += length
   for i in range(args.number,0, -1):
       if (i == args.number):
           output = all_model[i](data[i])  
           #print(output.shape, data[i-1].shape)
           if(neighbor):
               output = output.cpu().detach().numpy()
               label  = data[i-1].cpu().detach().numpy()
               cand = get_nearest_neighbor(output, label)   
           else:
               cand = output
           cand = torch.tensor(cand).cuda()
       else:
           cand = all_model[i](cand)
           if(neighbor):
               label  = data[i-1].cpu().detach().numpy()
               cand = cand.cpu().detach().numpy()
               cand = get_nearest_neighbor(cand, label)
               cand = torch.tensor(cand).cuda()
   label = data[0]
   #output = model(data)
   #output = output.cpu().detach().numpy()
   #cand = get_nearest_neighbor(output, label) 
   ### calculate acc
   label = data[0].cpu().detach().numpy()
   cand = cand.cpu().detach().numpy()
   if(not neighbor):
       cand = get_nearest_neighbor(cand, label)     
   for i in range(len(label)):
       if((cand[i] == label[i]).all()):
         total_correct += 1
   
   print("iter: %4f/%4f, accuracy: %4f" % (iters, len(dataset.len_data), total_correct/n_data), end = '\r' )

assert(n_data == dataset.data.shape[1])
print(total_correct/n_data, file=open("linear_all_no_neighbor.log", "a"))
