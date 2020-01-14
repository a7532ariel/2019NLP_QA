import torch
import numpy as np
from transformers import glue_processors as processors
from pytorch_pretrained_bert import BertTokenizer, BertModel
task = "mrpc"
data_dir = "./MRPC"
processor = processors[task]()
examples = processor.get_train_examples(data_dir)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
pretrain_model = BertModel.from_pretrained("bert-base-cased")
pretrain_model.eval()
encoded_data = [[] for _ in range(13)]
#data_len = []
#count = 0
for i, example in enumerate(examples):
    
    ### Preprocess input (tokenized)
    concat_text = example.text_a + ' ' + example.text_b
    tokenized_text = tokenizer.tokenize(concat_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    
    ### layer 0
    embedding_output = pretrain_model.embeddings(tokens_tensor)
    embedding_output = embedding_output.detach().numpy()
    embedding_output = np.expand_dims(embedding_output, axis=0)
    # print(embedding_output.shape)
    
    ### layer 1~12
    encoded_layers, _ = pretrain_model(tokens_tensor)	
    encoded_layers = torch.stack(encoded_layers)
    encoded_layers = encoded_layers.detach().numpy()
    # print(encoded_layers.shape)
    
    ### Stack together 0~12
    total_layers = np.vstack((embedding_output, encoded_layers))
    # print(total_layers.shape) 
    # data_len.append(encoded_layers.shape[2])
    # count += encoded_layers.shape[2]
    
    ### Append layer 0~12
    for j, layer in enumerate(total_layers):
        for data in layer[0]:
            encoded_data[j].append(data)
    print("iteration %d" % (i))

encoded_data = np.asarray(encoded_data)
print(encoded_data.shape)
np.save("./total_train.npy", encoded_data)
	

"""
example_text = examples[0].text_a + ' ' + examples[0].text_b
print(example_text)
tokenized_text = tokenizer.tokenize(example_text)
print(tokenized_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(indexed_tokens)
tokens_tensor = torch.tensor([indexed_tokens])

encoded_layers, _ = pretrain_model(tokens_tensor)
print(encoded_layers)
print(encoded_layers[0])
"""
