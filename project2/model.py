import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, 
                            hidden_size, 
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        #dim=1 or dim=2?
        #self.sm = nn.Softmax(dim=2)
        self.init_weights()
        
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        
    def forward(self, features, captions):
        # removing a sample in dim 1 of embeddings to get correct size
        embeddings = self.embedding(captions[:, :-1])
        # print(embeddings.shape)
        # print(features.unsqueeze(1).shape)
        # unsqueezing the features on dim=1 to give correct dims for concatenation. 
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        #print(inputs.shape)
        out, _ = self.lstm(inputs)
        out = self.fc(out)
        #out = self.sm(out)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        capt = []        
        for _ in range(max_len):
            out, states = self.lstm(inputs, states)
            out = self.fc(out)
            wrd = out.max(2)[1].item()
            capt.append(wrd)
            # update inputs
            inputs = self.embedding(out.max(2)[1])
            if wrd == 1:
                break
        return capt