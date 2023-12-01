import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from sklearn.metrics import accuracy_score

class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(SentimentAnalysisModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output = self.layers(embedded.mean(dim=1))
        return output
    
def get_initialized_model():
    model = SentimentAnalysisModel(30522, 768, 1)
    # Load the weights here
    return model

def get_optimizer(model_params, lr):
    return Adam(model_params, lr=lr)

def get_outputs(model, batch, device):
    batch = tuple(t.to(device) for t in batch)
    input_ids, attention_mask, segment_ids, label_id = batch
    outputs = model(input_ids, attention_mask, segment_ids, labels = label_id)
    return outputs

def get_loss(outputs):
    return outputs[0]

def get_label_from_batch(batch, device):
    batch = tuple(t.to(device) for t in batch)
    _, _, _, label_id = batch
    return label_id

def get_labels(outputs):
    q_logits = F.softmax(outputs[1],dim=1)
    pre_label_id = torch.argmax(q_logits,dim=1)
    return pre_label_id

def get_metric_score(label, preds):
    return accuracy_score(label, preds)