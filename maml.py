import gc
import torch
import numpy as np
from torch import nn
from copy import deepcopy
from torch.utils.data import DataLoader, RandomSampler
from inputs.model import *

class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Learner, self).__init__()
        
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr  = args.outer_update_lr
        self.inner_update_lr  = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = get_initialized_model()
        self.outer_optimizer = get_optimizer(self.model.parameters(), self.outer_update_lr)
        self.model.train()

    def forward(self, batch_tasks, training = True):
        """
        batch = [(support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset)]
        
        # support = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
        """
        task_accs = []
        sum_gradients = []
        num_task = len(batch_tasks)
        num_inner_update_step = self.inner_update_step if training else self.inner_update_step_eval

        for task_id, task in enumerate(batch_tasks):
            support = task[0]
            query   = task[1]
            
            fast_model = deepcopy(self.model)
            fast_model.to(self.device)
            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=self.inner_batch_size)
            
            inner_optimizer = get_optimizer(fast_model.parameters(), lr=self.inner_update_lr)
            fast_model.train()
            
            print('----Task',task_id, '----')
            for i in range(0,num_inner_update_step):
                all_loss = []
                for inner_step, batch in enumerate(support_dataloader):
                    outputs = get_outputs(fast_model, batch, self.device)
                    loss = get_loss(outputs)
                    loss.backward()
                    inner_optimizer.step()
                    inner_optimizer.zero_grad()
                    
                    all_loss.append(loss.item())
                
                if i % 4 == 0:
                    print("Inner Loss: ", np.mean(all_loss))

            fast_model.to(torch.device('cpu'))
            
            if training:
                meta_weights = list(self.model.parameters())
                fast_weights = list(fast_model.parameters())

                for i, (meta_params, fast_params) in enumerate(zip(meta_weights, fast_weights)):
                    gradient = meta_params - fast_params
                    if task_id == 0:
                        sum_gradients.append(gradient)
                    else:
                        sum_gradients[i] += gradient

            fast_model.to(self.device)
            fast_model.eval()

            with torch.no_grad():
                query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
                query_batch = iter(query_dataloader).next()
                q_outputs = get_outputs(fast_model, query_batch, self.device)
                q_label_id = get_label_from_batch(query_batch, self.device)
                pre_label_id = get_labels(q_outputs)
                pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
                q_label_id = q_label_id.detach().cpu().numpy().tolist()
                acc = get_metric_score(q_label_id, pre_label_id)
                task_accs.append(acc)
            
            del fast_model, inner_optimizer
            torch.cuda.empty_cache()
        
        if training:
            # Average gradient across tasks
            for i in range(0,len(sum_gradients)):
                sum_gradients[i] = sum_gradients[i] / float(num_task)

            #Assign gradient for original model, then using optimizer to update its weights
            for i, params in enumerate(self.model.parameters()):
                params.grad = sum_gradients[i]

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            
            del sum_gradients
            gc.collect()
        
        return np.mean(task_accs)