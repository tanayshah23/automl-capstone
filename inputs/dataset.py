import json
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

def get_train_test(data_dir):
    reviews = json.load(open(data_dir))
    low_resource_domains = ["office_products", "automotive", "computer_&_video_games"]
    train_examples = [r for r in reviews if r['domain'] not in low_resource_domains]
    test_examples = [r for r in reviews if r['domain'] in low_resource_domains]
    return train_examples, test_examples

def create_feature_set(examples, LABEL_MAP, tokenizer, max_seq_length):
    all_input_ids      = torch.empty(len(examples), max_seq_length, dtype = torch.long)
    all_attention_mask = torch.empty(len(examples), max_seq_length, dtype = torch.long)
    all_segment_ids    = torch.empty(len(examples), max_seq_length, dtype = torch.long)
    all_label_ids      = torch.empty(len(examples), dtype = torch.long)

    for id_,example in enumerate(examples):
        tokens = tokenizer(example['text'], add_special_tokens=True, pad_to_max_length=True, max_length=128, truncation=True)
        input_ids = tokens["input_ids"]

        label_id = LABEL_MAP[example['label']]
        all_input_ids[id_] = torch.Tensor(input_ids).to(torch.long)
        all_label_ids[id_] = torch.Tensor([label_id]).to(torch.long)

    tensor_set = TensorDataset(all_input_ids, all_label_ids)
    return tensor_set

def get_feature_set(examples):
    LABEL_MAP  = {'positive':0, 'negative':1, 0:'positive', 1:'negative'}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    return create_feature_set(examples, LABEL_MAP, tokenizer, 128)