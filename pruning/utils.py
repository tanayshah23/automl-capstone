import datasets
import torch
import transformers
from datasets import load_dataset
from nn_pruning.patch_coordinator import ModelPatchingCoordinator
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from transformers import TrainingArguments

import numpy as np
from datasets import load_metric

bert_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_ckpt)
datasets.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
print(
    f"Using transformers v{transformers.__version__} and datasets v{datasets.__version__} and torch v{torch.__version__}")


def process(examples):
    # Tokenize 'question' and 'passage'
    tokenized_inputs = tokenizer(examples['question'], examples['passage'], truncation="only_second")

    return tokenized_inputs


def prepare_and_tokenize_dataset(dataset_name):
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Apply the 'process' function to tokenize and process the dataset
    tokenized_datasets = dataset.map(process, batched=True)

    return tokenized_datasets


def compute_metrics(pred):
    accuracy_score = load_metric('accuracy')
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score.compute(predictions=predictions, references=labels)


def patch_and_save_model(model_checkpoint, sparse_args, save_dir="models/patched"):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the ModelPatchingCoordinator
    mpc = ModelPatchingCoordinator(
        sparse_args=sparse_args,
        device=device,
        cache_dir="checkpoints",
        logit_names="logits",
        teacher_constructor=None)

    # Load the model and move it to the device
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(device)

    # Patch the model
    mpc.patch_model(model)

    # Save the patched model
    model.save_pretrained(save_dir)

    return mpc, model


def create_training_arguments(config, dataset_length):
    # Extract parameters from the config
    batch_size = config.get('batch_size', 16)
    learning_rate = config.get('learning_rate', 2e-5)
    num_train_epochs = config.get('num_train_epochs', 6)

    # Calculate logging steps based on the dataset length
    logging_steps = dataset_length // batch_size

    # Calculate warmup steps (10% of total training steps)
    warmup_steps = int(logging_steps * num_train_epochs * 0.1)

    # Create TrainingArguments
    args = TrainingArguments(
        output_dir=config.get('output_dir', "checkpoints"),
        evaluation_strategy=config.get('evaluation_strategy', "epoch"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=config.get('weight_decay', 0.01),
        logging_steps=logging_steps,
        save_strategy=config.get('save_strategy', "epoch"),
        disable_tqdm=config.get('disable_tqdm', False),
        load_best_model_at_end=config.get('load_best_model_at_end', True),
        metric_for_best_model=config.get('metric_for_best_model', "accuracy"),
        report_to=config.get('report_to', None),
        warmup_steps=warmup_steps
    )

    return args
