from transformers import AutoTokenizer
from datasets import load_dataset

from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from huggingface_hub import HfFolder

from sklearn.metrics import accuracy_score
import torch

from knowledge_distillation.trainer import DistillationTrainingArguments

from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer


def initialize_and_compare_tokenizers(teacher_id, student_id):
    # Initialize teacher tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_id)

    # Initialize student tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(student_id)

    # Sample input for testing
    sample = "This is a basic example, with different words to test."

    # Asserting if both tokenizers produce the same output
    assert teacher_tokenizer(sample) == student_tokenizer(sample), "Need similar family of tokenizers"

    return teacher_tokenizer, student_tokenizer


tokenizer = AutoTokenizer.from_pretrained("some-model-id")


# update the process function to tokenize the dataset
def process(examples):
    # Tokenize 'question' and 'passage'
    tokenized_inputs = tokenizer(examples['question'], examples['passage'], truncation=True, padding='max_length')

    # Convert boolean 'answer' to integer
    tokenized_inputs['answer'] = [1 if answer else 0 for answer in examples['answer']]

    return tokenized_inputs


def prepare_and_tokenize_dataset(dataset_name):
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Apply the 'process' function to tokenize and process the dataset
    tokenized_datasets = dataset.map(process, batched=True)

    # Cast the 'answer' column to ClassLabel type
    class_label_feature = ClassLabel(num_classes=2, names=['False', 'True'])
    tokenized_datasets = tokenized_datasets.cast_column('answer', class_label_feature)

    # Rename 'answer' column to 'labels'
    tokenized_datasets = tokenized_datasets.rename_column("answer", "labels")

    return tokenized_datasets


# Example usage
# tokenized_dataset = prepare_and_tokenize_dataset("boolq")


def initialize_training_components(tokenizer, teacher_id, student_id, repo_name, tokenized_datasets, config):
    # Create label2id, id2label dicts
    labels = tokenized_datasets["train"].features["labels"].names
    num_labels = len(labels)
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Define training arguments from config
    training_args = DistillationTrainingArguments(
        output_dir=config.get("output_dir", repo_name),
        num_train_epochs=config.get("num_train_epochs", 7),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 16),
        fp16=config.get("fp16", True),
        learning_rate=config.get("learning_rate", 6e-5),
        seed=config.get("seed", 33),
        logging_dir=config.get("logging_dir", f"{repo_name}/logs"),
        logging_strategy=config.get("logging_strategy", "epoch"),
        evaluation_strategy=config.get("evaluation_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        save_total_limit=config.get("save_total_limit", 2),
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "accuracy"),
        report_to=config.get("report_to", "tensorboard"),
        push_to_hub=config.get("push_to_hub", True),
        hub_strategy=config.get("hub_strategy", "every_save"),
        hub_model_id=config.get("hub_model_id", repo_name),
        hub_token=config.get("hub_token", HfFolder.get_token()),
        alpha=config.get("alpha", 0.5),
        temperature=config.get("temperature", 4.0)
    )

    # Define data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define teacher model
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        teacher_id, num_labels=num_labels, id2label=id2label, label2id=label2id)

    # Define student model
    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_id, num_labels=num_labels, id2label=id2label, label2id=label2id)

    return teacher_model, student_model, data_collator, training_args


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.softmax(torch.tensor(logits), dim=-1).argmax(dim=-1)
    return {"accuracy": accuracy_score(labels, predictions.numpy())}
