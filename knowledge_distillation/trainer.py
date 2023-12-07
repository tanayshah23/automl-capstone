from huggingface_hub import HfFolder
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

from knowledge_distillation.utils import initialize_and_compare_tokenizers, compute_metrics, \
    prepare_and_tokenize_dataset

from huggingface_hub import HfApi


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


def create_trainer(student_model, args, train_dataset, eval_dataset, tokenizer, teacher_model, compute_metrics):
    trainer = DistillationTrainer(
        model=student_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        compute_metrics=compute_metrics,
    )
    return trainer


if __name__ == '__main__':
    teacher_id = "nfliu/roberta-large_boolq"
    student_id = "distilroberta-base"

    # name for our repository on the hub
    repo_name = "roberta-boolq-distilled"

    teacher_tokenizer, student_tokenizer = initialize_and_compare_tokenizers(teacher_id, student_id)

    training_args = DistillationTrainingArguments(
        output_dir=repo_name,
        num_train_epochs=7,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=True,
        learning_rate=6e-5,
        seed=33,
        # logging & evaluation strategies
        logging_dir=f"{repo_name}/logs",
        logging_strategy="epoch",  # to get more information to TB
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard",
        # push to hub parameters
        push_to_hub=True,
        hub_strategy="every_save",
        hub_model_id=repo_name,
        hub_token=HfFolder.get_token(),
        # distilation parameters
        alpha=0.5,
        temperature=4.0
    )

    tokenized_dataset = prepare_and_tokenize_dataset("boolq")

    trainer = create_trainer(student_id, training_args, tokenized_dataset['train'], tokenized_dataset['validation'],
                             teacher_tokenizer, teacher_id,
                             compute_metrics)

    trainer.train()

    whoami = HfApi().whoami()
    username = whoami['name']

    print(f"https://huggingface.co/{username}/{repo_name}")

    trainer.create_model_card(model_name=training_args.hub_model_id)
    trainer.push_to_hub()
