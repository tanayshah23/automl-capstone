from nn_pruning.patch_coordinator import SparseTrainingArguments
from transformers import Trainer, AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification
from nn_pruning.sparse_trainer import SparseTrainer

from pruning.utils import create_training_arguments, prepare_and_tokenize_dataset, patch_and_save_model, compute_metrics


class PruningTrainer(SparseTrainer, Trainer):
    def __init__(self, sparse_args, *args, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        SparseTrainer.__init__(self, sparse_args)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        We override the default loss in SparseTrainer because it throws an
        error when run without distillation
        """
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        self.metrics["ce_loss"] += float(loss)
        self.loss_counter += 1
        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':
    bert_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_ckpt)

    sparse_args = SparseTrainingArguments()

    example_config = {
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_train_epochs': 6,
        'output_dir': "example_checkpoints",
        'evaluation_strategy': "epoch",
        'weight_decay': 0.01,
        'save_strategy': "epoch",
        'disable_tqdm': False,
        'load_best_model_at_end': True,
        'metric_for_best_model': "accuracy",
        'report_to': None
    }

    boolq_enc = prepare_and_tokenize_dataset("boolq")

    args = create_training_arguments(example_config, len(boolq_enc["train"]))

    mpc, model = patch_and_save_model(bert_ckpt, sparse_args)

    trainer = PruningTrainer(
        sparse_args=sparse_args,
        args=args,
        model=model,
        train_dataset=boolq_enc["train"],
        eval_dataset=boolq_enc["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.set_patch_coordinator(mpc)

    trainer.train()

    output_model_path = "models/bert-base-uncased-finepruned-boolq"
    trainer.save_model(output_model_path)
