import numpy as np
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, Trainer
from transformers import BeitForImageClassification
import torch
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)
from transformers import ViTImageProcessor, BeitImageProcessor
from datasets import load_dataset

train_ds, test_ds = load_dataset(
    'cifar10', split=['train[:5000]', 'test[:2000]'])
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

print(train_ds)
print(train_ds.features)


id2label = {id: label for id, label in enumerate(
    train_ds.features['label'].names)}
label2id = {label: id for id, label in id2label.items()}
print(id2label)


processor = BeitImageProcessor.from_pretrained(
    'microsoft/beit-base-patch16-224')


image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
transform_train = Compose(
    [
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

transform_val = Compose(
    [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ]
)


def train_transforms(examples):
    examples['pixel_values'] = [transform_train(
        image.convert("RGB")) for image in examples['img']]
    return examples


def val_transforms(examples):
    examples['pixel_values'] = [transform_val(
        image.convert("RGB")) for image in examples['img']]
    return examples


train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"]
                               for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224',
                                                   id2label=id2label,
                                                   label2id=label2id,
                                                   ignore_mismatched_sizes=True)


metric_name = "accuracy"

args = TrainingArguments(
    f"checkpoints",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
    max_grad_norm=0.0,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))


trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)


trainer.train()
