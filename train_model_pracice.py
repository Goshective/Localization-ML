from huggingface_hub import notebook_login, login
from datasets import load_dataset
from transformers import (AutoTokenizer, 
                          DataCollatorForSeq2Seq, 
                          AutoModelForSeq2SeqLM, 
                          Seq2SeqTrainingArguments, 
                          Seq2SeqTrainer)
import evaluate
import numpy as np


"""Train and Set up"""

login()


# books = load_dataset("opus_books", "en-ru")

dataset = load_dataset("issai/kazparc")
dataset.column_names


def change_id(example): # ID tokens for precise training
    id_files = {"MM": '1', "GN": '2', "ES": '3', "LD": '4', "FI": '5'}
    prefix = example['id'][:2]
    example['id'] = int(id_files[prefix] + example['id'][2:-2])
    return example

# dataset = dataset.map(change_id)

dataset = dataset["train"].train_test_split(test_size=0.2)

print(dataset["train"][0])


checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "ru"
target_lang = "en"
prefix = "translate Russian to English: "

def preprocess_function_kazparc(examples):
    inputs = [prefix + example for example in examples[source_lang]]
    targets = [example for example in examples[target_lang]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

def preprocess_function_opus(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs


tokenized_books = dataset.map(preprocess_function_kazparc, batched=True)


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)


metric = evaluate.load("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


training_args = Seq2SeqTrainingArguments(
    output_dir="opus_books_model_1",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True, #change to bf16=True for XPU
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub("opus_books_model_1")