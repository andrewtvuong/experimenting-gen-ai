from transformers import BertModel, BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
from lora import LoRA
import torch.nn as nn

# Custom model incorporating BERT with a classification head
class BertWithLoRA(nn.Module):
    def __init__(self, bert_model, rank):
        super(BertWithLoRA, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # Assuming binary classification
        
        # Applying LoRA to the query matrix of the first attention layer
        input_size = self.bert.config.hidden_size
        output_size = self.bert.config.hidden_size
        self.lora_adaptation = LoRA(self.bert.encoder.layer[0].attention.self.query, rank, input_size, output_size)
        self.bert.encoder.layer[0].attention.self.query = self.lora_adaptation

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return loss, logits

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        loss, logits = model(**batch, labels=labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_base = BertModel.from_pretrained('bert-base-uncased')
    model = BertWithLoRA(bert_base, rank=32).to(device)

    dataset = load_dataset('imdb', split='train[:10%]')
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = DataLoader(dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader))

    train_loss = train(model, dataloader, optimizer, scheduler, device)
    print(f"Training loss: {train_loss}")

if __name__ == "__main__":
    main()
