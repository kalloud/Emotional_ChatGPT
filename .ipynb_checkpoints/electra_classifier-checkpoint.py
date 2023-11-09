from typing import Optional
import os

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchmetrics
from transformers import ElectraModel, ElectraPreTrainedModel, ElectraTokenizerFast as ElectraTokenizer, AdamW
from transformers.models.electra.modeling_electra import ElectraClassificationHead

from transformers import ElectraTokenizerFast as ElectraTokenizer

import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette (sns.color_palette (HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
#%matplotlib inline 
#%config InlineBackend.figure_format='retina'

pl.seed_everything(42)

class ElectraClassifier(ElectraPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    self.n_classes = config.num_labels
    self.config = config
    self.electra = ElectraModel(config)
    self.classifier = ElectraClassificationHead(config)

    self.post_init()

  def forward(
      self,
      input_ids=None,
      attention_mask=None
  ):
    discriminator_hidden_states = self.electra(input_ids, attention_mask)
    sequence_output = discriminator_hidden_states[0]
    logits = self.classifier(sequence_output)
    return logits

class EmotionClassifier(pl.LightningModule):
  def __init__(self, n_classes, learning_rate: Optional[float]=None):
    super().__init__()
    self.n_classes = n_classes
    self.classifier = ElectraClassifier.from_pretrained(
        "google/electra-base-discriminator",
        #"google/electra-base-discriminator",
        num_labels=n_classes
    )
    self.criterion = nn.CrossEntropyLoss()
    self.learning_rate = learning_rate

  def forward(self, input_ids, attention_mask):
    return self.classifier(input_ids, attention_mask)

  def run_step(self, batch, stage):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["label"].long()
    logits = self(input_ids, attention_mask)

    loss = self.criterion(logits, labels)
    self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    return loss

  def training_step(self, batch, batch_idx):
    return self.run_step(batch, "train")

  def validation_step(self, batch, batch_idx):
    return self.run_step(batch, "val")

  def test_step(self, batch, batch_idx):
    return self.run_step(batch, "test")

  def configure_optimizers(self):
    return AdamW(self.parameters(), lr=self.learning_rate)

def predict_emotion_from_utterance(text, top_k=3):
  encoding = tokenizer(
          text,
          max_length=64, 
          truncation=True,
          padding="max_length",
          add_special_tokens=True,
          return_token_type_ids=False,
          return_attention_mask=True,
          return_tensors="pt"
      )
  outputs = trained_model(**encoding)
  emotion_idx = torch.argmax(outputs, dim=-1).item() 
  predicted_emotion = emotion_categories[emotion_idx]

  return predicted_emotion