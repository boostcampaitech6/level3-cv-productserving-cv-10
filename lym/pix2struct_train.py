import os
import sys
import cv2
import json
import glob
import numpy as np
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
# import matplotlib.pyplot as plt

from random import shuffle

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

from transformers import (
    AutoProcessor,
    Pix2StructConfig,
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.simplefilter("ignore")

seed = 50
os.environ['PYTHONHASHSEED'] = str(seed)
# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

Config = {
    'IMAGE_DIR': '/home/lym/level3-cv-productserving-cv-10/data_select/task3_infographic/images/',
    'MAX_PATCHES': 1024,
    # 'MODEL_NAME': 'ybelkada/pix2struct-base',
    'MODEL_NAME': 'google/pix2struct-base',
    # 'MODEL_NAME': "google/pix2struct-infographics-vqa-base",
    'IMG_SIZE': (256, 256),
    'MAX_LEN': 256,
    'LR': 3e-5,
    'NB_EPOCHS': 10,
    'TRAIN_BS': 6,
    'VALID_BS': 2,
    'ALL_SAMPLES': int(1e+100),
    '_wandb_kernel': 'tanaym',
}

def wandb_log(**kwargs):
    for k, v in kwargs.items():
        wandb.log({k: v})

# Start W&B logging
# W&B Login
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# wb_key = user_secrets.get_secret("WANDB_API_KEY")

# wandb.login(key=wb_key)

run = wandb.init(
    project='pytorch',
    config=Config,
    group='multi_modal',
    job_type='train',
)

# Let's add chart types as special tokens and a special BOS token
BOS_TOKEN = "<|BOS|>"
X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"

new_tokens = [
    "<line>",
    "<vertical_bar>",
    "<scatter>",
    "<dot>",
    "<horizontal_bar>",
    X_START,
    X_END,
    Y_START,
    Y_END,
    BOS_TOKEN,
]

def augments():
    return A.Compose([
        A.Resize(width=Config['IMG_SIZE'][0], height=Config['IMG_SIZE'][1]),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ])

class BeneTechDataset(Dataset):
    def __init__(self, dataset, processor, augments=None):
        self.dataset = dataset
        self.processor = processor
        self.augments = augments

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img_path = '/home/lym/level3-cv-productserving-cv-10/data/images/'+ item['image_local_name']
        # img_path = '/home/lym/level3-cv-productserving-cv-10/data_select/task3_infographic/images/'+ item['image_local_name']
        print("img ->", item['image_local_name'])
        image = np.array(Image.open(img_path))
        # image = cv2.imread(item['image'])
        if self.augments:
            image = self.augments(image=image)['image']
        encoding = self.processor(
            images=image,
            return_tensors="pt", 
            add_special_tokens=True, 
            max_patches=Config['MAX_PATCHES']
        )
        
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = item['answers']
        return encoding
    
def get_model(extra_tokens=new_tokens):
    # processor = AutoProcessor.from_pretrained(Config['MODEL_NAME'])
    processor = Pix2StructProcessor.from_pretrained(Config['MODEL_NAME'])
    model = Pix2StructForConditionalGeneration.from_pretrained(Config['MODEL_NAME'])
    processor.image_processor.size = {
        "height": Config['IMG_SIZE'][0],
        "width": Config['IMG_SIZE'][1],
    }
    processor.image_processor.is_vqa = False

    processor.tokenizer.add_tokens(extra_tokens)
    model.resize_token_embeddings(len(processor.tokenizer))
    return processor, model


def collator(batch):
    new_batch = {"flattened_patches":[], "attention_mask":[]}
    
    texts = [item["text"][0] for item in batch]
    text_inputs = processor(
        text=texts, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt", 
        add_special_tokens=True, 
        max_length=Config['MAX_LEN']
    )
    new_batch["labels"] = text_inputs.input_ids
    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])
    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch

def train_one_epoch(model, processor, train_loader, optimizer, scaler):
    """
    Trains the model on all batches for one epoch with NVIDIA's AMP
    """
    
    model.train()
    avg_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with autocast():
        prog_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, batch in prog_bar:
            # print("icx, batch->", idx, batch.keys())
            labels = batch.pop("labels").to('cuda')
            flattened_patches = batch.pop("flattened_patches").to('cuda')
            attention_mask = batch.pop("attention_mask").to('cuda')
            # for i in labels:
            #     print("labels ->", processor.decode(i))
            outputs = model(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                labels=labels
            )
            # print(outputs.logits)
            # print(" -> ", [processor.decode(index) for index in outputs.decoder_input_ids[0]])
            # for i in outputs:
            #     print(i)
                # print(processor.decode(outputs.logits.argmax(dim=-1)))
            
            loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            prog_bar.set_description(f"loss: {loss.item():.4f}")
            wandb_log(train_step_loss=loss.item())
            avg_loss += loss.item()
            
    avg_loss = avg_loss / len(train_loader)
    print(f"Average training loss: {avg_loss:.4f}")
    wandb_log(train_loss=avg_loss)
    return avg_loss

@torch.no_grad()
def valid_one_epoch(model, processor, valid_loader):
    """
    Validates the model on all batches (in val set) for one epoch
    """
    model.eval()
    avg_loss = 0
    correct_predictions = 0
    total_predictions = 0

    prog_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for idx, batch in prog_bar:
        labels = batch.pop("labels").to('cuda')
        flattened_patches = batch.pop("flattened_patches").to('cuda')
        attention_mask = batch.pop("attention_mask").to('cuda')
        
        outputs = model(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        prog_bar.set_description(f"loss: {loss.item():.4f}")
        wandb_log(val_step_loss=loss.item())
        avg_loss += loss.item()
    
        # predictions = outputs.logits.argmax(dim=-1)
        # correct_predictions = (predictions == labels).sum().item()
        # total_predictions += correct_predictions
        
    avg_loss = avg_loss / len(valid_loader)
    # accuracy = correct_predictions / total_predictions
    print(f"Average validation loss: {avg_loss:.4f}")
    # print(f"Validation accuracy: {accuracy:.4f}")
    wandb_log(val_loss=avg_loss)
    return avg_loss


def fit(model, processor, train_loader, valid_loader, optimizer, scaler):
    """
    A nice function that binds it all together and reminds me of Keras days from 2018 :)
    """
    best_val_loss = int(1e+5)
    # print('->->')
    for epoch in range(Config['NB_EPOCHS']):
        # print('->->->')
        print(f"{'='*20} Epoch: {epoch+1} / {Config['NB_EPOCHS']} {'='*20}")
        _ = train_one_epoch(model, processor, train_loader, optimizer, scaler)
        val_avg_loss = valid_one_epoch(model, processor, valid_loader)
        
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            print(f"Saving best model so far with loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), f"pix2struct_base_benetech.pt")
    print(f"Best model with val_loss: {best_val_loss:.4f}")

# Training cell
if __name__ == "__main__":
    # Read the processed JSON file
    with open("/home/lym/level3-cv-productserving-cv-10/data/qas/infographicsVQA_train_v1.0_custom.json", "r") as fl:
        train_dataset = json.load(fl)['data']
    # with open("/home/lym/level3-cv-productserving-cv-10/data/qas/infographicsVQA_val_v1.0_withQT.json", "r") as fl:
    #     val_dataset = json.load(fl)['data']
        
    # Shuffle the dataset and select however samples you want for training
    # shuffle(train_dataset)  # 이거 나중에 풀자.
    # dataset = dataset[:Config['ALL_SAMPLES']]
    
    # We are splitting the data naively for now
    split = 0.90
    # train_samples = int(len(dataset) * split)
    # train_ds = dataset[:train_samples+1]
    # valid_ds = dataset[train_samples:]
    # train_ds = train_dataset
    # valid_ds = val_dataset

    train_samples = int(len(train_dataset) * split)
    train_ds = train_dataset[:train_samples+1]
    valid_ds = train_dataset[train_samples:]

    # Yeah all that
    processor, model = get_model()
    # check_point = 'pix2struct_base_benetech.pt'
    
    # checkpoint = torch.load(check_point)
    # model.load_state_dict(torch.load(check_point))

    model.to('cuda')
    wandb.watch(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=Config['LR'])
    
    # Load the data into Datasets and then make DataLoaders for training
    train_dataset = BeneTechDataset(train_ds, processor, augments=augments())
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=Config['TRAIN_BS'], collate_fn=collator) # sulffle true 주자
    
    valid_dataset = BeneTechDataset(valid_ds, processor, augments=augments())
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=Config['VALID_BS'], collate_fn=collator)
    
    # nb_train_steps = int(train_samples / Config['TRAIN_BS'] * Config['NB_EPOCHS'])
    
    # Print out the data sizes we are training on
    print(f"Training on {len(train_ds)} samples, Validating on {len(valid_ds)} samples")
    # Train the model now
    fit(
        model=model,
        processor=processor,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        optimizer=optimizer,
        scaler=GradScaler(),
    )


