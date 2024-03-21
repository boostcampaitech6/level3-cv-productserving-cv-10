import os,json
import torch
import wandb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from transformers import(
    Pix2StructForConditionalGeneration, 
    Pix2StructProcessor,
    Trainer,
    TrainingArguments
)

# import albumentations as A
# from albumentations.pytorch import ToTensorV2

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

torch.multiprocessing.set_start_method('spawn')

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-infographics-vqa-base").to("cuda")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-infographics-vqa-base")

# def wandb_log(**kwargs):
#     for k, v in kwargs.items():
#         wandb.log({k: v})

class Pix2StructDataset(Dataset):
    def __init__(self, image_dir, json_dir, processor, train):
        self.img_dir = image_dir
        with open(json_dir) as f:
            self.json_data = json.load(f)
        self.processor = processor
        self.file_list = os.listdir(image_dir)
        self.train = train
        
    def __getitem__(self, index): 
        data = self.json_data["data"][index]
        image_name = data["image_local_name"]
        img = Image.open(os.path.join(self.img_dir, image_name))
        q = data["question"]
        inputs = self.processor(images=img, text=q, return_tensors="pt").to("cuda")
        if self.train:
            a = data["answers"][0]
            label = self.processor.tokenizer(text=a, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=45).input_ids.to("cuda")
            inputs["labels"] = label
            return inputs
        return inputs
  
    
    def __len__(self): 
        return len(self.json_data['data'])
    
def collator(batch):
  new_batch = {"flattened_patches":[], "attention_mask":[], "labels":[]}
  
  for item in batch:
    new_batch["flattened_patches"].append(item["flattened_patches"][0])
    new_batch["attention_mask"].append(item["attention_mask"][0])
    new_batch["labels"].append(item["labels"][0])
  new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
  new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
  new_batch["labels"] = torch.stack(new_batch["labels"])

  return new_batch

img_dir = './hy_info/task3//images/'
train_dataset = Pix2StructDataset(image_dir=img_dir, json_dir='./hy_info/task3/qas/infographicsVQA_train_v1.0.json', processor=processor, train=True)
val_dataset = Pix2StructDataset(image_dir=img_dir, json_dir='./hy_info/task3/qas/infographicsVQA_val_v1.0_withQT.json', processor=processor, train=True)
test_dataset = Pix2StructDataset(image_dir=img_dir, json_dir='./hy_info/task3/qas/infographicsVQA_test_v1.0.json', processor=processor, train=False)
# train_dataset = Pix2StructDataset(image_dir=img_dir, json_dir='./hy_info/Task3_test/qas/infographicsVQA_train_v1.0.json', processor=processor, train=True)
# val_dataset = Pix2StructDataset(image_dir=img_dir, json_dir='./hy_info/Task3_test/qas/infographicsVQA_val_v1.0_withQT.json', processor=processor, train=True)
# test_dataset = Pix2StructDataset(image_dir=img_dir, json_dir='./hy_info/Task3_test/qas/infographicsVQA_test_v1.0.json', processor=processor, train=False)



training_args = TrainingArguments(
    output_dir="pix2struct_1",
    learning_rate=2e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator
)

trainer.train()