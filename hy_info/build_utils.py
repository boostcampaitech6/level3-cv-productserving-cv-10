
import torch
import transformers

from transformers import get_scheduler


def build_optimizer(model, length_train_loader, config):
    optimizer_class = getattr(transformers, 'AdamW')
    optimizer = optimizer_class(model.model.parameters(), lr=float(config['lr']))
    num_training_steps = config['train_epochs'] * length_train_loader
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=config['warmup_iterations'], num_training_steps=num_training_steps
    )

    return optimizer, lr_scheduler


def build_model(config):

    available_models = ['layoutlmv2', 'layoutlmv3','hy']
    if config['model_name'].lower() == 'layoutlmv2':
        from models.LayoutLMv2 import LayoutLMv2
        model = LayoutLMv2(config)

    elif config['model_name'].lower() == 'layoutlmv3':
        from models.LayoutLMv3 import LayoutLMv3
        model = LayoutLMv3(config)

    elif config['model_name'].lower() == 'hy':
        from models.LayoutLMv3_hy import LayoutLMv3_hy
        model = LayoutLMv3_hy(config)

    else:
        raise ValueError("Value '{:s}' for model selection not expected. Please choose one of {:}".format(config['model_name'], ', '.join(available_models)))

    if config['device'] == 'cuda' and config['data_parallel'] and torch.cuda.device_count() > 1:
        model.parallelize()

    model.model.to(config['device'])
    return model


def build_dataset(config, split):

    # Specify special params for data processing depending on the model used.
    dataset_kwargs = {}

    if config['model_name'].lower() in ['layoutlmv2', 'layoutlmv3', 'lt5', 'vt5', 'hilt5', 'hi-lt5', 'hivt5', 'hi-vt5','hy']:
        dataset_kwargs['get_raw_ocr_data'] = True

    if config['model_name'].lower() in ['layoutlmv2', 'layoutlmv3', 'vt5', 'hivt5', 'hi-vt5','hy']:
        dataset_kwargs['use_images'] = True

    # Build dataset
    if config['dataset_name'] == 'infographicVQA':
        from datasets.IF_DocVQA import IFDocVQA
        dataset = IFDocVQA(config['imdb_dir'], config['images_dir'], split, dataset_kwargs)
    else:
        raise ValueError

    return dataset
