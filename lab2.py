from config import *
from utils.utils import *
from models.models import *
import torch
import os
from torchinfo import summary
import pdb

def _part1_aux(model_name, model, dataloader, optimizer):
    print(f"MODEL: {model_name}")
    
    if os.path.isfile(os.path.join(MODEL_SAVE_DIR, model_name)):
        # Load the model if exist
        print(f"MODEL {model_name} is already trained")
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, model_name)))
    else:
        # Train the model if not
        print(f"Training MODEL {model_name}")
        model = train_model(model, dataloader, CRITERION, optimizer, num_epochs=NUM_EPOCHS)
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, model_name))
    print(f"EVALUATING MODE {model_name}")
    eval_model(model, dataloader['val'], CRITERION)


def part1():
    print("BUILDING DATALOADERS...")
    dataloaders_dict_base = buildDataLoaders_denseNet()
    dataloaders_dict_data_augmentation = buildDataLoaders_denseNet(data_augmentation=True)

    # MODELO 1 -> Preentrenado + no data augmentation
    model = denseNet121_pretrained()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_DENSENET)
    _part1_aux(MODEL2_PRETRAINED_NOT_AUGMENTATION, model, dataloaders_dict_base, optimizer)

    # MODELO 2 -> Preentrenado + data augmentation
    model = denseNet121_pretrained()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_DENSENET)
    _part1_aux(MODEL1_PRETRAINED_AUGMENTATION, model, dataloaders_dict_data_augmentation, optimizer)

    # MODELO 3 -> Scratch + no data augmentation
    model = denseNet121_basic()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_DENSENET)
    _part1_aux(MODEL4_SCRATCH_NOT_AUGMENTATION, model, dataloaders_dict_base, optimizer)

    # MODELO 4 -> Scratch + data augmentation
    model = denseNet121_basic()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_DENSENET)
    _part1_aux(MODEL3_SCRATCH_AUGMENTATION, model, dataloaders_dict_data_augmentation, optimizer)


def part2():
    print("BUILDING DATALOADERS...")
    dataloader = buildDataLoaders_UNET((572,572))

    model = UNet()
    model = model.to(DEVICE)
    print(model)
    summary(model, input_size=(1, 1, 572, 572))

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)

    if os.path.isfile(os.path.join(MODEL_SAVE_DIR, MODEL_UNET)):
        # Load the model if exist
        print(f"MODEL {MODEL_UNET} is already trained")
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, MODEL_UNET)))
    else:
        # Train the model if not
        print(f"Training MODEL {MODEL_UNET}")
        model = train_unet(model, dataloader, CRITERION, optimizer, num_epochs=NUM_EPOCHS)
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, MODEL_UNET))
    print(f"EVALUATING MODE {MODEL_UNET}")
    eval_unet(model, dataloader['val'], 10)

if __name__ == "__main__":
    torch.manual_seed(88)
    print("LAB2 - PART1 EXECUTION")
    part1()
    part2()
