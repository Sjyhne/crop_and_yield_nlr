import pathlib
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.crop_model import CropModel, ResNetCropModel
from models.yield_model import YieldModel


from loader.crop_dataset import get_crop_dataset_loader, get_yield_dataset_loader


def train_one_epoch(model, optimizer, train_loader, cuda_idx=0):
    model.train()
    
    training_acc = list()
    training_loss = list()
    
    for idx, (images, labels, info) in tqdm(enumerate(train_loader), total=len(train_loader)):
        
        images = images.to(f"cuda:{cuda_idx}")
        labels = labels.to(f"cuda:{cuda_idx}")
        
        optimizer.zero_grad()

        preds = model(images)
        
        predictions = torch.argmax(torch.softmax(preds, dim=1), dim=1)
        
        loss = F.cross_entropy(preds, labels)
        
        training_loss.append(loss.item())
        training_acc.append(torch.sum(predictions == labels).item() / len(labels))
        
        loss.backward()
        optimizer.step()
    
    print(f"Train accuracy: {sum(training_acc)/len(training_acc)}")
    print(f"Train loss: {sum(training_loss)/len(training_loss)}")
    
def validate(model, val_loader, cuda_idx=0):
    model.eval()
    
    validation_acc = list()
    validation_loss = list()
    
    for _, (images, labels, info) in tqdm(enumerate(val_loader), total=len(val_loader)):
        
        images = images.to(f"cuda:{cuda_idx}")
        labels = labels.to(f"cuda:{cuda_idx}")
        
        preds = model(images)
        
        predictions = torch.argmax(torch.softmax(preds, dim=1), dim=1)
        
        loss = F.cross_entropy(preds, labels)
        
        validation_loss.append(loss.item())
        validation_acc.append(torch.sum(predictions == labels).item() / len(labels))
        
        
    print(f"Validation accuracy: {sum(validation_acc)/len(validation_acc)}")
    print(f"Validation loss: {sum(validation_loss)/len(validation_loss)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="crop")
    args = parser.parse_args()

    learning_rate = 0.0001

    if args.model == "crop":
        train_loader = get_crop_dataset_loader("data/crop/train", 8)
        val_loader = get_crop_dataset_loader("data/crop/test", 1, shuffle=False)
        model = CropModel((17, 12, 25, 25), 4)
        model = ResNetCropModel((17, 12, 25, 25), 4)
    elif args.model == "yield":
        train_loader = get_yield_dataset_loader("data/yield/train", 8)
        val_loader = get_yield_dataset_loader("data/yield/test", 1, shuffle=False)
        model = YieldModel(17, 12, None)

    cuda_idx = 3

    model = model.to(f"cuda:{cuda_idx}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 100
    
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, train_loader, cuda_idx=cuda_idx)
        validate(model, val_loader, cuda_idx=cuda_idx)
    

    """

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'trained_models/{args.model}/',
        filename=args.model + '-{epoch:02d}-{val_loss:.2f}'
    )

    trainer = L.Trainer(callbacks=[checkpoint_callback], accelerator='gpu', devices=[0], max_epochs=100)


    trainer.fit(lightning_model, train_loader, val_loader)
    """
