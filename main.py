import pathlib
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.crop_model import get_crop_model
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
        
        
    validation_acc = sum(validation_acc)/len(validation_acc)
    validation_loss = sum(validation_loss)/len(validation_loss)
    
    print(f"Validation accuracy: {validation_acc}")
    print(f"Validation loss: {validation_loss}")
    
    return validation_acc, validation_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="crop")
    args = parser.parse_args()

    learning_rate = 0.0001
    batch_size = 8
    model_size = "large" # small, medium, large
    with_resnet = True # True, False
    epochs = 100
    bands = 12 # Number of bands in the input data
    weeks = 10 # Number of weeks in the input data
    height, width = 25, 25 # Height and width of the input data
    n_classes = 4 # Number of classes for the crop model
    weights_output_folder = pathlib.Path("weights")
    weights_output_folder.mkdir(exist_ok=True)
    experiment_name = f"{args.model}_{model_size}_{'resnet' if with_resnet else 'convnet'}_lr{learning_rate}_bs{batch_size}_b{bands}_w{weeks}"
    experiment_name = "experiment_1" # --> Definer dette et annet sted
    
    
    # 10% av data ---> 100% av data
    

    if args.model == "crop":
        train_loader = get_crop_dataset_loader("data/crop/train", batch_size)
        val_loader = get_crop_dataset_loader("data/crop/test", 1, shuffle=False)
        model = get_crop_model("resnet_crop", (weeks, bands, height, width), n_classes, model_size)
    elif args.model == "yield":
        train_loader = get_yield_dataset_loader("data/yield/train", 8)
        val_loader = get_yield_dataset_loader("data/yield/test", 1, shuffle=False)
        model = YieldModel(17, 12, None)

    cuda_idx = 1

    model = model.to(f"cuda:{cuda_idx}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 100
    
    import math
    
    validation_acc = -math.inf
    validation_loss = math.inf
    
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, train_loader, cuda_idx=cuda_idx)
        temp_validation_acc, temp_validation_loss = validate(model, val_loader, cuda_idx=cuda_idx)


        if temp_validation_loss > validation_loss:
            validation_acc = temp_validation_acc
            validation_loss = temp_validation_loss
            
            torch.save(model.state_dict(), f"{weights_output_folder}/{experiment_name}.pt")
        
    

    """

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'trained_models/{args.model}/',
        filename=args.model + '-{epoch:02d}-{val_loss:.2f}'
    )

    trainer = L.Trainer(callbacks=[checkpoint_callback], accelerator='gpu', devices=[0], max_epochs=100)


    trainer.fit(lightning_model, train_loader, val_loader)
    """
