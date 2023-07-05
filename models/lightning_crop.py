import lightning as L
import torch
import torch.nn.functional as F

class Crop(L.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

        self.train_acc = list()
        self.train_loss = list()

        self.val_acc = list()
        self.val_loss = list()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels, info = batch

        preds = self.forward(images)

        predictions = torch.argmax(torch.softmax(preds, dim=1), dim=1)

        self.train_acc.append(torch.sum(predictions == labels).item() / len(labels))

        loss = F.cross_entropy(preds, labels)

        self.train_loss.append(loss.item())

        return {'loss': loss}


    def on_train_epoch_end(self) -> None:
        print(f"Train accuracy: {sum(self.train_acc)/len(self.train_acc)}")
        print(f"Train loss: {sum(self.train_loss)/len(self.train_loss)}")

    def validation_step(self, batch, batch_idx):
        images, labels, info = batch

        preds = self.forward(images)

        predictions = torch.argmax(torch.softmax(preds, dim=1), dim=1)

        self.val_acc.append(torch.sum(predictions == labels).item() / len(labels))

        loss = F.cross_entropy(preds, labels)

        self.val_loss.append(loss.item())

        return {'loss': loss}

    def on_validation_epoch_end(self) -> None:
        print(f"Val accuracy: {sum(self.val_acc)/len(self.val_acc)}")
        print(f"Val loss: {sum(self.val_loss)/len(self.val_loss)}")
        self.log("val_acc", sum(self.val_acc)/len(self.val_acc))
        self.log("val_loss", sum(self.val_loss)/len(self.val_loss))


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

