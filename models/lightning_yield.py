import lightning as L
import torch
import torch.nn.functional as F

class Yield(L.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

        self.train_loss = list()

        self.val_loss = list()

    def forward(self, images, weather, feature):
        return self.model(images, weather, feature)

    def training_step(self, batch, batch_idx):
        images, weather, feature, labels, info = batch

        preds = self.forward(images, weather, feature)

        loss = F.l1_loss(preds, labels)

        self.train_loss.append(loss.item())

        return {'loss': loss}


    def on_train_epoch_end(self) -> None:
        print(f"Train loss: {sum(self.train_loss)/len(self.train_loss)}")

    def validation_step(self, batch, batch_idx):
        images, weather, feature, labels, info = batch

        preds = self.forward(images, weather, feature)

        loss = F.l1_loss(preds, labels)

        self.val_loss.append(loss.item())

        return {'loss': loss}

    def on_validation_epoch_end(self) -> None:
        print(f"Val loss: {sum(self.val_loss)/len(self.val_loss)}")
        self.log("val_loss", sum(self.val_loss)/len(self.val_loss))


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

