import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import hawtorch
import hawtorch.io as io
from hawtorch import Trainer

#import models2 as models
import models
import losses

from torchvision import datasets, transforms
from torchvision.utils import make_grid

args = io.load_json("configs_caps.json")
logger = io.logger(args["workspace_path"])


class CapsTrainer(Trainer):
    def train_step(self, data):
        inputs, truths = data
        truths = torch.eye(args["num_classes"]).to(inputs.device)[truths]
        outputs, recons = self.model(inputs, truths)
        loss = self.objective(inputs, truths, outputs, recons)
        return outputs, loss

    def eval_step(self, data):
        inputs, truths = data
        truths = torch.eye(args["num_classes"]).to(inputs.device)[truths]
        outputs, recons = self.model(inputs, truths)
        loss = self.objective(inputs, truths, outputs, recons)
        # save recons
        if self.local_step % 10 == 0:
            recons = recons.view(-1, 1, 28, 28)
            recons = make_grid(recons, normalize=True, scale_each=True)
            self.writer.add_image("Image", recons, self.epoch * 10000 + self.local_step)
        return outputs, loss


def create_trainer():
    logger.info("Start creating trainer...")
    logger.logblock(args)

    device = args["device"]
    model = getattr(models, args["model"])(args)
    #model = getattr(models, args["model"])()

    logger.logblock(model)

    objective = getattr(losses, args["objective"])()
    optimizer = getattr(optim, args["optimizer"])(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    trainer = globals()[args["trainer"]]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args["lr_decay_step"], gamma=args["lr_decay"])
    metrics = [hawtorch.metrics.ClassificationAverager(args["num_classes"]), ]
    loaders = create_loaders()

    mytrainer = trainer(model, optimizer, scheduler, objective, device, loaders, logger,
                  metrics=metrics, 
                  workspace_path=args["workspace_path"],
                  eval_set="test",
                  report_step_interval=-1,
                  )

    logger.info("Trainer Created!")

    return mytrainer


def create_loaders():
    train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ])),
                batch_size=args["train_batch_size"], 
                shuffle=True,
                num_workers=1,
                pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./data', train=False, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ])),
                batch_size=args["test_batch_size"],
                shuffle=True,
                num_workers=1,
                pin_memory=True)

    print("create loaders", len(train_loader), len(test_loader))

    loaders = {
        "train": train_loader,
        "test": test_loader
    }

    return loaders



if __name__ == "__main__":
    trainer = create_trainer()
    trainer.train(args["epochs"])
    #trainer.evaluate()
    
