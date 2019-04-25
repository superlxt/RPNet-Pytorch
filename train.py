from torch.autograd import Variable
import torch.nn as nn
import torch
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils



class Train():
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - use_cuda (``bool``): If ``True``, the training is performed using
    CUDA operations (GPU).

    """

    def __init__(self, model, data_loader, optim, criterion, metric, use_cuda, step):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.use_cuda = use_cuda
        self.step = step

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
        
            inputs, labels = batch_data

            # Use augmentation
            inputs, labels = utils.dataaug(inputs, labels)
 
            # Wrap them in a Varaible
            inputs, labels = Variable(inputs), Variable(labels.float())


            if self.use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            labels=labels.unsqueeze(1)

            labels2 = torch.nn.functional.interpolate(labels, scale_factor=0.5, mode='nearest').squeeze(1)
            labels3 = torch.nn.functional.interpolate(labels, scale_factor=0.25, mode='nearest').squeeze(1)
            labels4 = torch.nn.functional.interpolate(labels, scale_factor=0.125, mode='nearest').squeeze(1)
            labels1 = labels.squeeze(1)


            # Forward propagation
            outputs1, outputs2, outputs3, outputs4 = self.model(inputs)

            # Loss computation
            loss1 = self.criterion(outputs1, labels1.long())
            loss2 = self.criterion(outputs2, labels2.long())
            loss3 = self.criterion(outputs3, labels3.long())
            loss4 = self.criterion(outputs4, labels4.long())

            # Step Loss
            loss= eval('loss{}'.format(self.step))

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Metric
            self.metric.add(eval('outputs{}'.format(self.step)).data, eval('labels{}'.format(self.step)).data)

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
