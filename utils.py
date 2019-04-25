import torch
import torch as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2

def batch_transform(batch, transform):
    """Applies a transform to a batch of samples.

    Keyword arguments:
    - batch (): a batch os samples
    - transform (callable): A function/transform to apply to ``batch``

    """

    # Convert the single channel label to RGB in tensor form
    # 1. F.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = [transform(tensor) for tensor in F.unbind(batch)]

    return F.stack(transf_slices)


def imshow_batch(images, labels):
    """Displays two grids of images. The top grid displays ``images``
    and the bottom grid ``labels``

    Keyword arguments:
    - images (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    - labels (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)

    """

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()


def save_checkpoint(model, optimizer, epoch, miou, args):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - args (``ArgumentParser``): An instance of ArgumentParser which contains
    the arguments used to train ``model``. The arguments are written to a text
    file in ``args.save_dir`` named "``args.name``_args.txt".

    """
    name = args.name
    save_dir = args.save_dir

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)

    # Save arguments
    summary_filename = os.path.join(save_dir, name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Mean IoU: {0}\n". format(miou))


def load_checkpoint(model, optimizer, folder_dir, filename):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - optimizer (``torch.optim``): The stored optimizer state is copied to this
    optimizer instance.
    - folder_dir (``string``): The path to the folder where the saved model
    state is located.
    - filename (``string``): The model filename.

    Returns:
    The epoch, mean IoU, ``model``, and ``optimizer`` loaded from the
    checkpoint.

    """
    assert os.path.isdir(
        folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # Create folder to save model and information
    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(
        model_path), "The model file \"{0}\" doesn't exist.".format(filename)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = 0
    miou = 0

    return model, optimizer, epoch, miou



def dataaug(inputs, labels):
    """use flip and warpAffine
    """
    for i in range(inputs.shape[0]):
        inputs1 = inputs[i, :, :, :]
        labels1 = labels[i, :, :]
        inputs1 = inputs1.numpy()
        labels1 = labels1.numpy()
        inputs1 = inputs1.transpose((1, 2, 0))

        x = random.randint(-2, 2)
        y = random.randint(-2, 2)
        H = np.float32([[1,0,x],[0,1,y]])
        inputs1 = cv2.warpAffine(inputs1,H,(inputs1.shape[1],inputs1.shape[0]))
        labels1 = cv2.warpAffine(labels1.astype(np.float32),H,(inputs1.shape[1],inputs1.shape[0])).astype(np.uint8) 


        if random.random() < 0.5:
            inputs1 = cv2.flip(inputs1, 1) # horizontal flip
            labels1 = cv2.flip(labels1, 1) # horizontal flip
            
        inputs1 = inputs1.transpose((2, 0, 1))
        inputs1 = torch.from_numpy(inputs1)
        labels1 = torch.from_numpy(labels1)
        #print(inputs.shape)
        inputs[i, : , : ,:]=inputs1
        labels[i, : ,:]=labels1
    return inputs, labels
