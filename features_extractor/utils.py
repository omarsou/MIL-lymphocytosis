import torchvision.transforms.functional as TF
import random
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pickle
import gzip


class MyRotateTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def train(log_interval, model, device, train_loader, optimizer, epoch, criterion=None, best=None,
          save_model_path='/content/drive/MyDrive/DLMI_Challenge/'):

    # set model as training mode
    model.train()

    losses_mse = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, X in enumerate(train_loader):
        # distribute data to device
        X = X.to(device)
        N_count += X.size(0)

        optimizer.zero_grad()
        X_reconst, _, _ = model(X)
        loss_total = criterion(X_reconst, X)
        losses_mse.append(loss_total.item())
        loss_total.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss MSE: {:.6f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss_total.item()))

        # Show some pairs of image, reconstructed image during training
        if (batch_idx + 1) % 150 == 0:
            example = X_reconst.data[0].cpu()
            x = X.data[0].cpu()
            img2 = transforms.ToPILImage(mode='RGB')(example)
            img = transforms.ToPILImage(mode='RGB')(x)
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.subplot(2, 2, 2)
            plt.imshow(img2)
            plt.show()

    # save Pytorch models of best record
    if sum(losses_mse)/len(losses_mse) < best:
        torch.save(model.state_dict(), os.path.join(save_model_path, 'model_vae_efficient.pth'))  # save motion_encoder
        torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_vae_efficient.pth'))  # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))

    return losses_mse


def save(object, filename, protocol=0):
    """Saves a compressed object to disk
    """
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, protocol))
    file.close()
