import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader

from Dataset import SIDDataset
from Model import SwinDenoiser

transforms = torch.nn.Sequential(
    torchvision.transforms.Resize((244, 244)),
)

def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = SIDDataset('SIDD', transform=None)
    SIDDLoader = DataLoader(dataset=dataset, pin_memory=True, batch_size=1, num_workers=4, shuffle=True)

    model = SwinDenoiser()

    criterion = nn.MSELoss()
    optimizer = optim.RAdam(model.parameters(), lr=0.001)

    for epoch in range(300):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in enumerate(SIDDLoader):
            inputs, labels = data

            inputs = inputs[0]
            labels = labels[0]

            inputs = inputs
            labels = labels

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model.forward(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.4f}')
            running_loss = 0.0

        # calculate validation loss

if __name__ == '__main__':
    print('Sudoku CNN started')

    torch.multiprocessing.freeze_support()

    run()
