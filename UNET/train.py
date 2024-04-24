import torch
import torch.nn as nn
from CustomDataset import CustomDataset
from Unet import UNet
from torch.optim import Adam
import tqdm
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-paths', type=str, default='data/images/*.PNG', help='Path to images (default: \'data/images/*.PNG\')')
    parser.add_argument('--mask-paths', type=str, default='data/masks/*.PNG', help='Path to masks (default: \'data/masks/*.PNG\')')                                                                         
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--save-model', action='store_true', help='Save model (default: False)')
    args = parser.parse_args()


    img_paths = args.img_paths
    mask_paths = args.mask_paths
    bboxes = args.bboxes
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # we use l1 loss
    criterion_bbox = nn.L1Loss()
    # segmentation loss is binary cross entropy
    criterion_seg = nn.BCEWithLogitsLoss()

    dataset = CustomDataset(img_paths=img_paths, bboxes=bboxes, mask_paths=mask_paths)
    
    model = UNet()
    optimizer = Adam(model.parameters(), lr=lr)

    # split dataset into train and validation sets
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # move model to device
    model = model.to(device)

    def train_model(model, train_dataloader, val_dataloader, criterion_bbox, criterion_seg, optimizer, num_epochs):
        train_losses = []
        val_losses = []
        best_model = None

        for epoch in tqdm.tqdm(range(num_epochs)):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            model.train()   
            train_loss = 0.0
            val_loss = 0.0

            for img, mask in train_dataloader:

                img = img.to(device)
                mask = mask.to(device)

                img = img.unsqueeze(1)
                mask = mask.unsqueeze(1)
                
                pred_mask,pred_bbox = model(img)
                # threshold the mask
                pred_mask = torch.where(pred_mask > 0.5, 1, 0)

                pred_mask = pred_mask.float()
                
                # calculate loss
                loss_bbox = criterion_bbox(pred_bbox, bbox)
                loss_seg = criterion_seg(pred_mask, mask)
                loss = loss_bbox + loss_seg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * img.size(0)


            model.eval()
            with torch.no_grad():
                for img, bbox,mask in val_dataloader:
                    img = img.to(device)
                    bbox = bbox.to(device)
                    mask = mask.to(device)

                    img = img.unsqueeze(1)
                    bbox = bbox.squeeze(1)
                    mask = mask.unsqueeze(1)

                    pred_mask,pred_bbox  = model(img)

                    pred_mask = torch.where(pred_mask > 0.5, 1, 0)
                    pred_mask = pred_mask.float()

                    loss_bbox = criterion_bbox(pred_bbox, bbox)
                    loss_seg = criterion_seg(pred_mask, mask)
                    loss = loss_bbox + loss_seg

                    val_loss += loss.item() * img.size(0)

            train_loss = train_loss / train_size
            val_loss = val_loss / val_size
            
            print('Train Loss: {:.4f}'.format(train_loss))
            print('Val Loss: {:.4f}'.format(val_loss))
            
            if val_loss < min(val_losses, default=1000):
                print('Best model updated')
                best_model = model

            train_losses.append(train_loss)
            val_losses.append(val_loss)


            print()

        return best_model, train_losses, val_losses
    model,train_losses,val_losses = train_model(model, train_dataloader, val_dataloader, criterion_bbox=criterion_bbox,criterion_seg=criterion_seg, optimizer=optimizer, num_epochs=num_epochs)
    if args.save_model:
        model_name = 'model_batch_size_{}_num_epochs_{}_lr_{}.pth'.format(batch_size, num_epochs, lr)
        torch.save(model.state_dict(), model_name)
        print('Model saved to model.pth')

    # plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    # save plot to file
    plt.savefig('loss_plot.png')
    plt.show()


if __name__ == '__main__':
    main()
