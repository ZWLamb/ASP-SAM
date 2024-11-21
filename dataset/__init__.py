import torchvision.transforms as transforms
from .dsb import DSB
from torch.utils.data import DataLoader

def get_dataloader(args):
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_train_seg = transforms.Compose([
        transforms.Resize((args.out_size, args.out_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test_seg = transforms.Compose([
        transforms.Resize((args.out_size, args.out_size)),
        transforms.ToTensor(),
    ])
    '''dsb data'''
    dsb_train_dataset = DSB(args, args.data_path, transform=transform_train,
                            transform_msk=transform_train_seg, mode='Training')
    dsb_test_dataset = DSB(args, args.data_path, transform=transform_test, transform_msk=transform_test_seg,
                           mode='Test')

    train_loader = DataLoader(dsb_train_dataset, batch_size=args.b, shuffle=True, num_workers=8,
                                   pin_memory=True)
    test_loader = DataLoader(dsb_test_dataset, batch_size=args.b_test, shuffle=False, num_workers=8,
                                  pin_memory=True)

    return train_loader, test_loader