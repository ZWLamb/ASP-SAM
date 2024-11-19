import torchvision.transforms as transforms
from .dsb import DSB
from .liz import LIZ
from .consep import CONSEP

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
    if args.dataset == 'DSB':
        '''dsb data'''
        dsb_train_dataset = DSB(args, args.data_path, transform=transform_train,
                                transform_msk=transform_train_seg, mode='Training')
        dsb_test_dataset = DSB(args, args.data_path, transform=transform_test, transform_msk=transform_test_seg,
                               mode='Test')

        nice_train_loader = DataLoader(dsb_train_dataset, batch_size=args.b, shuffle=True, num_workers=8,
                                       pin_memory=True)
        nice_test_loader = DataLoader(dsb_test_dataset, batch_size=args.b_test, shuffle=False, num_workers=8,
                                      pin_memory=True)
    elif args.dataset == 'LIZ' or args.dataset == 'GLAS' or args.dataset == 'MoNuSeg':
        '''data'''
        dsb_train_dataset = LIZ(args, args.data_path, transform=transform_train,
                                transform_msk=transform_train_seg, mode='Training')
        dsb_test_dataset = LIZ(args, args.data_path, transform=transform_test, transform_msk=transform_test_seg,
                               mode='Test')

        nice_train_loader = DataLoader(dsb_train_dataset, batch_size=args.b, shuffle=True, num_workers=8,
                                       pin_memory=True)
        nice_test_loader = DataLoader(dsb_test_dataset, batch_size=args.b_test, shuffle=False, num_workers=8,
                                      pin_memory=True)

    elif args.dataset == 'PAN':
        '''data'''
        dsb_train_dataset = LIZ(args, args.data_path, transform=transform_train,
                                transform_msk=transform_train_seg, mode='Training')
        dsb_test_dataset = LIZ(args, args.data_path, transform=transform_test, transform_msk=transform_test_seg,
                               mode='Test',max_size = 100)

        nice_train_loader = DataLoader(dsb_train_dataset, batch_size=args.b, shuffle=True, num_workers=8,
                                       pin_memory=True)
        nice_test_loader = DataLoader(dsb_test_dataset, batch_size=args.b_test, shuffle=False, num_workers=8,
                                      pin_memory=True)
    elif args.dataset == 'CryoNuSeg':
        '''data'''
        dsb_train_dataset = LIZ(args, args.data_path, transform=transform_train,
                                transform_msk=transform_train_seg, mode='Training')
        dsb_test_dataset = LIZ(args, args.data_path, transform=transform_test, transform_msk=transform_test_seg,
                               mode='Test')

        nice_train_loader = DataLoader(dsb_train_dataset, batch_size=args.b, shuffle=True, num_workers=8,
                                       pin_memory=True)
        nice_test_loader = DataLoader(dsb_test_dataset, batch_size=args.b_test, shuffle=False, num_workers=8,
                                      pin_memory=True)
    # elif args.dataset == 'CONSEP' :
    #     consep_train_dataset = CONSEP(args, args.data_path, transform=transform_train,
    #                                   transform_msk=transform_train_seg, mode='Training')
    #     consep_test_dataset = CONSEP(args, args.data_path, transform=transform_test, transform_msk=transform_test_seg,
    #                                  mode='Test')
    #     nice_train_loader = DataLoader(consep_train_dataset, batch_size=args.b, shuffle=True, num_workers=8,
    #                                    pin_memory=True)
    #     nice_test_loader = DataLoader(consep_test_dataset, batch_size=args.b, shuffle=False, num_workers=8,
    #                                   pin_memory=True)
    else:
        print("the dataset is not supported now!!!")

    return nice_train_loader, nice_test_loader