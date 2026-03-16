from torch.utils.data import DataLoader

from data_provider.data_loader import PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader

data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
}


def data_provider(args, flag):
    if args.task_name != 'anomaly_detection':
        raise ValueError('Only anomaly_detection is supported in this simplified repository.')
    if args.data not in data_dict:
        raise ValueError(f'Unsupported anomaly dataset: {args.data}. Supported: {list(data_dict.keys())}')

    Data = data_dict[args.data]
    data_set = Data(
        args=args,
        root_path=args.root_path,
        win_size=args.seq_len,
        flag=flag,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False if flag in ['test', 'TEST'] else True,
        num_workers=args.num_workers,
        drop_last=False)
    return data_set, data_loader
