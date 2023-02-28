import torch
from torchvision import transforms, datasets
from galaxy_datasets import gz2
from galaxy_datasets.pytorch.galaxy_dataset import GalaxyDataset  # generic Dataset for galaxies

def get_gz2(data_dir, download=False):
    train_catalog, train_label_cols = gz2(
        root=data_dir,
        train=True,
        download=download
    )
    val_catalog, val_label_cols = train_catalog.sample(frac=0.1), train_label_cols
    train_catalog.drop(val_catalog.index)
    test_catalog, test_label_cols = gz2(
        root=data_dir,
        train=False,
        download=download
    )
    train_dataset = GalaxyDataset(
    catalog=test_catalog.sample(len(test_catalog)),  # from gz2(...) above
    label_cols=['has-spiral-arms-gz2_yes', 'has-spiral-arms-gz2_no'],
    target_transform=transforms.Lambda(lambda y: int(y[0] > y[1]))
    )
    val_dataset = GalaxyDataset(
        catalog=val_catalog.sample(len(val_catalog)),  # from gz2(...) above
        label_cols=['has-spiral-arms-gz2_yes', 'has-spiral-arms-gz2_no'],
        target_transform=transforms.Lambda(lambda y: int(y[0] > y[1]))
    )
    test_dataset = GalaxyDataset(
        catalog=train_catalog.sample(len(train_catalog)),  # from gz2(...) above
        label_cols=['has-spiral-arms-gz2_yes', 'has-spiral-arms-gz2_no'],
        target_transform=transforms.Lambda(lambda y: int(y[0] > y[1]))
    )
    dataset_dict = {
        "train": train_dataset,
        "val": val_dataset,
        "test":  test_dataset
    }

    mu = [0.0419, 0.0406, 0.0244]
    std = [0.0885, 0.0742, 0.0629]
    input_size = 224

    # Data augmentation and normalization for training
    # Just normalization for validation
    dataset_dict['train'].transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mu, std)
        ])
    dataset_dict['val'].transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mu, std)
        ])
    dataset_dict['test'].transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mu, std)
        ])

    return dataset_dict

def calc_normalization(dataset_dict):
    for key in dataset_dict.keys():
        dataset_dict[key].transform = transforms.ToTensor()

    mu = 0
    num_samples = 1000
    for i in tqdm(range(num_samples)):
        mu += (1/num_samples) * dataset_dict['train'][i][0].mean(axis=[1,2])
    var = 0
    for i in range(num_samples):
        var += (1/num_samples) * ((dataset_dict['train'][i][0] - mu[:,None,None])**2).mean(axis=[1,2])
    std = np.sqrt(var)

    return mu, std
