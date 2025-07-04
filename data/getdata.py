from torchvision import datasets, transforms

def get_dataset(dir, name):

    if name == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.MNIST(dir, train=False, transform=transform_test)

    elif name == 'fashion-mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        train_dataset = datasets.FashionMNIST(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.FashionMNIST(dir, train=False, transform=transform_test)

    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

        train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)


    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        train_dataset = datasets.CIFAR100(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR100(dir, train=False, transform=transform_test)

    return train_dataset, eval_dataset #ignore