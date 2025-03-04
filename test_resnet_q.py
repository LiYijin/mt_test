import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', '-M', help='Specify model path of quanted resnet', required=True)
parser.add_argument('--dataset', '-D', help='Specify cifar-100 dataset path', required=True)
args = parser.parse_args()



CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    transforms.CenterCrop(32),
])

# load dataset
infer_dataset = CIFAR100(root=args.dataset, train=False, download=True, transform=transform)
infer_dataset = DataLoader(dataset=infer_dataset, batch_size=24, shuffle=False)
resnet_test = ort.InferenceSession(args.model, providers=['MUSAExecutionProvider'])

def evaluate( val_loader):
    top1_correct = 0
    top5_correct = 0
    total = 0
    total_time = 0.0
    batch_cnt = 0
    log_iter = 100
    for i, (inputs, targets) in enumerate(val_loader):
        if inputs.shape[0] < 24:
            print("expand", inputs.shape)
            last_image = inputs[-1].unsqueeze(0)
            inputs = torch.cat((inputs, last_image.repeat(8, 1, 1, 1)), dim=0)
            last_tag = targets[-1].unsqueeze(0)
            targets = torch.cat((targets, last_tag.repeat(8)), dim=0)
        np_dtype = np.float32
        start_time = time.time()
        outputs = resnet_test.run(['output'], {'input': np.array(inputs, dtype=np_dtype)})[0]
        total_time += time.time() - start_time
        outputs = torch.from_numpy(outputs.astype(np.float32))
        
        _, predicted = outputs.topk(5, 1, True, True)
        predicted = predicted.t()
        
        total += targets.size(0)
        correct = predicted.eq(targets.view(1, -1).expand_as(predicted))
        
        top1_correct += correct[:1].view(-1).float().sum(0, keepdim=True)
        top5_correct += correct[:5].reshape(-1).float().sum(0, keepdim=True)

        batch_cnt += 1
        # if i % log_iter == 0:
        #     top1_accuracy = 100. * top1_correct / total
        #     top5_accuracy = 100. * top5_correct / total
        #     print(f'Top-1 accuracy: {top1_accuracy.item():.2f}%')
        #     print(f'Top-5 accuracy: {top5_accuracy.item():.2f}%')
    top1_accuracy = 100. * top1_correct / total
    top5_accuracy = 100. * top5_correct / total

    print("Batch Size: {}\nTotal Time: {:.2f} Seconds\nLatency: {:.2f} ms / batch".format(batch_cnt, total_time, 1000.0 * total_time / batch_cnt))

    return top1_accuracy.item(), top5_accuracy.item()

# Perform evaluation
top1_acc, top5_acc = evaluate(infer_dataset)
print(f'Top-1 accuracy: {top1_acc:.2f}%')
print(f'Top-5 accuracy: {top5_acc:.2f}%')
