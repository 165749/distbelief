import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from datetime import datetime
import time
from models.alexnet import AlexNet
from models.mobilenet import MobileNetV2
from models.googlenet import GoogLeNet
from models.inception import Inception3
from models.resnet import Resnet50, Resnet101, Resnet152
from models.vgg import Vgg11, Vgg13, Vgg16, Vgg19
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

import torch.optim as optim
from distbelief.optim import DownpourSGD
from distbelief.optim.downpour_sgd import build_distributed_model
from distbelief.server import ParameterServer
from distbelief.utils.trace import Tracer

_name_to_model = {
    "alexnet": AlexNet,
    "mobilenet": MobileNetV2,
    "googlenet": GoogLeNet,
    "inception3": Inception3,
    "resnet50": Resnet50,
    "resnet101": Resnet101,
    "resnet152": Resnet152,
    "vgg11": Vgg11,
    "vgg13": Vgg13,
    "vgg16": Vgg16,
    "vgg19": Vgg19,
}


def prepare_data(args):
    image_size = 0
    if args.image_size < 0:
        _name_to_image_size = {
            "alexnet": 224 + 3,
            "mobilenet": 224,
            "googlenet": 224,
            "inception3": 299,
            "resnet50": 224,
            "resnet101": 224,
            "resnet152": 224,
            "vgg11": 224,
            "vgg13": 224,
            "vgg16": 224,
            "vgg19": 256,
        }
        if args.model in _name_to_image_size.keys():
            image_size = _name_to_image_size[args.model]
    else:
        image_size = args.image_size

    assert image_size >= 0
    if image_size == 0:
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    else:
        transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    if args.dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    return trainloader, testloader


def main(args, trainloader, testloader):
    logs = []

    tracer = Tracer(cuda=args.cuda)
    root_span = tracer.start_span('worker {}'.format(dist.get_rank()))
    if args.no_distributed:
        net = AlexNet()
    else:
        if args.model in _name_to_model.keys():
            net = build_distributed_model(_name_to_model[args.model], lr=args.lr, tracer=tracer, cuda=args.cuda, ignore_bn=args.ignore_bn, no_overlap=args.no_overlap, all_reduce=args.all_reduce)(num_classes=10)
        else:
            raise Exception("Not implemented yet: {}".format(args.model))

    if args.no_distributed:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
    else:
        optimizer = DownpourSGD(net.parameters(), lr=args.lr, model=net)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, min_lr=1e-3)

    # train
    net.train()
    if args.cuda:
        net = net.cuda()

    steps_per_epoch = len(trainloader)
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print("Training for epoch {}".format(epoch))
        for i, data in enumerate(trainloader, 0):
            print('step {}'.format(i))

            if args.display_time:
                start = time.time()

            # Inform server starting next step (i.e., starting pushing the model to the worker)
            net.step_begin(epoch, i)

            with tracer.start_active_span('prepare_data'):
                inputs, labels = data
                if args.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

            with tracer.start_active_span('zero_grad'):
                # Clear the parameter gradients
                optimizer.zero_grad()

            with tracer.start_active_span('forward'):
                net.init_tracer_span()
                outputs = net(inputs)
                net.finish_tracer_span()

            with tracer.start_active_span('loss'):
                loss = F.cross_entropy(outputs, labels)

            with tracer.start_active_span('backward'):
                net.init_tracer_span()
                loss.backward()
                net.finish_tracer_span()
            optimizer.step()

            if args.display_time:
                end = time.time()
                print('time: {}'.format(end - start))

            if args.num_batches > 0 and epoch * steps_per_epoch + i + 1 >= args.num_batches:
                break
            if args.log_interval > 0 and i % args.log_interval == 0 and i > 0:
                _, predicted = torch.max(outputs, 1)
                if args.cuda:
                    labels = labels.view(-1).cpu().numpy()
                    predicted = predicted.view(-1).cpu().numpy()
                accuracy = accuracy_score(predicted, labels)

                log_obj = {
                    'timestamp': datetime.now(),
                    'iteration': i,
                    'training_loss': loss.item(),
                    'training_accuracy': accuracy,
                }

                log_obj['test_loss'], log_obj['test_accuracy'] = evaluate(net, testloader, args)
                print("Timestamp: {timestamp} | "
                      "Iteration: {iteration:6} | "
                      "Loss: {training_loss:6.4f} | "
                      "Accuracy : {training_accuracy:6.4f} | "
                      "Test Loss: {test_loss:6.4f} | "
                      "Test Accuracy: {test_accuracy:6.4f}".format(**log_obj))
                logs.append(log_obj)

        if args.num_batches > 0 and (epoch + 1) * steps_per_epoch >= args.num_batches:
            break

        if args.evaluate_per_epoch:
            val_loss, val_accuracy = evaluate(net, testloader, args, verbose=True)
            scheduler.step(val_loss)

    # Stop training
    net.stop_training()
    root_span.finish()
    tracer.export_traces("worker{}.json".format(dist.get_rank()))

    df = pd.DataFrame(logs)
    print(df)
    if not os.path.exists('log'):
        os.mkdir('log')
    if args.no_distributed:
        if args.cuda:
            df.to_csv('log/gpu.csv', index_label='index')
        else:
            df.to_csv('log/single.csv', index_label='index')
    else:
        df.to_csv('log/node{}.csv'.format(dist.get_rank()), index_label='index')

    print('Finished Training')


def evaluate(net, testloader, args, verbose=False):
    if args.dataset == 'MNIST':
        classes = [str(i) for i in range(10)]
    else:
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net.eval()

    # Temporarily remove hooks for evaluation
    net.remove_hooks()

    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            test_loss += F.cross_entropy(outputs, labels).item()

    if args.cuda:
        labels = labels.view(-1).cpu().numpy()
        predicted = predicted.view(-1).cpu().numpy()

    test_accuracy = accuracy_score(predicted, labels)
    if verbose:
        print('Loss: {:.3f}'.format(test_loss))
        print('Accuracy: {:.3f}'.format(test_accuracy))
        print(classification_report(predicted, labels, target_names=classes))

    # Restore hooks
    net.register_hooks()

    return test_loss, test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N', help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA for training')
    parser.add_argument('--evaluate-per-epoch', action='store_true', default=False, help='whether to evaluate after each epoch')
    parser.add_argument('--log-interval', type=int, default=-1, metavar='N', help='how often to evaluate and print out')
    parser.add_argument('--no-distributed', action='store_true', default=False, help='whether to use DownpourSGD or normal SGD')
    parser.add_argument('--display-time', action='store_true', default=False, help='whether to displace time of each training step')
    parser.add_argument('--ignore-bn', action='store_true', default=False, help='whether to ignore bn layers when transmitting parameters')
    parser.add_argument('--no-overlap', action='store_true', default=False, help='whether not to overlap communication and computation')
    parser.add_argument('--all-reduce', action='store_true', default=False, help='whether to use all_reduce collective communications')
    parser.add_argument('--worker-id', type=int, default=1, metavar='N', help='rank of the current worker (starting from 1)')
    parser.add_argument('--worker-num', type=int, default=1, metavar='N', help='number of workers in the training')
    parser.add_argument('--server', action='store_true', default=False, help='server node?')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='which dataset to train on')
    parser.add_argument('--model', type=str, default='alexnet', help='which model to train')
    parser.add_argument('--num-batches', type=int, default=100, metavar='N', help='number of batches to train (default: 100)')
    parser.add_argument('--image-size', type=int, default=-1, metavar='N', help='size of images to train (default: -1, which will set the recommended image size for some models)')
    parser.add_argument('--master', type=str, default='localhost', help='ip address of the master (server) node')
    parser.add_argument('--port', type=str, default='2222', help='port on master node to communicate with')
    parser.add_argument('--interface', type=str, default='none', help='Choose network interface to use')
    parser.add_argument('--threads', type=int, default='0', help='How many threads to run')
    parser.add_argument('--sync', action='store_true', default=False, help='Enable synchronous training')
    args = parser.parse_args()
    print(args)

    if not args.no_distributed:
        """ Initialize the distributed environment.
        Server and clients must call this as an entry point.
        """
        os.environ['MASTER_ADDR'] = args.master
        os.environ['MASTER_PORT'] = args.port
        if args.interface != "none":
            os.environ['GLOO_SOCKET_IFNAME'] = args.interface
            print('Set network interface {}'.format(args.interface))

        if args.server:
            if args.model in _name_to_model.keys():
                model = _name_to_model[args.model](num_classes=10)
            else:
                raise Exception("Not implemented yet: {}".format(args.model))
            server = ParameterServer(args=args, model=model, worker_num=args.worker_num)
            server.run()
        else:
            # Prepare data before joining communication group
            trainloader, testloader = prepare_data(args)
            dist.init_process_group('gloo', rank=2 * args.worker_id - 1, world_size=2 * args.worker_num + 1)
            print("worker {} initialized".format(dist.get_rank()))

            # Set number of threads for each worker
            if args.threads > 0:
                torch.set_num_threads(args.threads)
            print('number of threads: {}'.format(torch.get_num_threads()))
            main(args, trainloader, testloader)
        dist.destroy_process_group()
    else:
        main(args)
