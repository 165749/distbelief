import os
import logging 
import argparse
import csv
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from datetime import datetime
import time
from models import LeNet, AlexNet
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

import torch.optim as optim
from distbelief.optim import DownpourSGD
from distbelief.optim.downpour_sgd import build_distributed_model
from distbelief.server import ParameterServer
from distbelief.utils.tracer import tracer, numbers_to_trace_context, trace_context_to_numbers
from opentracing.propagation import Format

def get_dataset(args, transform):
    """
    :param dataset_name:
    :param transform:
    :param batch_size:
    :return: iterators for the dataset
    """
    if args.dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    return trainloader, testloader

def main(args):

    logs = []

    transform = transforms.Compose([
                # transforms.Resize(64),  # For torchvision.models.alexnet
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    trainloader, testloader = get_dataset(args, transform)

    if args.no_distributed:
        net = AlexNet()
    else:
        net = build_distributed_model(AlexNet, lr=args.lr, cuda=args.cuda)()
        # net = build_distributed_model(torchvision.models.ResNet, lr=args.lr, cuda=args.cuda)(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=10)

    if args.no_distributed:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
    else:
        optimizer = DownpourSGD(net.parameters(), lr=args.lr, model=net)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, min_lr=1e-3)

    # train
    net.train()
    if args.cuda:
        net = net.cuda()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print("Training for epoch {}".format(epoch))
        with tracer.start_active_span('Epoch {}'.format(epoch)):
            for i, data in enumerate(trainloader, 0):
                with tracer.start_active_span('Step {}'.format(i)):
                    # get the inputs
                    inputs, labels = data

                    if args.cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    with tracer.start_active_span('Forward') as scope:
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        net.init_tracer_span(scope.span)
                        # forward + backward + optimize
                        outputs = net(inputs)
                        net.finish_tracer_span()

                    with tracer.start_active_span('loss'):
                        loss = F.cross_entropy(outputs, labels)
                    with tracer.start_active_span('Backward') as scope:
                        net.reset_senders()
                        net.init_tracer_span(scope.span)
                        loss.backward()
                        net.finish_tracer_span()
                    optimizer.step()

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

                    if i % args.log_interval == 0 and i > 0:    # print every n mini-batches
                        log_obj['test_loss'], log_obj['test_accuracy'] = evaluate(net, testloader, args)
                        print("Timestamp: {timestamp} | "
                              "Iteration: {iteration:6} | "
                              "Loss: {training_loss:6.4f} | "
                              "Accuracy : {training_accuracy:6.4f} | "
                              "Test Loss: {test_loss:6.4f} | "
                              "Test Accuracy: {test_accuracy:6.4f}".format(**log_obj))
                    logs.append(log_obj)

        val_loss, val_accuracy = evaluate(net, testloader, args, verbose=True)
        scheduler.step(val_loss)

    # Stop training
    optimizer.stop()

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

def init_server():
    model = AlexNet()
    # model = torchvision.models.resnet50(num_classes=10)
    server = ParameterServer(model=model, active_worker=args.world_size-1)
    server.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N', help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA for training')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N', help='how often to evaluate and print out')
    parser.add_argument('--no-distributed', action='store_true', default=False, help='whether to use DownpourSGD or normal SGD')
    parser.add_argument('--rank', type=int, metavar='N', help='rank of current process (0 is server, 1+ is training node)')
    parser.add_argument('--world-size', type=int, default=3, metavar='N', help='size of the world')
    parser.add_argument('--server', action='store_true', default=False, help='server node?')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='which dataset to train on')
    parser.add_argument('--master', type=str, default='localhost', help='ip address of the master (server) node')
    parser.add_argument('--port', type=str, default='29500', help='port on master node to communicate with')
    args = parser.parse_args()
    print(args)

    if not args.no_distributed:
        """ Initialize the distributed environment.
        Server and clients must call this as an entry point.
        """
        os.environ['MASTER_ADDR'] = args.master
        os.environ['MASTER_PORT'] = args.port
        dist.init_process_group('gloo', rank=args.rank, world_size=args.world_size)

        if args.server:
            with tracer.start_active_span('distbelief') as scope:
                span = tracer.active_span
                context = {}
                tracer.inject(span, Format.TEXT_MAP, context)
                numbers = trace_context_to_numbers(context)
                tensor = torch.from_numpy(np.array(numbers, dtype=np.int64))
                for idx in range(1, args.world_size):
                    dist.send(tensor=tensor, dst=idx)  # send to every worker
                with tracer.start_active_span('server'):
                    init_server()
        else:
            tensor = torch.zeros(4, dtype=torch.int64)  # TODO (zhuojin): Remove hard-code
            dist.recv(tensor=tensor)
            context = numbers_to_trace_context(tensor.tolist())
            span_ctx = tracer.extract(Format.TEXT_MAP, context)
            with tracer.start_active_span('worker {}'.format(args.rank), child_of=span_ctx):
                main(args)
        dist.destroy_process_group()
        # Wait for trace collection
        time.sleep(2)
    else:
        main(args)
