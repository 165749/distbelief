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
from models.resnet import Resnet50
from models.inception import Inception3
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

import torch.optim as optim
from distbelief.optim import DownpourSGD
from distbelief.optim.downpour_sgd import build_distributed_model
from distbelief.server import ParameterServer
from distbelief.utils.trace import Tracer


def prepare_data(args):
    if args.model == "alexnet":
        transform = transforms.Compose([
                    transforms.Resize(224 + 3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    elif args.model == "inception3":
        transform = transforms.Compose([
                    transforms.Resize(299),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    elif args.model == "resnet50":
        transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    else:
        transform = transforms.Compose([
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

    tracer = Tracer()
    root_span = tracer.start_span('worker {}'.format(dist.get_rank()))
    if args.no_distributed:
        net = AlexNet()
    else:
        if args.model == "alexnet":
            net = build_distributed_model(AlexNet, lr=args.lr, tracer=tracer, cuda=args.cuda, no_overlap=args.no_overlap)()
        elif args.model == "resnet50":
            # net = build_distributed_model(torchvision.models.ResNet, lr=args.lr, tracer=tracer, cuda=args.cuda, ignore_bn=args.ignore_bn, no_overlap=args.no_overlap)(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=10)
            net = build_distributed_model(Resnet50, lr=args.lr, tracer=tracer, cuda=args.cuda, ignore_bn=args.ignore_bn, no_overlap=args.no_overlap)(num_classes=10)
        elif args.model == "inception3":
            # net = build_distributed_model(torchvision.models.Inception3, lr=args.lr, tracer=tracer, cuda=args.cuda, ignore_bn=args.ignore_bn, no_overlap=args.no_overlap)(aux_logits=False, num_classes=10)
            net = build_distributed_model(Inception3, lr=args.lr, tracer=tracer, cuda=args.cuda, ignore_bn=args.ignore_bn, no_overlap=args.no_overlap)(aux_logits=False, num_classes=10)
        else:
            raise Exception("Not implemented yet: {}".format(args.model))

    if args.no_distributed:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
    else:
        optimizer = DownpourSGD(net.parameters(), lr=args.lr, model=net, no_overlap=args.no_overlap)
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
            with tracer.start_active_span('epoch {} step {}'.format(epoch, i)):
                if args.display_time:
                    start = time.time()

                # Inform server starting next step (i.e., starting pushing the model to the worker)
                net.step_begin()

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
                    net.reset_senders()
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

    root_span.finish()
    tracer.export_traces("worker{}.json".format(dist.get_rank()))
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
    parser.add_argument('--worker-id', type=int, default=1, metavar='N', help='rank of the current worker (starting from 1)')
    parser.add_argument('--worker-num', type=int, default=1, metavar='N', help='number of workers in the training')
    parser.add_argument('--server', action='store_true', default=False, help='server node?')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='which dataset to train on')
    parser.add_argument('--model', type=str, default='alexnet', help='which model to train')
    parser.add_argument('--num-batches', type=int, default=100, metavar='N', help='number of batches to train (default: 100)')
    parser.add_argument('--master', type=str, default='localhost', help='ip address of the master (server) node')
    parser.add_argument('--port', type=str, default='2222', help='port on master node to communicate with')
    parser.add_argument('--interface', type=str, default='none', help='Choose network interface to use')
    parser.add_argument('--threads', type=int, default='0', help='How many threads to run')
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
            if args.model == "alexnet":
                model = AlexNet()
            elif args.model == "resnet50":
                # model = torchvision.models.resnet50(num_classes=10)
                model = Resnet50(num_classes=10)
            elif args.model == "inception3":
                # model = torchvision.models.Inception3(aux_logits=False, num_classes=10)
                model = Inception3(aux_logits=False, num_classes=10)
            else:
                raise Exception("Not implemented yet: {}".format(args.model))
            if args.ignore_bn:
                bn_names = [name for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)]
                parameters_with_names = [(name, para) for name, para in model.named_parameters() if name.rsplit('.', maxsplit=1)[0] not in bn_names]
            else:
                # TODO (zhuojin): Verify name conflict
                parameters_with_names = [(name, para) for name, para in model.named_parameters()]
            server = ParameterServer(parameters_with_names=parameters_with_names, worker_num=args.worker_num)
            server.run()
        else:
            # Prepare data before joining communication group
            trainloader, testloader = prepare_data(args)
            dist.init_process_group('gloo', rank=2 * args.worker_id - 1, world_size=2 * args.worker_num + 1)
            print("worker {} initialized".format(dist.get_rank()))

            dist.recv(tensor=torch.zeros(1))
            # Set number of threads for each worker
            if args.threads > 0:
                torch.set_num_threads(args.threads)
            print('number of threads: {}'.format(torch.get_num_threads()))
            main(args, trainloader, testloader)
        dist.destroy_process_group()
    else:
        main(args)
