import torch
import torchvision

from MultiFL.fed_server import CloudServer

def print_device_info():
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = False
        print(torch.version.cuda)
        print(torch.backends.cudnn.version())
        print(torch.cuda.get_device_name(0))

def main():
    print_device_info()

    cloud = CloudServer()

    cloud.SinkModelToEdges()
    for edge in cloud.edges:
        edge.SinkModelToClients()
    cloud.run()

if __name__ == '__main__':
    main()
