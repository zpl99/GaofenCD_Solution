import os
import torch
import sys


class CheckPointsSaver():
    def __init__(self, module):
        self.logs = {}
        self.thisValLog = sys.maxsize
        self.thisCheckPointsPath = ''
        self.module = module  # name of checkpoints

    def push(self, epoch, model, optimizer, configs, valLog):
        check_points_path = os.path.join(configs["save_model_path"], 'checkpoints')
        check_points_path = os.path.join(check_points_path, self.module + 'epoch%d.pth' % epoch)
        if self.thisValLog > valLog:
            if epoch != 0:
                os.remove(self.thisCheckPointsPath)
                print("remove %s successful!" % self.thisCheckPointsPath)
            self.save_checkpoint(epoch, model, optimizer, configs, self.module)
            self.thisValLog = valLog
            self.thisCheckPointsPath = check_points_path

    def get(self, epoch_index):
        return self.logs.get(epoch_index)

    def save_checkpoint(self, epoch, model, optimizer, configs, module=""):
        save_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        path = os.path.join(configs["save_model_path"], 'checkpoints')
        print("saving........")
        if not os.path.exists(path):
            os.makedirs(path)
            print(path + "dir maked")
        save_name = os.path.join(path, module + 'epoch%d.pth' % epoch)
        torch.save(save_state, save_name)
        print('Saved model to %s' % save_name)
