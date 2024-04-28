import importlib
import torch.utils.data
from data_test_flow.dd_dataset import DDDataset

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader

# def CreateTestDataLoader(opt):
#     data_loader = CustomTestDatasetDataLoader()
#     data_loader.initialize(opt)
#     return data_loader

class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data(self):
        return None

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'
    
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = DDDataset()
        self.dataset.initialize(opt)
        '''
        sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=opt.batch_size, 
            shuffle=False, 
            sampler=sampler)       
        '''
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.shuffle,
            drop_last=True,
            num_workers=int(opt.num_threads))
        
    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
    
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

# class CustomTestDatasetDataLoader(BaseDataLoader):
#     def name(self):
#         return 'CustomDatasetDataLoader'
    
#     def initialize(self, opt):
#         BaseDataLoader.initialize(self, opt)
#         self.dataset = DDDatasetTest()
#         self.dataset.initialize(opt)
#         '''
#         sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
#         self.dataloader = torch.utils.data.DataLoader(
#             self.dataset, 
#             batch_size=opt.batch_size, 
#             shuffle=False, 
#             sampler=sampler)       
#         '''
#         self.dataloader = torch.utils.data.DataLoader(
#             self.dataset,
#             batch_size=opt.batch_size,
#             shuffle=opt.shuffle,
#             drop_last=True,
#             num_workers=int(opt.num_threads))
        
#     def load_data(self):
#         return self

#     def __len__(self):
#         return min(len(self.dataset), self.opt.max_dataset_size)
    
#     def __iter__(self):
#         for i, data in enumerate(self.dataloader):
#             if i * self.opt.batch_size >= self.opt.max_dataset_size:
#                 break
#             yield data
