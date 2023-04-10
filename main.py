import torch 
from torchvision import transforms 
from torchvision.datasets import CIFAR10 
from numpy import ndarray
import numpy as np
from torch.optim import Adam 
from torch.nn import Module , Conv2d , Sequential , BatchNorm2d , ReLU , MaxPool2d , Linear , CrossEntropyLoss , ModuleDict
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt 
from functools import lru_cache

class FC(Module):
    def __init__(self , out_norun , Linear_len) -> None:
        super().__init__()
        self._out_norun , self._Linear_len = out_norun , Linear_len

    @lru_cache
    def _Linear_create(self , _x ):
        
        _x = int(_x)
        
        def _create(_data , _index):
            try:
                return [str(_index) , Linear(int(_data[_index]) , int(_data[_index + 1])) ] 
            except:
                return None  
        
        (_data := list(
                        np.round(
                            np.linspace(
                                    self._out_norun , 
                                    _x , 
                                    self._Linear_len 
                                )
                            )
                        )
        ).reverse()
        
        self.Linear = ModuleDict(
                                [   _s 
                                    for _index in range(len(_data)) 
                                    if (_s:= _create(_data , _index)) is not None
                                ]
                    )
        
        return len(_data) - 1
             
    def forward(self , x : ndarray):
        x = x.reshape(x.size(0) , -1)
        for _ in range(self._Linear_create(
                                        str(x.shape[1]) 
                                           )
                ):
            x = self.Linear[str(_)](
                    x
                )
        
        return x

class Conv(Module):
    def __init__(self , _input , _output , _linear = [
            "Conv2d",
            "BatchNorm2d",
            "ReLU",
            "MaxPool2d",
        ]) -> None:
        super().__init__()
        
        # function running 
        self._create_Linear(
            _input ,
            _output , 
            _linear
        )
    
    def _create_Linear(self , _input , _output , _linear_name ):
        
        def _kernel(x , Df = 16 , _pading = 0):
            for _ in range(x):
                if _ > 1 and x % _ == 0 and _ <= x / Df:
                    return [ _ , _pading ]
                
            if (_kernel_size := _kernel(x+1 , Df=Df , _pading=_pading + 1)) is not None:
                return (_kernel_size[0] , _pading + 1 )
        
        def _create_conv(kernel_size , padding):
            return Conv2d(_input , _output , 
                kernel_size = kernel_size , 
                stride = 1 , 
                padding=padding
            )
    
        def _creae_relu():
            return ReLU()
        
        def _create_maxpool(kernel_size):
            return MaxPool2d(
                kernel_size , 
                kernel_size             
                        )
        
        def _create_BatchNorm():
            return BatchNorm2d(_output)
        
        @lru_cache
        def __run(_sid):

            _model = []
            _kernel_size , _padding = _kernel(_output , 16)
        
            if (_:="Conv2d") in _linear_name:
                _model.append([_ , _create_conv(
                _kernel_size,
                _padding
            )])
        
            if (_:="BatchNorm2d") in _linear_name:
                _model.append([_ , _create_BatchNorm()])
        
            if (_:="ReLU") in _linear_name:
                _model.append([_ , _creae_relu()])
        
            if (_:="MaxPool2d") in _linear_name:
                _model.append([_ , _create_maxpool(
                _kernel_size
            )])
            
            return _model
        
        self._linear , self._linear_name = ModuleDict(__run("a")) , _linear_name
        print(self._linear)
        
    def forward(self , x):
        if x.shape[-1] == 1:
            return x 
        
        for _ in self._linear_name:
            x = self._linear[_](x)
        
        return x 
    
class Model(Module):
    def __init__(self) -> None:
        super().__init__()
        self._Linear = Sequential(
            Conv(3 , 32), 
            Conv(32 , 64), 
            Conv(64 , 96), 
            Conv(96 , 128), 
            Conv(128 , 160), 
            
            FC(10 , 5),
        )
    
    def forward(self , x : ndarray):
        return self._Linear(x)

class DataSets():
    def __init__(self) -> None:
        # run function
        self._run()
        self._split()
        self._lodaer()
    
    def _run(self):
        self._datasets = CIFAR10(
            root="../CIFAR10/" , 
            train= True , 
            transform= transforms.ToTensor() ,
            download= False
        )
    
    def _split(self):
        self._tarin_data , self._test_data = train_test_split(
            self._datasets , 
            test_size=0.4 , 
            shuffle=True , 
            random_state= 0 
        )
        
    def _lodaer(self):
        self._tarin_data , self._test_data = (
            torch.utils.data.DataLoader(
                dataset = self._tarin_data , 
                batch_size = 10 , 
                shuffle = True 
            ) , 
            torch.utils.data.DataLoader(
                dataset =  self._test_data , 
                batch_size = 10 , 
                shuffle = True 
            )
        )

class Train_Model():
    def __init__(self , model : Model , dataset , step ) -> None:
        self._dataset = dataset
        self._model   = model
        self._step    = step 
        self._len_datasets = len(dataset)
        # print(model.parameters())
        self._optimizer = Adam(model.parameters() , lr=0.001)
        self._loss = CrossEntropyLoss()

    def _echo(self , i , j , loss):
        print("Step : [{}/{}] Train : [{}/{}] Loss : [{:.4f}]".format(i , self._step , j , self._len_datasets , loss))

    def _run_step(self , i):
        for j , (_image , _label) in enumerate(self._dataset):
            self._optimizer.zero_grad()
            
            (_loss_value := self._loss(
                _model(_image),
                _label 
            )).backward()
            
            # lr_func = torch.optim.lr_scheduler.StepLR(optimizer , 100 , gamma=0.0005)

            self._echo(i , j , _loss_value.item())
            
            self._optimizer.step()
    
    def run(self):
        for _ in range(self._step):
            self._run_step(_)

class Test_Model():
    def __init__(self , model , datasets) -> None:
        self._model = model 
        self._datasets = datasets 
        
    def _echo(self , corrects , _labale ):
        print('acc : [{}/{}] , test true : {}%' . format(
            corrects.item() , 
            len(_labale) , 
            int((corrects / len(_labale))*100))
        )
    
    def run(self):
        for j , (_data , _labale) in enumerate(self._datasets):
            self._echo(
                torch.sum(torch.argmax(self._model(_data) , dim=1) == _labale),
                _labale
            )
            
            


_ = DataSets()
_model = Model()
# _model = torch.load("model.pt")

_model.train()
Train_Model(model=_model , dataset=_._tarin_data , step= 1).run()

_model.eval()
Test_Model(model=_model , datasets=_._test_data).run()

torch.save(_model ,'model.pt')
# torch.load()
