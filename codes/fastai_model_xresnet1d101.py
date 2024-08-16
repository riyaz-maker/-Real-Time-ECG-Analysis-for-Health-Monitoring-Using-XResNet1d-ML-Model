from models.timeseries_utils import *

from fastai import *
from fastai.basic_data import *
from fastai.basic_train import *
from fastai.train import *
from fastai.metrics import *
from fastai.torch_core import *
from fastai.callbacks.tracker import SaveModelCallback
from fastai.callback import Callback

from pathlib import Path
from functools import partial
from models.xresnet1d import xresnet1d101
import math
import torch 
import matplotlib
import matplotlib.pyplot as plt

class ClassificationModel(object):
    def __init__(self):
        pass
    def fit(self, X_train, y_train, X_val, y_val):
        pass     
    def predict(self, X, full_sequence=True):
        pass

def evaluate_experiment(y_true, y_pred, thresholds=None):
    results = {}

    if not thresholds is None:
        # binary predictions
        y_pred_binary = apply_thresholds(y_pred, thresholds)
        # PhysioNet/CinC Challenges metrics
        challenge_scores = challenge_metrics(y_true, y_pred_binary, beta1=2, beta2=2)
        results['F_beta_macro'] = challenge_scores['F_beta_macro']
        results['G_beta_macro'] = challenge_scores['G_beta_macro']

    # label based metric
    results['macro_auc'] = roc_auc_score(y_true, y_pred, average='macro')
    
    df_result = pd.DataFrame(results, index=[0])
    return df_result

class metric_func(Callback):
    def __init__(self, func, name="metric_func", ignore_idx=None, one_hot_encode_target=True, argmax_pred=False, softmax_pred=True, flatten_target=True, sigmoid_pred=False,metric_component=None):
        super().__init__()
        self.func = func
        self.ignore_idx = ignore_idx
        self.one_hot_encode_target = one_hot_encode_target
        self.argmax_pred = argmax_pred
        self.softmax_pred = softmax_pred
        self.flatten_target = flatten_target
        self.sigmoid_pred = sigmoid_pred
        self.metric_component = metric_component
        self.name=name

    def on_epoch_begin(self, **kwargs):
        self.y_pred = None
        self.y_true = None
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        y_pred_flat = last_output.view((-1,last_output.size()[-1]))
        
        if(self.flatten_target):
            y_true_flat = last_target.view(-1)
        y_true_flat = last_target

        if(self.argmax_pred is True):
            y_pred_flat = y_pred_flat.argmax(dim=1)
        elif(self.softmax_pred is True):
            y_pred_flat = F.softmax(y_pred_flat, dim=1)
        elif(self.sigmoid_pred is True):
            y_pred_flat = torch.sigmoid(y_pred_flat)
        
        if(self.ignore_idx is not None):
            selected_indices = (y_true_flat!=self.ignore_idx).nonzero().squeeze()
            y_pred_flat = y_pred_flat[selected_indices]
            y_true_flat = y_true_flat[selected_indices]
        
        y_pred_flat = to_np(y_pred_flat)
        y_true_flat = to_np(y_true_flat)

        if(self.one_hot_encode_target is True):
            y_true_flat = one_hot_np(y_true_flat,last_output.size()[-1])

        if(self.y_pred is None):
            self.y_pred = y_pred_flat
            self.y_true = y_true_flat
        else:
            self.y_pred = np.concatenate([self.y_pred, y_pred_flat], axis=0)
            self.y_true = np.concatenate([self.y_true, y_true_flat], axis=0)
    
    def on_epoch_end(self, last_metrics, **kwargs):
        self.metric_complete = self.func(self.y_true, self.y_pred)
        if(self.metric_component is not None):
            return add_metrics(last_metrics, self.metric_complete[self.metric_component])
        else:
            return add_metrics(last_metrics, self.metric_complete)

def fmax_metric(targs,preds):
    return evaluate_experiment(targs,preds)["Fmax"]

def auc_metric(targs,preds):
    return evaluate_experiment(targs,preds)["macro_auc"]

def mse_flat(preds,targs):
    return torch.mean(torch.pow(preds.view(-1)-targs.view(-1),2))

def nll_regression(preds,targs):
    #preds: bs, 2
    #targs: bs, 1
    preds_mean = preds[:,0]
    #warning: output goes through exponential map to ensure positivity
    preds_var = torch.clamp(torch.exp(preds[:,1]),1e-4,1e10)
    #print(to_np(preds_mean)[0],to_np(targs)[0,0],to_np(torch.sqrt(preds_var))[0])
    return torch.mean(torch.log(2*math.pi*preds_var)/2) + torch.mean(torch.pow(preds_mean-targs[:,0],2)/2/preds_var)
    
def nll_regression_init(m):
    assert(isinstance(m, nn.Linear))
    nn.init.normal_(m.weight,0.,0.001)
    nn.init.constant_(m.bias,4)

def lr_find_plot(learner, C:/Users/riyaz/OneDrive/Documents/BioMed/results, filename="lr_find", n_skip=10, n_skip_end=2):
   
    learner.lr_find()
    
    backend_old= matplotlib.get_backend()
    plt.switch_backend('agg')
    plt.ylabel("loss")
    plt.xlabel("learning rate (log scale)")
    losses = [ to_np(x) for x in learner.recorder.losses[n_skip:-(n_skip_end+1)]]

    plt.plot(learner.recorder.lrs[n_skip:-(n_skip_end+1)],losses )

    plt.xscale('log')
    plt.savefig(str(C:/Users/riyaz/OneDrive/Documents/BioMed/results(lr_find+'.png')))
    plt.switch_backend(backend_old)

def losses_plot(learner, C:/Users/riyaz/OneDrive/Documents/BioMed/results, filename="losses", last:int=None):

    backend_old= matplotlib.get_backend()
    plt.switch_backend('agg')
    plt.ylabel("loss")
    plt.xlabel("Batches processed")

    last = ifnone(last,len(learner.recorder.nb_batches))
    l_b = np.sum(learner.recorder.nb_batches[-last:])
    iterations = range_of(learner.recorder.losses)[-l_b:]
    plt.plot(iterations, learner.recorder.losses[-l_b:], label='Train')
    val_iter = learner.recorder.nb_batches[-last:]
    val_iter = np.cumsum(val_iter)+np.sum(learner.recorder.nb_batches[:-last])
    plt.plot(val_iter, learner.recorder.val_losses[-last:], label='Validation')
    plt.legend()

    plt.savefig(str(C:/Users/riyaz/OneDrive/Documents/BioMed/results(losses+'.png')))
    plt.switch_backend(backend_old)

class fastai_model(ClassificationModel):
    def __init__(self,name,n_classes,freq,outputfolder,input_shape,input_size=2.5,input_channels=12,pretrainedfolder=None,chunkify_train=False,chunkify_valid=True,bs=128,ps_head=0.5,lin_ftrs_head=[128],wd=1e-2,epochs=50,lr=1e-2,kernel_size=5,loss="binary_cross_entropy",gradual_unfreezing=True,discriminative_lrs=True,epochs_finetuning=30,aggregate_fn="max",concat_train_val=False):
        super().__init__()
        
        self.name = name
        self.num_classes = n_classes if loss!= "nll_regression" else 2
        self.target_fs = freq
        self.outputfolder = Path(outputfolder)

        self.input_size=int(input_size*self.target_fs)
        self.input_channels=input_channels

        self.chunkify_train=chunkify_train
        self.chunkify_valid=chunkify_valid

        self.chunk_length_train=2*self.input_size
        self.chunk_length_valid=self.input_size

        self.min_chunk_length=self.input_size

        self.stride_length_train=self.input_size
        self.stride_length_valid=self.input_size//2

        self.copies_valid = 0
        
        self.bs=bs
        self.ps_head=ps_head
        self.lin_ftrs_head=lin_ftrs_head
        self.wd=wd
        self.epochs=epochs
        self.lr=lr
        self.kernel_size = kernel_size
        self.loss = loss
        self.input_shape = input_shape
        self.discriminative_lrs = discriminative_lrs
        self.gradual_unfreezing = gradual_unfreezing
        self.epochs_finetuning = epochs_finetuning
        self.aggregate_fn = aggregate_fn
        self.concat_train_val = concat_train_val

    def fit(self, X_train, y_train, X_val, y_val):
        #convert everything to float32
        X_train = [l.astype(np.float32) for l in X_train]
        X_val = [l.astype(np.float32) for l in X_val]
        y_train = [l.astype(np.float32) for l in y_train]
        y_val = [l.astype(np.float32) for l in y_val]

        if(self.concat_train_val):
            X_train += X_val
            y_train += y_val
        
        if(self.pretrainedfolder is None):
            print("Training from the start...")
            learn = self._get_learner(X_train,y_train,X_val,y_val)
            learn.model.apply(weight_init)
            
            #initialization for regression output
            if(self.loss=="nll_regression" or self.loss=="mse"):
                output_layer_new = learn.model.get_output_layer()
                output_layer_new.apply(nll_regression_init)
                learn.model.set_output_layer(output_layer_new)
            
            lr_find_plot(learn, self.outputfolder)    
            learn.fit_one_cycle(self.epochs,self.lr)
            losses_plot(learn, self.outputfolder)
    
    def predict(self, X):
        X = [l.astype(np.float32) for l in X]
        y_dummy = [np.ones(self.num_classes,dtype=np.float32) for _ in range(len(X))]
        
        learn = self._get_learner(X,y_dummy,X,y_dummy)
        learn.load(self.name)
        
        preds,targs=learn.get_preds()
        preds=to_np(preds)
        
        idmap=learn.data.valid_ds.get_id_mapping()

        return aggregate_predictions(preds,idmap=idmap,aggregate_fn = np.mean if self.aggregate_fn=="mean" else np.amax)  
        
    def _get_learner(self, X_train,y_train,X_val,y_val,num_classes=None):
        df_train = pd.DataFrame({"data":range(len(X_train)),"label":y_train})
        df_valid = pd.DataFrame({"data":range(len(X_val)),"label":y_val})
        
        tfms_ptb_xl = [ToTensor()]
                
        ds_train=TimeseriesDatasetCrops(df_train,self.input_size,num_classes=self.num_classes,chunk_length=self.chunk_length_train if self.chunkify_train else 0,min_chunk_length=self.min_chunk_length,stride=self.stride_length_train,transforms=tfms_ptb_xl,annotation=False,col_lbl ="label",npy_data=X_train)
        ds_valid=TimeseriesDatasetCrops(df_valid,self.input_size,num_classes=self.num_classes,chunk_length=self.chunk_length_valid if self.chunkify_valid else 0,min_chunk_length=self.min_chunk_length,stride=self.stride_length_valid,transforms=tfms_ptb_xl,annotation=False,col_lbl ="label",npy_data=X_val)
    
        db = DataBunch.create(ds_train,ds_valid,bs=self.bs)

        if(self.loss == "binary_cross_entropy"):
            loss = F.binary_cross_entropy_with_logits
        elif(self.loss == "cross_entropy"):
            loss = F.cross_entropy
        elif(self.loss == "mse"):
            loss = mse_flat
        elif(self.loss == "nll_regression"):
            loss = nll_regression    
        else:
            print("loss not found")
            assert(True)   
               
        self.input_channels = self.input_shape[-1]
        metrics = []

        print("model:",self.name)
        num_classes = self.num_classes if num_classes is None else num_classes
        if(self.name.startswith("fastai_resnet1d101")):
            model = resnet1d101(num_classes=num_classes,input_channels=self.input_channels,inplanes=128,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        else:
            print("Model not found.")
            assert(True)
            
        learn = Learner(db,model, loss_func=loss, metrics=metrics,wd=self.wd,path=self.outputfolder)
        
        if(self.name.startswith("fastai_lstm") or self.name.startswith("fastai_gru")):
            learn.callback_fns.append(partial(GradientClipping, clip=0.25))
            
        return learn
    