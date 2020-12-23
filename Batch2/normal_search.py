from subprocess import Popen, PIPE
from sys import stdout, stderr
import random
import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
import torchvision.transforms as transforms
import os
import torchvision.transforms as T
import os
import glob
import sys
from ray.tune.suggest.nevergrad import NevergradSearch
import nevergrad as ng
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler   
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.suggest import ConcurrencyLimiter 
from ray import tune

from ray.tune.logger import *
class TestLogger(tune.logger.Logger):
    def _init(self):
        progress_file = os.path.join('', "randomsearchMTM"+str(2)+".csv")
        self._continuing = os.path.exists(progress_file)
        self._file = open(progress_file, "a")   
        self._csv_out = None

    def on_result(self, result):
        tmp = result.copy()
        #if "done" in tmp:
         #   if(tmp["done"] != True):

        if "config" in tmp:
            del tmp["config"]
        result = flatten_dict(tmp, delimiter="/")
        if self._csv_out is None:
            self._csv_out = csv.DictWriter(self._file, result.keys())
            if not self._continuing:
                self._csv_out.writeheader()
        self._csv_out.writerow(
            {k: v
             for k, v in result.items() if k in self._csv_out.fieldnames})
        self._file.flush()




def draw_val(array):
    return array[random.randint(0, len(array)-1)]

def train_MTM(params):
    temp=params['useless']
    params.pop('useless')
    os.chdir(savedPath)
    params_bool = {}
    params['init_lr'] = 10**-params['init_lr']  
    params['hidden_dim'] = round(2**params['hidden_dim'])
    params['gen_hidden_dim'] = round(params['gen_hidden_dim'])
    if(params['model_form']<1/3):
        x='cnn'
    elif(params['model_form']<2/3):
        x='lstm'
    else:
        x='gru'
    #x='cnn'   #TOCHANGE
    params['model_form'] = x

    params_bool['bidirectional']= params['bidirectional']<0.5
    params_bool['dependent']= params['dependent']<0.5
    #params_bool['gen_bidirectional']= params['gen_bidirectional']<0.5
    params.pop('dependent')
    params.pop('gen_bidirectional')

    params['weight_decay'] = 10**-params['weight_decay']

    params['filter_num'] = round(2**params['filter_num'])
    params['filters'] = str(round(params['f1']))+','+str(round(params['f2']))+','+str(round(params['f3']))
    params.pop('f1')
    params.pop('f2')
    params.pop('f3')
    params.pop('bidirectional')
    params['gumbel_temperature'] = 10**params['gumbel_temperature'] #0.1 to 10


   #        params['dependent']= ''
    #else:
    #    params.pop('dependent')

    if(params['gen_form']<0.5):
        x='lstm'
    else:
        x='gru'
    params['gen_form'] = x
    #if(params['gen_bidirectional']<0.5):
    #        params['gen_bidirectional']= ''
    #else:
    #    params.pop('gen_bidirectional')




    '''
    if(params['dependent']<0.5):
                params['dependent']='True'
    else:
        params['dependent'] = 'False'
    params['gen_bidirectional'] = params['gen_bidirectional']<0.5
    params['dependent'] = params['dependent']<0.5
    '''


    '''
    params['model_form'] = draw_val(['cnn', 'lstm'])
    params['weight_decay'] = draw_val([0, 1e-6, 1e-8, 1e-10])
    params['init_lr'] = draw_val([0.001, 0.0005, 0.00075])
    params['dropout'] = draw_val([0, 0.1, 0.2]) 
    params['hidden_dim'] = draw_val([50, 100, 200])
    params['filter_num'] = 50

    params_bool['dependent'] = draw_val([True]) 
    params_bool['gen_bidirectional'] = False

    params_bool['bidirectional'] = False
    if params['model_form'] == 'lstm':
        params_bool['bidirectional'] = draw_val([True, False])
    else:
        params['filter_num'] = draw_val([50, 100, 200])


    params['results_path'] = '{:.8f}_{:.8f}_{:.8f}_{}_{}_{:.10f}_{:.5f}_{:.1f}_{}_{}.pkl'.format(params['selection_lambda'], params['selection_prior'], params['continuity_lambda'], params['model_form'], params_b
ool['bidirectional'], params['weight_decay'], params['init_lr'], params['dropout'], params['hidden_dim'], params['filter_num'])
    output_file = 'out/' + params['results_path'].replace('pkl', 'txt')
    params['epochs'] = 15
    '''


    params['results_path'] = '{:.8f}_{:.8f}_{:.8f}_{}.pkl'.format(params['selection_lambda'], params['selection_prior'], params['continuity_lambda'], params['model_form'])
    output_file = 'out/' + params['results_path'].replace('pkl', 'txt')
   



    params['epochs'] = 1 #2
    params['batch_size'] = 16 #
    params['gpu']=1

    if os.path.exists(output_file):
        return

    random_id = random.randint(1, 100000000)
    params['results_path'] = '{}_'.format(random_id) + params['results_path']

    params_command_line = ['--{}={}'.format(k,v) for k,v in sorted(params.items(), key=lambda x:x[0])]
    params_command_line += ['--{}'.format(k) for k, v in sorted(params_bool.items(), key=lambda x:x[0]) if v]
    program = ['python3', 'main.py']
    command_line = program + params_command_line
    Popen(' '.join(command_line) + ' 2>&1 | tee {}'.format(output_file), stdin=PIPE, stdout=stdout, stderr=stderr, shell=True).communicate()

    print('Training Done!')
    #output_file = '/Users/diego/Github/MultiAspectTagging-snapshot/WeaklysMAM/0.05000000_0.10000000_0.06000000_cnncopy.txt'

    # Compute Mask
    with open(output_file, 'r', encoding='utf-8') as fp:
        content = [line.strip() for line in fp] 
        content = [line for line in content if len(line) > 0]
        content = content[-1]
        path = content.split()[-1]  #snapshot/8115325_0.05000000_0.10000000_0.06000000_cnn_test.pkl
        snapshot = path[:path.find('.')] + '.pt'
        test_pkl = path

    #snapshot = '/Users/diego/Github/MultiAspectTagging-snapshot/WeaklysMAM/8115325_0.pt'
    #test_pkl = '/Users/diego/Github/MultiAspectTagging-snapshot/WeaklysMAM/8115325_0.05000000_0.10000000_0.06000000_cnn_test.pkl'

    print(snapshot)
    print(test_pkl)


    with open('test3.csv','w') as outfile: #same with "w" or "a" as opening mode

        program = ['python3', 'main_mask_eval.py', '--snapshot={}'.format(snapshot), '--dataset=hotel_annotation']
        Popen(' '.join(program) + ' 2>&1 | tee -a {}'.format(output_file), stdin=PIPE, stdout=outfile, stderr=stderr, shell=True).communicate()


    #a,b = parse_file(output_file,1)
    #print(a)
    #print(b)

    loss=-1
    with open('test3.csv', 'r', encoding='utf-8') as fp:
        for line in fp:
            print("priti", line)
            if(line[0:9]=='f1_macro:'):
                print(line)
                print(line[11:-2])
                loss = float(line[11:-2])

            if 'f1_macro:' in line:
                print(line)
                print(line[-8:-2])
                loss = float(line[11:-2])

    #program = ['python3', 'main_mask_eval.py', '--snapshot={}'.format(snapshot), '--dataset=hotel_annotation_new']
    #Popen(' '.join(program) + ' 2>&1 | tee -a {}'.format(output_file), stdin=PIPE, stdout=stdout, stderr=stderr, shell=True).communicate()
    #print("3333")



    tune.report(score=loss)

    params['useless'] = temp

    dir_name = "snapshot"
    test = os.listdir(dir_name)

    for item in test:
        if os.stat(os.path.join(path,f)).st_mtime < now - 7 * 8640 :

            os.remove(os.path.join(dir_name, item))

    dir_name = "out"
    test = os.listdir(dir_name)

    for item in test:
        if  os.stat(os.path.join(path,f)).st_mtime < now - 7 * 8640:
            os.remove(os.path.join(dir_name, item))

    # Compute NPMI
    #  program = ['python3', 'rationales_prob_hist.py', test_pkl]
    #  Popen(' '.join(program) + ' 2>&1 | tee -a {}'.format(output_file), stdin=PIPE, stdout=stdout, stderr=stderr, shell=True).communicate()

    #  program = ['python3', 'topics/compute_npmi.py', os.path.realpath('.') + '/topics/topics_' + os.path.basename(test_pkl)[:-3] + 'txt']
    # Popen(' '.join(program) + ' 2>&1 | tee -a {}'.format(output_file), stdin=PIPE, stdout=stdout, stderr=stderr, shell=True).communicate()
    return
    print('Launching a new simulation')



if __name__ == '__main__':
    savedPath = os.getcwd()


    params = {}
    params_bool = {}
    SELECT_LAMBDA_RANGE = [0.02, 0.03, 0.04, 0.05]
    SELECT_PRIOR_RANGE = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    SELECT_CONT_RANGE = [0.04, 0.06, 0.08, 0.10]

    if not os.path.exists('out'):
        os.makedirs('out')


    name="randomMTM1"

    algo = NevergradSearch(
        optimizer=ng.optimizers.RandomSearch,
        metric="score",     mode="max")  

    algo = ConcurrencyLimiter(algo, max_concurrent=1)
    scheduler = ASHAScheduler()
    from ray import tune
    analysis = tune.run(
    train_MTM,
    name=name,
    scheduler=scheduler,
    reuse_actors=False,
    search_alg=algo,
    verbose=2,
    checkpoint_at_end=False,
    num_samples=32, # 64
    # export_formats=[ExportFormat.MODEL],
    config= {
    "init_lr":  tune.uniform(1, 8) #tune.uniform(1e-4, 0.1 ),#,1e-4), #*10
    ,"model_form":tune.uniform(0, 1)#tune.uniform(1, 5)#,1e-4), #*10 et 0
    ,"hidden_dim": tune.uniform(6, 12)#,1e-4), #*10 et 0
    ,"bidirectional": tune.uniform(0, 1)#,1e-4), #*10 et 0
    , "dropout": tune.uniform(0,0.7)
    ,"weight_decay": tune.uniform(3, 10)
    ,"filter_num": tune.uniform(4, 10)
    ,    "f1": tune.uniform(1, 5)
    ,    "f2": tune.uniform(2, 6)
    , "f3":tune.uniform(3,7)
,     "gumbel_temperature": tune.uniform(-1, 1)
    ,    "gen_hidden_dim": tune.uniform(50, 200)
    , "dependent":tune.uniform(0,1)
    ,     "gen_form": tune.uniform(0, 1)
    ,    "gen_bidirectional": tune.uniform(0, 1)
    , "selection_lambda":tune.uniform(0.01,0.05)
    , "selection_prior":tune.uniform(0.04,0.15)
    , "continuity_lambda":tune.uniform(0.04,0.1)


 ,    "useless":tune.uniform(0,0.2)

                    
        },          stop={"training_iteration": 1}

            ,        metric="score",
        mode="max" ,resources_per_trial={'gpu': 1, 'cpu': 24}

            ,    loggers=[TestLogger]
    )






