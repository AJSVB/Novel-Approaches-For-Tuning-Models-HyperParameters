from subprocess import Popen, PIPE
from sys import stdout, stderr
import random
import os


def draw_val(array):
    return array[random.randint(0, len(array)-1)]


if __name__ == '__main__':
    params = {
    "init_lr":  3.4261 #tune.uniform(1e-4, 0.1 ),#,1e-4), #*10
    ,"model_form":0.78484 #tune.uniform(1, 5)#,1e-4), #*10 et 0
    ,"hidden_dim": 7.3172 #,1e-4), #*10 et 0
    ,"bidirectional": 0.82704#,1e-4), #*10 et 0
    , "dropout": 0.052484
    ,"weight_decay": 5.7742
    ,"filter_num": 8.6708
    ,    "f1": 2.2624
    ,    "f2": 5.4772
    , "f3":4.8156
,     "gumbel_temperature": -0.42937
    ,    "gen_hidden_dim": 78.407
    , "dependent":0.50821
    ,     "gen_form": 0.12656
    ,    "gen_bidirectional": 0.24235
    , "selection_lambda":0.035116
    , "selection_prior":0.10878
    , "continuity_lambda":0.064685


 ,    "useless":0.11012

                    
    }
    params_bool = {}


    if not os.path.exists('out'):
        os.makedirs('out')


    temp=params['useless']
    params.pop('useless')
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


    params['results_path'] = '{:.8f}_{:.8f}_{:.8f}_{}_{}_{:.10f}_{:.5f}_{:.1f}_{}_{}.pkl'.format(params['selection_lambda'], params['selection_prior'], params['continuity_lambda'], params['model_form'], params_bool['bidirectional'], params['weight_decay'], params['init_lr'], params['dropout'], params['hidden_dim'], params['filter_num'])
    output_file = 'out/' + params['results_path'].replace('pkl', 'txt')
    params['epochs'] = 15
    '''

    params['results_path'] = '{:.8f}_{:.8f}_{:.8f}_{}.pkl'.format(params['selection_lambda'], params['selection_prior'], params['continuity_lambda'], params['model_form'])
    output_file = 'out/' + params['results_path'].replace('pkl', 'txt')
    params['epochs'] = 30


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

    program = ['python3', 'main_mask_eval.py', '--snapshot={}'.format(snapshot), '--dataset=hotel_annotation']
    Popen(' '.join(program) + ' 2>&1 | tee -a {}'.format(output_file), stdin=PIPE, stdout=stdout, stderr=stderr, shell=True).communicate()

    program = ['python3', 'main_mask_eval.py', '--snapshot={}'.format(snapshot), '--dataset=hotel_annotation_new']
    Popen(' '.join(program) + ' 2>&1 | tee -a {}'.format(output_file), stdin=PIPE, stdout=stdout, stderr=stderr, shell=True).communicate()


    print('Launching a new simulation')
