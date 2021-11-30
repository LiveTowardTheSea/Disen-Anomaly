#!/usr/bin/env python3
#
# This is the script I use to tune the hyper-parameters automatically.
#
import subprocess
import train
import hyperopt

min_y = 0
min_c = None
max_t = 0


def trial(hyperpm):
    global min_y, min_c, max_t
    # Plz set nbsz manually. Maybe a larger value if you have a large memory.
    cmd = 'python driver/train.py --dataname cora --nbsz 30'
    #cmd = 'CUDA_VISIBLE_DEVICES=5 ' + cmd
    for k in hyperpm:
        v = hyperpm[k]
        cmd += ' --' + k
        if int(v) == v:
            cmd += ' %d' % int(v)
        else:
            cmd += ' %g' % float('%.1e' % float(v))
    try:
        result_str = subprocess.check_output(cmd, shell=True).strip()
        # val, tst = eval(subprocess.check_output(cmd, shell=True))
        val, tst = result_str.split()
        val =float(val)
        tst = float(tst)
    except subprocess.CalledProcessError:
        print('...')
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL}
    print('val=%5.2f%% tst=%5.2f%% @ %s' % (val * 100, tst * 100, cmd))
    score = -val
    if score < min_y:
        min_y, min_c = score, cmd
        max_t = tst  
    return {'loss': score, 'status': hyperopt.STATUS_OK}


space = {'lr': hyperopt.hp.loguniform('lr', -5.5, -3.5),
         'reg': hyperopt.hp.loguniform('reg', -6, -3),
         'clip':hyperopt.hp.uniform('clip', 1.0, 1.5),
         'n_layer': hyperopt.hp.quniform('n_layer', 3, 6, 1),
         'channel': hyperopt.hp.quniform('channel',3,8,1),
         'kdim': hyperopt.hp.quniform('kdim', 2, 32, 2),
         'deck':hyperopt.hp.quniform('deck', 1, 6, 1),
         'dropout': hyperopt.hp.uniform('dropout', 0.1, 0.5),
         'routit': hyperopt.hp.quniform('routit',3,8,1)}
hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=100)
print('>>>>>>>>>> val=%5.2f%% test=%5.2f%% @ %s' % (-min_y * 100, max_t*100, min_c))