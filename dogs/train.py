# dogs.train.py

import dojo
from analysis.plot import plot_logs

if __name__ == '__main__':
    hyperparameters_list = [
        dojo.Hyperparameters(
            coarse_learning_rate_initial=1e-3,
            coarse_learning_rate_min=1e-8,
            coarse_decay_factor=0.1,
            coarse_l2_regularization=1e-2,
            coarse_dropout=0.3,
            coarse_epochs=50,
            fine_learning_rate=1e-4,
            fine_decay_rate=0.5,
            fine_decay_steps=10000,
            fine_l2_regularization=1e-1,
            fine_dropout=0.5,
            fine_epochs=50),
    ]
#        dojo.Hyperparameters(
#            coarse_learning_rate_initial=1e-3,
#            coarse_learning_rate_min=1e-8,
#            coarse_decay_factor=0.1,
#            coarse_epochs=20,
#            fine_learning_rate=1e-4,
#            fine_decay_rate=0.5,
#            fine_decay_steps=10000,
#            fine_epochs=50,
#            l2_regularization=1e-1,
#            dropout=0.9),
#        dojo.Hyperparameters(
#            coarse_learning_rate_initial=1e-2,
#            coarse_learning_rate_min=1e-8,
#            coarse_decay_factor=0.1,
#            coarse_epochs=20,
#            fine_learning_rate=1e-4,
#            fine_decay_rate=0.5,
#            fine_decay_steps=10000,
#            fine_epochs=50,
#            l2_regularization=1e-1,
#            dropout=0.9),
#    ]

    for hyperparameters in hyperparameters_list:
        hyper_dict = hyperparameters.__dict__
        #model_name = 'model' + ''.join(
        #    ['_%s_%s' % (key, hyper_dict[key]) for key in hyper_dict])
         
        dojo.start('testtttt', hyperparameters)
