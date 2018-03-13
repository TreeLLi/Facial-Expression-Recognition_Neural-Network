import numpy as np
import os

from numpy.random import uniform, randint
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data, get_FER2013_data_from_binary
from src.fcnet import FullyConnectedNet

from src.evaluator import draw_loss_acc

COARSE_RANGE = {"lower" : 800, "upper" : 1800}

MAX_ITER = 10

LEARNING_RATE = 0.00334
MOMENTUM = 0.9
LEARNING_DECAY = 0.90


BATCH_SIZE = 100
EPOCH_NUM = 50

CHECKOUT = False

DROPOUT = 0.23524
REGULAR = 0

TRAIN_NUM = None
VALID_NUM = 2000
TEST_NUM = None
CLASS_NUM = 7

def stage_optim_lr(fcnn, data):
    # 1st stage: search in coarse range
    lower = COARSE_RANGE["lower"]
    upper = COARSE_RANGE["upper"]
    
    lrs = []
    match_values = []
    accs = []
    
    for it in range(MAX_ITER):
        lr = 10**uniform(lower, upper)
        lrs.append(lr)
        solver = Solver(fcnn,
                        data,
                        update_rule='sgd_momentum',
                        optim_config={"learning_rate":lr, "momentum":MOMENTUM},
                        lr_decay=LEARNING_DECAY,
                        print_every=100,
                        batch_size=BATCH_SIZE,
                        checkpoint_name="checkpoints/test" if CHECKOUT else None,
                        num_epochs=EPOCH_NUM,
                        tune_lr=True)
        solver.train()
        match_values.append(lr_update_match(solver.updates))
        accs.append(solver.best_val_acc)
        loss = [ x for x in solver.loss_history if not x > 2.0]
        t_acc = solver.train_acc_history
        v_acc = solver.val_acc_history
        draw_loss_acc(loss, t_acc, v_acc, "lr_{}_{}/{}-{:.5f}".format(lower, upper, it+1, lr))
        fcnn.reset()

    match_values = np.asarray(match_values)
        
    path = "params/"
    filename = "lr_{}_{}.txt".format(lower, upper)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+filename, 'w') as file:
        for line, lr in enumerate(lrs):
            stat = "{}. lr: {:.5f} - match ratio: {:.4f} - accuracy: {:.4f}".format(line+1,
                                                                        lr,
                                                                        match_values[line],
                                                                        accs[line])
            file.write(stat + "\n")

        best_lr_idx = np.argmax(match_values, axis=0)
        best_lr = lrs[best_lr_idx]
        best_lr_acc = accs[best_lr_idx]

        file.write("Best match lr - {}: {:.5f} with accuracy {:.4f}\n".format(best_lr_idx, best_lr, best_lr_acc))

        best_lr_idx = np.argmax(np.asarray(accs), axis=0)
        best_lr = lrs[best_lr_idx]
        best_lr_acc = accs[best_lr_idx]

        file.write("Best accur lr - {}: {:.5f} with accuracy {:.4f}\n".format(best_lr_idx, best_lr, best_lr_acc))


        
def lr_update_match(updates):
    count = 0.0
    for update in updates:
        if update < 0.0015 and update > 0.0005:
            count += 1

    return count / len(updates)

def stage_optim_reg(fcnn, data):
    # 1st stage: search in coarse range
    lower = COARSE_RANGE["lower"]
    upper = COARSE_RANGE["upper"]
    
    regs = []
    accs = []
    
    for it in range(MAX_ITER):
        reg = 10**uniform(lower, upper)
        print ("Experiment {} for reg: {}".format(it+1, reg))
        regs.append(reg)
        fcnn.reg = reg
        solver = Solver(fcnn,
                        data,
                        update_rule='sgd_momentum',
                        optim_config={"learning_rate":LEARNING_RATE, "momentum":MOMENTUM},
                        lr_decay=LEARNING_DECAY,
                        print_every=100,
                        batch_size=BATCH_SIZE,
                        checkpoint_name="checkpoints/test" if CHECKOUT else None,
                        num_epochs=EPOCH_NUM)
        solver.train()
        accs.append(solver.best_val_acc)
        loss = solver.loss_history
        t_acc = solver.train_acc_history
        v_acc = solver.val_acc_history
        draw_loss_acc(loss, t_acc, v_acc, "reg_{}_{}/{}-{:.5f}".format(lower, upper, it+1, reg))
        fcnn.reset()

    path = "params/"
    filename = "reg_{}_{}.txt".format(lower, upper)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+filename, 'w') as file:
        for line, reg in enumerate(regs):
            stat = "{}. reg: {:.5f} - accuracy: {:.4f}".format(line+1,
                                                               reg,
                                                               accs[line])
            file.write(stat + "\n")

        best_reg_idx = np.argmax(np.asarray(accs), axis=0)
        best_reg = regs[best_reg_idx]
        best_reg_acc = accs[best_reg_idx]

        file.write("Best accur reg - {}: {:.5f} with accuracy {:.4f}\n".format(best_reg_idx+1, best_reg, best_reg_acc))

        
def stage_optim_drop(fcnn, data):
    # 1st stage: search in coarse range
    lower = COARSE_RANGE["lower"]
    upper = COARSE_RANGE["upper"]
    
    drops = []
    accs = []
    
    for it in range(MAX_ITER):
        drop = uniform(lower, upper)
        print ("Experiment {} for drop: {}".format(it+1, drop))
        drops.append(drop)
        fcnn.dropout_params['p'] = drop
        solver = Solver(fcnn,
                        data,
                        update_rule='sgd_momentum',
                        optim_config={"learning_rate":LEARNING_RATE, "momentum":MOMENTUM},
                        lr_decay=LEARNING_DECAY,
                        print_every=100,
                        batch_size=BATCH_SIZE,
                        checkpoint_name="checkpoints/test" if CHECKOUT else None,
                        num_epochs=EPOCH_NUM)
        solver.train()
        accs.append(solver.best_val_acc)
        loss = solver.loss_history
        t_acc = solver.train_acc_history
        v_acc = solver.val_acc_history
        draw_loss_acc(loss, t_acc, v_acc, "drop_{}_{}/{}-{:.5f}".format(lower, upper, it+1, drop))
        fcnn.reset()

    path = "params/"
    filename = "drop_{}_{}.txt".format(lower, upper)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+filename, 'w') as file:
        for line, drop in enumerate(drops):
            stat = "{}. drop: {:.5f} - accuracy: {:.4f}".format(line+1,
                                                               drop,
                                                               accs[line])
            file.write(stat + "\n")

        best_drop_idx = np.argmax(np.asarray(accs), axis=0)
        best_drop = drops[best_drop_idx]
        best_drop_acc = accs[best_drop_idx]

        file.write("Best accur drop - {}: {:.5f} with accuracy {:.4f}\n".format(best_drop_idx+1, best_drop, best_drop_acc))

        
def stage_optim_layers(fcnn, data):
    # 1st stage: search in coarse range
    lower = COARSE_RANGE["lower"]
    upper = COARSE_RANGE["upper"]
    
    layers = []
    accs = []
    
    for it in range(MAX_ITER):
        layer = randint(lower, upper, 2)
        print ("Experiment {} for layer: {}".format(it+1, layer))
        layers.append(layer)
        fcnn.set_hidden_dims(layer)
        solver = Solver(fcnn,
                        data,
                        update_rule='sgd_momentum',
                        optim_config={"learning_rate":LEARNING_RATE, "momentum":MOMENTUM},
                        lr_decay=LEARNING_DECAY,
                        print_every=100,
                        batch_size=BATCH_SIZE,
                        checkpoint_name="checkpoints/test" if CHECKOUT else None,
                        num_epochs=EPOCH_NUM)
        solver.train()
        accs.append(solver.best_val_acc)
        loss = solver.loss_history
        t_acc = solver.train_acc_history
        v_acc = solver.val_acc_history
        draw_loss_acc(loss, t_acc, v_acc, "layer_{}_{}/{}-{:4}-{:4}".format(lower, upper, it+1, layer[0], layer[1]))

    path = "params/"
    filename = "layer_{}_{}.txt".format(lower, upper)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+filename, 'w') as file:
        for line, layer in enumerate(layers):
            stat = "{}. layer: [{:4}, {:4}] - accuracy: {:.4f}".format(line+1,
                                                                       layer[0],
                                                                       layer[1],
                                                                       accs[line])
            file.write(stat + "\n")

        best_layer_idx = np.argmax(np.asarray(accs), axis=0)
        best_layer = layers[best_layer_idx]
        best_layer_acc = accs[best_layer_idx]

        file.write("Best accur layer - {}: [{:4}, {:4}] with accuracy {:.4f}\n".format(best_layer_idx+1, best_layer[0], best_layer[1], best_layer_acc))



data = get_FER2013_data(TRAIN_NUM, VALID_NUM, TEST_NUM)
# data = get_FER2013_data_from_binary(VALID_NUM)

INPUT_DIMS = np.prod(data["X_train"].shape[1:])
HIDDEN_DIMS = np.asarray([1155, 1155])

fcnn = FullyConnectedNet(HIDDEN_DIMS,
                         INPUT_DIMS,
                         CLASS_NUM,
                         DROPOUT,
                         REGULAR,
                         weight_scale=5e-3)
stage_optim_layers(fcnn, data)
# stage_optim_drop(fcnn, data)
