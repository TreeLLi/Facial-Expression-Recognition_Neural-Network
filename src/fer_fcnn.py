import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.evaluator import draw_loss_acc, evaluate

from src.utils.data_utils import get_FER2013_data, get_FER2013_data_from_binary

TRAIN_NUM = None
VALID_NUM = 2000
TEST_NUM = None

CLASS_NUM = 7

DROPOUT = 0
# DROPOUT = 0.23524
REGULAR = 0
# REGULAR = 0.00091
BATCH_SIZE = 100
EPOCH_NUM = 50

data = get_FER2013_data(TRAIN_NUM, VALID_NUM, TEST_NUM)
# data = get_FER2013_data_from_binary(VALID_NUM)

INPUT_DIMS = np.prod(data["X_train"].shape[1:])
HIDDEN_DIMS = np.asarray([1150, 1150])

LEARNING_RATE = 0.00344
MOMENTUM = 0.9
LEARNING_DECAY = 0.9

CHECKOUT = False

fcnn = FullyConnectedNet(HIDDEN_DIMS,
                         INPUT_DIMS,
                         CLASS_NUM,
                         DROPOUT,
                         REGULAR,
                         weight_scale=5e-3)

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

y = fcnn.predict(data["X_test"])
name = "fer_{:4}_{:4}_{:.4f}_{:.4f}_{:2}".format(HIDDEN_DIMS[0], HIDDEN_DIMS[1], DROPOUT, REGULAR, EPOCH_NUM)
evaluate(y, data["y_test"], CLASS_NUM, name)
fcnn.save(name)

draw_loss_acc(solver.loss_history, solver.train_acc_history, solver.val_acc_history, name)
