import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data
from src.evaluator import draw_loss_acc, evaluate

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50% 
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################

TRAIN_NUM = 490
VALID_NUM = 10
TEST_NUM = 100

data = get_CIFAR10_data(TRAIN_NUM, VALID_NUM, TEST_NUM)

CLASS_NUM = 10
INPUT_DIMS = np.prod(data["X_train"].shape[1:])
HIDDEN_DIMS = np.asarray([100, 100])

fcnn = FullyConnectedNet(HIDDEN_DIMS, INPUT_DIMS, CLASS_NUM)
solver = Solver(fcnn, data, update_rule='sgd', optim_config={"learning_rate":1e-3}, print_every=100,
                num_epochs=20, lr_decay=0.95, num_train_samples=TRAIN_NUM, num_val_samples=VALID_NUM)

solver.train()

y = fcnn.predict(data["X_test"])

evaluate(y, data["y_test"], CLASS_NUM)

# draw_loss_acc(solver.loss_history, solver.train_acc_history, solver.val_acc_history, "train")


##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
