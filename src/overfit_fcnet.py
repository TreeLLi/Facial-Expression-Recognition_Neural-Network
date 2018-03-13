import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

from src.evaluator import draw_loss_acc

"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
TRAIN_NUM = 50
VALID_NUM = 1
TEST_NUM = 10

CLASS_NUM = 10

data = get_CIFAR10_data(TRAIN_NUM, VALID_NUM, TEST_NUM)

print (data["y_test"].shape)
print (data["y_test"])

INPUT_DIMS = np.prod(data["X_train"].shape[1:])
HIDDEN_DIMS = np.asarray([400, 400])

fcnn = FullyConnectedNet(HIDDEN_DIMS, INPUT_DIMS, CLASS_NUM)
solver = Solver(fcnn, data, update_rule='sgd', optim_config={"learning_rate":1e-3}, print_every=1, num_epochs=20)
solver.train()

y = fcnn.predict(data["X_test"])
print (y)
fcnn.save()

draw_loss_acc(solver.loss_history, solver.train_acc_history, solver.val_acc_history, "overfit")

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
