# overall test
test:	linear relu dropout softmax fcnn fcnnoverfit fcnn2layer

# individual test on the different functionalities
linear:
	python --version
	python -m test.test_layers TestLinearLayer

relu:
	python --version
	python -m test.test_layers TestReLULayer

dropout:
	python --version
	python -m test.test_layers TestDropoutLayer

softmax:
	python --version
	python -m test.test_classifiers

fcnn:
	python --version
	python -m test.test_fcnet

fcnnoverfit:
	python --version
	python -m src.overfit_fcnet

fcnn2layer:
	python --version
	python -m src.train_fcnet

fcnnfer:
	python --version
	python -m src.fer_fcnn

fcnnpredict:
	python --version
	python -m test.test_predictions TestFCNNPrediction

cnnpredict:
	python --version
	python -m test.test_predictions TestCNNPrediction

optim:
	python --version
	python -m src.utils.fcnn_optimizer

# clean
clean:	cleansrc cleantest

cleansrc:
	rm -f src/*.pyc -r src/__pycache__

cleantest:
	rm -f test/*.pyc

# convert the markdown to the pdf
pdf:
	pandoc manuals/assignment2_advanced.md --pdf-engine=xelatex -o manuals/assignment2_advanced.pdf -V geometry:margin=1in --variable urlcolor=cyan --template eisvogel --listings
