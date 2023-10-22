from sklearn.model_selection import train_test_split
from torch import optim

from Code.RNN_Classifier.models.classifiers import BinaryRNNClassifier
from Code.RNN_Classifier.samplers import BinaryBalancedSampler, IdentitySampler
from Code.RNN_Classifier.trainers import BinaryClassificationTrainer
from Code.RNN_Classifier.utils import numpy_to_cuda


def mean_feature_error(X_real, X_synth):
    return (1 - X_synth.mean(axis=0)/X_real.mean(axis=0)).mean(axis=0).mean()


# TODO(dsevero) in theory, we should seperate the
# training and testing datasets.
def tstr(X_synth, y_synth, X_real, y_real, epochs=3_000, batch_size=None):

    sequence_length = X_real.shape[1]
    sequence_size = X_real.shape[2]
    X_train, y_train = X_synth, y_synth
    X_test, y_test = X_real, y_real

    X_train, y_train = numpy_to_cuda(X_train, y_train)
    y_test, y_train = y_test.float(), y_train.float()

    sampler_train = BinaryBalancedSampler(X_train, y_train, 
                                          tile=True, batch_size=batch_size)
    sampler_test = IdentitySampler(X_test, y_test, tile=True)

    model = BinaryRNNClassifier(sequence_length=sequence_length,
                                input_size=sequence_size,
                                dropout=0.5,
                                hidden_size=100).cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=0.001)
    trainer = BinaryClassificationTrainer(model,
                                          optimizer,
                                          sampler_train,
                                          sampler_test,
                                          metrics_prepend='tstr_')
    trainer.train(epochs)
    return trainer


def classify(X, y, epochs=3_000, batch_size=None):
    sequence_length = X.shape[1]
    sequence_size = X.shape[2]

    (X_train, y_train, 
     X_test, y_test) = train_test_split(X, y, 0.3)  # Warning! modified original file
    (X_train, y_train, 
     X_test, y_test) = numpy_to_cuda(X_train, y_train,
                                     X_test, y_test)
    y_test, y_train = y_test.float(), y_train.float()

    sampler_train = BinaryBalancedSampler(X_train, y_train, 
                                          tile=True, batch_size=batch_size)
    sampler_test = IdentitySampler(X_test, y_test, tile=True)

    model = BinaryRNNClassifier(sequence_length=sequence_length,
                                input_size=sequence_size,
                                dropout=0.8,
                                hidden_size=100).cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=0.001)
    trainer = BinaryClassificationTrainer(model,
                                          optimizer,
                                          sampler_train,
                                          sampler_test)

    trainer.train(epochs)
    return trainer
