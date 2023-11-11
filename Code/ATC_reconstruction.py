import numpy as np
from torch import Tensor
from torch.nn import Parameter

from Code.ATCNet import ATCNet as TorchATCNet
from _LegacyCode.EEG_ATCNet.models import ATCNet as TfATCNet


def copy(legacy_model: str):
    kwargs = dict(
        # Dataset parameters
        n_classes=4,
        in_chans=22,
        in_samples=1125,
        # Sliding window (SW) parameter
        n_windows=5,
        # Attention (AT) block parameter
        attention="mha",  # Options: None, 'mha','mhla', 'cbam', 'se'
        # Convolutional (CV) block parameters
        eegn_F1=16,
        eegn_D=2,
        eegn_kernelSize=64,
        eegn_poolSize=7,
        eegn_dropout=0.3,
        # Temporal convolutional (TC) block parameters
        tcn_depth=2,
        tcn_kernelSize=4,
        tcn_filters=32,
        tcn_dropout=0.3,
        tcn_activation="elu",
    )
    torch_model = TorchATCNet(**kwargs)
    tf_model = TfATCNet(**kwargs)
    tf_model.load_weights(legacy_model)
    tf_weights = tf_model.weights
    parameter_weights = tuple(map(Tensor, map(np.array, tf_weights)))
    torch_model.conv_block.conv1.weight = Parameter(
        parameter_weights[0].permute((3, 2, 0, 1))
    )
    torch_model.conv_block.depthwise_conv.weight = Parameter(
        parameter_weights[5].reshape((32, 1, 22, 1))
    )
    # x = self.depthwise_conv(x)
    torch_model.conv_block.conv2.weight = Parameter(
        parameter_weights[10].permute((3, 2, 0, 1))
    )
    torch_model.mha_block.mha
    torch_model.tcn_block.layers1[0].weight = Parameter(
        parameter_weights[25].permute((3, 2, 0, 1))
    )
    return torch_model


if __name__ == "__main__":
    copy("results/tf_models/run-10/subject-7.h5")
