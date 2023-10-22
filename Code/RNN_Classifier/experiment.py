import tempfile

import click
import mlflow
import numpy as np
import torch
from torch import optim
from torchgan import losses

from Code.Dataset import BCIDataset
from Code.RNN_Classifier.data import synthesis_df
from Code.RNN_Classifier.metrics import classify, tstr, mean_feature_error
from Code.RNN_Classifier.models.cnngan import CNNCGANGenerator, CNNCGANDiscriminator
from Code.RNN_Classifier.models.sequence_trainer import SequenceTrainer
from Code.RNN_Classifier.samplers import BinaryBalancedSampler


@click.command()
@click.option("--lr", type=float)
@click.option("--epochs", type=int)
@click.option("--ncritic", type=int)
@click.option("--batch_size", type=int)
@click.option("--dataset_transform", type=str)
@click.option("--signals", type=int)
@click.option("--gen_dropout", type=float)
@click.option("--noise_size", type=int)
@click.option("--hidden_size", type=int)
@click.option("--flag", type=str)
def cli(**opt):
    main(**opt)


def main(**opt):
    batch_size = opt['batch_size'] if opt['batch_size'] != -1 else None

    dataset = BCIDataset()
    X = torch.from_numpy(dataset.X).cuda()
    y = torch.from_numpy(dataset.y).long().cuda()
    sampler = BinaryBalancedSampler(X, y, batch_size=batch_size)

    network = {
        'generator': {
            'name': CNNCGANGenerator,
            'args': {
                'output_size': opt['signals'],
                'dropout': opt['gen_dropout'],
                'noise_size': opt['noise_size'],
                'hidden_size': opt['hidden_size']
            },
            'optimizer': {
                'name': optim.RMSprop,
                'args': {
                    'lr': opt['lr']
                }
            }
        },
        'discriminator': {
            'name': CNNCGANDiscriminator,
            'args': {
                'input_size': opt['signals'],
                'hidden_size': opt['hidden_size']
            },
            'optimizer': {
                'name': optim.RMSprop,
                'args': {
                    'lr': opt['lr']
                }
            }
        }
    }

    wasserstein_losses = [losses.WassersteinGeneratorLoss(),
                          losses.WassersteinDiscriminatorLoss(),
                          losses.WassersteinGradientPenalty()]

    trainer = SequenceTrainer(models=network,
                              recon=None,
                              ncritic=opt['ncritic'],
                              losses_list=wasserstein_losses,
                              epochs=opt['epochs'],
                              retain_checkpoints=1,
                              checkpoints=f"{MODEL_DIR}/",
                              mlflow_interval=50,
                              device=DEVICE)

    trainer(sampler=sampler)
    trainer.log_to_mlflow()
    X_synth, y_synth = synthesis_df(trainer.generator, dataset)

    X_real = X.detach().cpu().numpy()
    mfe = np.abs(mean_feature_error(X_real, X_synth))

    mlflow.set_tag('flag', opt['flag'])
    mlflow.log_metric('mean_feature_error', mfe)

    trainer_class = classify(X_synth, y_synth, epochs=2_000, batch_size=batch_size)
    trainer_tstr = tstr(X_synth, y_synth, X, y, epochs=3_000, batch_size=batch_size)


if __name__ == '__main__':
    with mlflow.start_run():
        with tempfile.TemporaryDirectory() as MODEL_DIR:
            if torch.cuda.is_available():
                DEVICE = torch.device("cuda")
                torch.backends.cudnn.deterministic = True
            else:
                DEVICE = torch.device("cpu")
            cli()
