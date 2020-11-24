from typing import Tuple
import pytorch_lightning as pl

from torch.optim import Adam
from torch import nn, Size
from torch.tensor import Tensor

from efficientnet_pytorch import EfficientNet
from transformers import T5ForConditionalGeneration, T5Tokenizer


class ConvolutionalBridge(nn.Module):
    def __init__(self,
                 encoder_output_size: Size,
                 decoder_input_size: Size) -> None:
        
        self.encoder_output_size = encoder_output_size
        self.decoder_input_size = decoder_input_size

    def forward(self, images):
        pass


class EfficientNetEncoder(nn.Module):
    def __init__(self,
                 pretrained_model: str,
                 in_channels: int = 3) -> None:
        super().__init__()

        self.net = EfficientNet.from_pretrained(pretrained_model,
                                                in_channels=in_channels)

        # TODO Make dynamic based on pretrained_model
        self.output_size = (1280, 7, 7)  

    def forward(self, images):
        return self.net.extract_features(images)


class T5Decoder(nn.Module):
    def __init__(self, pretrained_model: str) -> None:
        super().__init__()

        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model)

        self.input_size = (-1, self.t5.model_dim)  # Get Encoder Shape

    def forward(self, encoder_inputs, targets = None):
        if self.training:
            assert targets is not None

            output = self.t5(input_embeds=encoder_inputs,
                             labels=targets,
                             output_dict=True)
            
            return output.loss

        return None  #  TODO: Implement generation method.


class Finetuner(pl.LightningModule):
    def __init__(self,
                 encoder: str,
                 decoder: str,
                 bridge: str,
                 learning_rate: float,
                 pretrained_encoder: str = None,
                 pretrained_decoder: str = None,
                 freeze_settings: Tuple[bool, bool, bool] = (False, False, False),
                 **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.encoder = getattr(globals(), encoder)(pretrained_encoder, **kwargs)
        self.decoder = getattr(globals(), decoder)(pretrained_decoder, **kwargs)

        self.bridge = getattr(globals(), bridge)(
            self.encoder.output_size,
            self.decoder.input_size)

        self.setup_layer(freeze_settings[0], self.encoder)
        self.setup_layer(freeze_settings[1], self.bridge)
        self.setup_layer(freeze_settings[2], self.decoder)

    def setup_layer(self, freeze: bool, layer: nn.Module):
        if freeze is True:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, images: Tensor, targets: Tensor):
        encoded = self.encoder(images)
        bridge_output = self.bridge(encoded)
        decoded = self.decoder(bridge_output, targets)

        return decoded

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        loss = self(batch['image'], batch['target'])

        self.log('train_loss', loss, on_epoch=True)

        return loss

    
