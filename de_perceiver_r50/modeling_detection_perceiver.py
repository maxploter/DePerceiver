from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
import torchvision
from perceiver_pytorch import Perceiver
from torchvision.ops.misc import FrozenBatchNorm2d


class DetectionPerceiver(nn.Module):
	def __init__(self, backbone, perceiver, class_embed, bbox_embed):
		super().__init__()
		self.backbone = backbone
		self.perceiver = perceiver
		self.class_embed = class_embed
		self.bbox_embed = bbox_embed

	def forward(self, x):
		x = self.backbone(x)
		x = x['0'].permute(0, 2, 3, 1)
		x = self.perceiver(data=x, return_embeddings=True)
		return {'pred_logits': self.class_embed(x), 'pred_boxes': self.bbox_embed(x)}


def build_model(config):
	# Backbone
	backbone = IntermediateLayerGetter(
		torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, norm_layer=FrozenBatchNorm2d),
		return_layers={'layer4': "0"}
	)

	# Perceiver
	fourier_channels = 2 * ((config["num_freq_bands"] * 2) + 1)
	perceiver = Perceiver(
		input_channels=config["num_channels"],
		input_axis=2,
		num_freq_bands=config["num_freq_bands"],
		max_freq=config["max_freq"],
		depth=config["enc_layers"],
		num_latents=config["num_queries"],
		latent_dim=config["hidden_dim"],
		cross_heads=config["enc_nheads_cross"],
		latent_heads=config["nheads"],
		cross_dim_head=(config["num_channels"] + fourier_channels) // config["enc_nheads_cross"],
		latent_dim_head=config["hidden_dim"] // config["nheads"],
		self_per_cross_attn=config["self_per_cross_attn"],
		fourier_encode_data=True,
		attn_dropout=config["dropout"],
		ff_dropout=config["dropout"],
		final_classifier_head=False
	)

	# Embeddings
	bbox_embed = nn.Sequential(
		nn.Linear(config["hidden_dim"], config["hidden_dim"]),
		nn.ReLU(),
		nn.Linear(config["hidden_dim"], config["hidden_dim"]),
		nn.ReLU(),
		nn.Linear(config["hidden_dim"], 4),
		nn.Sigmoid()
	)

	class_embed = nn.Linear(config["hidden_dim"], config["num_classes"] + 1)

	return DetectionPerceiver(backbone, perceiver, class_embed, bbox_embed)