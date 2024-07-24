
from typing import Any
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, colorstr
from ultralytics.nn.tasks import *
from ultralytics.nn.modules.conv import Concat
from ultralytics.nn.modules.head import Detect
from torch.nn.modules.upsampling import Upsample
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f
import torch.nn as nn

from ultralytics.nn.modules.block import SPPF

class YoloDetectionModel(DetectionModel):
	def __init__(self, cfg: str = 'yolov8n.yaml', ch: int = 3, nc: Any | None = None, verbose: bool = True) -> None:
		super().__init__(cfg, ch, nc, verbose)
		self.model = nn.Sequential([
			Conv(3,48,3,2),
			Conv(48,96,3,2),
			C2f(96,96,2,True),
			Conv(96,192,3,2),
			C2f(192,192,4,True),
			Conv(192,384,3,2),
			C2f(384,384,4,True),
			Conv(384,576,3,2),
			C2f(576,576,2,True),
			SPPF(576,576,5),
			Upsample(None,2,"nearest"),
			Concat(1),
			C2f(960,384,2),
			Upsample(None,2,"nearest"),
			Concat(1),
			C2f(576,192,2),
			Conv(192,192,3,2),
			Concat(1),
			C2f(576,384,2),
			Conv(384,384,3,2),
			Concat(1),
			C2f(960,576,2),
			Detect(20,[192, 384, 576])
		])

        # Build strides
		m = self.model[-1]  # Detect()

		s = 256  # 2x min stride
		m.inplace = self.inplace
		m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
		self.stride = m.stride
		m.bias_init()  # only run once

        # Init weights, biases
		initialize_weights(self)
		
		
	#expand the forward of the model
	def _predict_once(self, x, profile=False, visualize=False):
		y, dt = [], []
		""" ultralytics.nn.modules.conv.Conv """
		m = self.model[0]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.conv.Conv """
		m = self.model[1]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.block.C2f """
		m = self.model[2]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.conv.Conv """
		m = self.model[3]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.block.C2f """
		m = self.model[4]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.conv.Conv """
		m = self.model[5]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.block.C2f """
		m = self.model[6]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.conv.Conv """
		m = self.model[7]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.block.C2f """
		m = self.model[8]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.block.SPPF """
		m = self.model[9]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" torch.nn.modules.upsampling.Upsample """
		m = self.model[10]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.conv.Concat """
		m = self.model[11]
		x =  [x if j == -1 else y[j] for j in [-1, 6]]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.block.C2f """
		m = self.model[12]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" torch.nn.modules.upsampling.Upsample """
		m = self.model[13]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.conv.Concat """
		m = self.model[14]
		x =  [x if j == -1 else y[j] for j in [-1, 4]]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.block.C2f """
		m = self.model[15]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.conv.Conv """
		m = self.model[16]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.conv.Concat """
		m = self.model[17]
		x =  [x if j == -1 else y[j] for j in [-1, 12]]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.block.C2f """
		m = self.model[18]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.conv.Conv """
		m = self.model[19]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.conv.Concat """
		m = self.model[20]
		x =  [x if j == -1 else y[j] for j in [-1, 9]]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.block.C2f """
		m = self.model[21]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		""" ultralytics.nn.modules.head.Detect """
		m = self.model[22]
		x =  [x if j == -1 else y[j] for j in [15, 18, 21]]
		if profile:
				self._profile_one_layer(m, x, dt)
		x = m(x)
		y.append(x if m.i in self.save else None)  # save output
		
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	
		return x

    