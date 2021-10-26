from config import config
from helper_functions import n_classes
import segmentation_models_pytorch as smp
import torch
import deepspeed
from deepspeed_argparser import add_argument
from collections import OrderedDict

def return_dict(module):

	state_dict = OrderedDict() if torch.distributed.get_rank() == 0 else None
	shared_weights = {}
	def get_layer_state_dict(module, prefix=""):
		# gather one layer at a time to be memory-efficient
		with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False))):
			if torch.distributed.get_rank() == 0:
				for name, param in module.named_parameters(recurse=False):
					if param is None:
						continue
					key = prefix + name
					# for shared weights we want to make sure not to unshare them when copying to cpu
					data_ptr_id = param.storage().data_ptr()
					if data_ptr_id in shared_weights:
						# shared weights
						#print(f"`{key}` is shared with `{shared_weights[data_ptr_id]}`")
						state_dict[key] = state_dict[ shared_weights[data_ptr_id] ]
					else:
						state_dict[key] = param.detach().cpu()
						shared_weights[data_ptr_id] = key

				# now buffers - not sure if need to take care of potentially shared weights here
				for name, buf in module.named_buffers(recurse=False):
					if buf is not None and name not in module._non_persistent_buffers_set:
						state_dict[prefix + name] = buf.detach().cpu()

		for name, child in module.named_children():
			if child is not None:
				get_layer_state_dict(child, prefix + name + ".")

	#see_memory_usage("before get_layer_state_dict", force=True)
	# XXX: not sure about starting prefix? see pretrained load
	get_layer_state_dict(module, prefix="")
	#see_memory_usage("after get_layer_state_dict", force=True)

	return state_dict

def save_ds_as_torch(model, ds_path, save_path):

	args = add_argument()

	parameters = filter(lambda p: p.requires_grad, model.parameters())

	model_engine, optimizer, train_loader, _ = deepspeed.initialize(
		args = args,
		model = model,
		model_parameters = parameters
		)

	model_engine.load_checkpoint(ds_path)

	state_dict = return_dict(model_engine.module)

	torch.save(state_dict, save_path)

	print('Done')

def save_deepspeed(model_engine, save_path):

	state_dict = return_dict(model_engine.module)

	torch.save(state_dict, save_path)