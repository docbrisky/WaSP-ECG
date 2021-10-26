import json

# ended up being a bit convoluted due to last-minute change in approach:

class Config():

	def __init__(self):

		pass

config = Config()

with open('config.json') as json_file:
			variables = json.load(json_file)

for k in list(variables.keys()):
	setattr(config, k, variables[k])