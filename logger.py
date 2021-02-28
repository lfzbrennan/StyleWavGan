import os 
import datetime

class Logger():
	def __init__(self, output_file):
		self.output_file = output_file

	def log(self, message):
		timestamp = datetime.datetime.now()
		output = f"{timestamp}\t" + message + "\n"
		with open(self.output_file, "a") as f:
			f.write(output)