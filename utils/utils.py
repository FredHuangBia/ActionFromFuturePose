from collections import defaultdict
import numpy as np


class acc_calculater():
	def __init__(self, classes=None):
		self.classes = classes
		self.num_class = len(classes)
		self.total_num = 0
		self.total_correct = 0
		self.class_nums = defaultdict(int)
		self.class_correct = defaultdict(int)


	def update(self, output, label, thresholded=False):
		if not thresholded:
			pred = np.argmax(output, axis = 1)
		else:
			pred = output
		now_correct = np.sum(pred == label)
		now_num = len(label)
		self.total_correct += now_correct
		self.total_num += now_num
		for cID in range(self.num_class):
			self.class_nums[cID] += np.sum(label==cID)
			idxs = np.where(label==cID)
			self.class_correct[cID] += np.sum(pred[idxs]==cID)
		now_acc = now_correct / now_num
		return now_acc


	def reset(self):
		self.total_num = 0
		self.total_correct = 0
		self.class_nums = defaultdict(int)
		self.class_correct = defaultdict(int)


	def get_acc(self):
		return self.total_correct / self.total_num


	def get_class_accs(self):
		class_accs = defaultdict(float)
		for cID in range(self.num_class):
			class_accs[self.classes[cID]] = self.class_correct[cID] / self.class_nums[cID]
		return class_accs


	def print_class_accs(self):
		class_accs = self.get_class_accs()
		for key in class_accs:
			print(key, class_accs[key])