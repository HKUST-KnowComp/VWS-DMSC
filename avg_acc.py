import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='trip', choices=['trip', 'beer'])
args = parser.parse_args()
if args.dataset == 'trip':
	num_aspects = 7
elif args.dataset == 'beer':
	num_aspects = 4

dev_acc_list = []
dev_text_list = []
dev_corr_list = []
dev_total_list = []
test_acc_list = []
test_text_list = []
test_corr_list = []
test_total_list = []
for i in range(num_aspects):
	with open("{}_asp_{}".format(args.dataset,i),'r') as fin:
		for line in fin:
			if line[:len("Dev Acc ")] == "Dev Acc ":
				line = line.strip()[len("Dev Acc "):]
				acc, corr_total = line.split(" ")
				corr, total = corr_total[1:-1].split("/")
				dev_acc_list.append(float(acc))
				dev_text_list.append(line)
				dev_corr_list.append(float(corr))
				dev_total_list.append(float(total))
			elif line[:len("Test Acc ")] == "Test Acc ":
				line = line.strip()[len("Test Acc "):]
				acc, corr_total = line.split(" ")
				corr, total = corr_total[1:-1].split("/")
				test_acc_list.append(float(acc))
				test_text_list.append(line)
				test_corr_list.append(float(corr))
				test_total_list.append(float(total))
			else:
				pass
print("{} Avg Dev Accuracy {:.4f} {}/{}".format(args.dataset,sum(dev_corr_list)/sum(dev_total_list), int(sum(dev_corr_list)), int(sum(dev_total_list)) ))
print("{} Avg Test Accuracy {:.4f} {}/{}".format(args.dataset,sum(test_corr_list)/sum(test_total_list), int(sum(test_corr_list)), int(sum(test_total_list)) ))
