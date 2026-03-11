import argparse
import json
import os



def run(args):
	# Load id2label mapping
	with open(args.id2label_path) as f:
		id2label = json.load(f)
	label2id = {v: int(k) for k, v in id2label.items()}

	# Load new texts
	with open(args.new_texts_path) as f:
		new_texts = json.load(f)

	# Creating additional data
	additional_data = []
	for i_sample, new_text in enumerate(new_texts):
		new_id = args.preffix + "__" + str(i_sample+1)
		labels = [label2id[args.label]]
		additional_data.append({
			"text": new_text,
			"labels": labels,
			"id": new_id
		})

	# Saving additional data
	filename = "additional_train_data_" + args.preffix + ".json"
	with open(os.path.join(args.output_path, filename), mode='w') as f:
		f.write(json.dumps(additional_data))



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--preffix")
	parser.add_argument("--new-texts-path")
	parser.add_argument("--id2label-path")
	parser.add_argument("--label")
	parser.add_argument("--output-path")

	args = parser.parse_args()

	run(args)

