MSG = "Update at $(shell /bin/date '+%Y-%m-%d %H-%M-%S')"

push:
	git add -A
	git commit -m $(MSG)
	git push origin master

pull:
	git pull

preprocess:
	python preprocess.py

find_players:
	# clone the tensorflow models first
	cd ./tensorflow-models/
	protoc ./object_detection/protos/*.proto --python_out=.
	cd ../
	python find_players.py

explore:
	python explore.py ../ds/

models:
	python train_vgg16.py
	python train_vgg19.py
	python train_inc.py
	python train_mb.py
	python train_resnet.py
	python train_lstm.py
	python train_conv3d.py

players_model:
	python train_with_players.py vgg16
	python train_with_players.py vgg19
	python train_with_players.py resnet
	python train_with_players.py inc