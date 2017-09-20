MSG = "Update at $(shell /bin/date '+%Y-%m-%d %H-%M-%S')"

push:
	git add -A
	git commit -m $(MSG)
	git push origin master

pull:
	git pull

convnet:
	python train_vgg16.py
	python train_vgg19.py
	python train_inc.py
	python train_mb.py
	python train_resnet.py
