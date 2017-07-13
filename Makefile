MSG = "Update at $(shell /bin/date '+%Y-%m-%d %H-%M-%S')"

push:
	git add -A
	git commit -m $(MSG)
	git push origin master

pull: 
	git pull