train:
	rlaunch --cpu=4 --gpu=2 --memory=38000 --comment='train_recons' -- python3 train.py