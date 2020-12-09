# Novel Approaches For Tuning Models HyperParameters
Hyperparameters are crucial in machine learning, as they determine if the model will be efficient and accurate. Usually, a tuning process is used to find good combinations of hyperparameters. This tuning process used to be very time-consuming as it historically consisted of training the model with all combinations of hyperparameters among the space of all possibilities for grid-search, and of a random set of combinations for random search. Then results were compared to choose the best combination. Now notice that the space of possibilities grows exponentially with the number of hyperparameters. Furthermore, for complex deep-learning problems a single training can take days. 
Thus new approaches are needed to find good hyperparameters faster.
In this project, we would like to choose a few deep-learning problems as they are usually long to train and compare different tuning approaches using the Ray Tune library. We would like to empirically quantify and qualify the speed for each method as well as the efficiency of the parameters found. 

Grade: Unknown until end of January
