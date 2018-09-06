import numpy as np
import csv
import sys

DATASET_FILE = 'pima-indians-diabetes.csv'
NUM_ITERATIONS = 1000
LEARNING_RATE = .4
# portion from whole data for cross-validation(in %)
CV_PORTION = .3
# treshold for cross-validation
CV_TRESHOLD = .5

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def predict(x, theta):
	# vectorized hypothesis
	h = np.inner(theta.transpose(), x)
	# sigmoid(logistic) function applied to hyphotesis
	return sigmoid(h)

def compute_cost(x, theta):
	h = predict(x[:, :-1], theta)
	y = x[:, -1]
	cost = (1 / len(x)) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
	return cost

# provide feature scaling with mean normalization
def feature_scaling(dataset):
	max_values = np.amax(dataset, axis=0)
	min_values = np.amin(dataset, axis=0)
	mean_values = np.mean(dataset, axis=0)
	for col in range(dataset.shape[1] - 1):
		mean = mean_values[col]
		# also here can be standart deviation
		difference = max_values[col] - min_values[col]
		for row in range(len(dataset[:, col])):
			# new value with applied scaling and normalization
			dataset[row, col] = (dataset[row, col] - mean) / difference

#  estimating theta using stochastic gradient descent
def get_theta_stochastic_gd(train_data, learning_rate, num_iterations):
	# theta vector
	theta = np.zeros(len(train_data[0, :-1]), float)
	for i in range(num_iterations):
		# random shuffle train_data
		np.random.shuffle(train_data)
		sum_error = 0
		for train_row in range(len(train_data)):
			for coef_row in range(len(theta)):
				# pass training row(except output value) to predict function 
				predicted = predict(train_data[train_row, :-1], theta)
				error = predicted - train_data[train_row, -1]
				sum_error += error**2
				# update current_coefficient(do gradient descent step)
				theta[coef_row] = theta[coef_row] - learning_rate * error * train_data[train_row, coef_row]
		print('>iteration={}, learning_rate={}, error={:.2f}, cost={:.3f}'.format(
			i, 
			learning_rate, 
			sum_error,
			compute_cost(train_data, theta),
			))
	return theta

# estimating theta using batch gradient descent
def get_theta_batch_gd(train_data, learning_rate, num_iterations):
	theta = np.zeros(len(train_data[0, :-1]), float)
	m = len(train_data)
	for i in range(num_iterations):
		theta -=\
		 (learning_rate / m) * np.inner(train_data[:, :-1].transpose(), (predict(train_data[:, :-1], theta) - train_data[:, -1]))
		print('>iteration={}, cost={:.3f}'.format(i, compute_cost(train_data, theta)))
	return theta

# calculating cross-validation error
def get_cross_validation_error(dataset, theta):
	error = 0
	examples_count = len(dataset)
	for row in range(examples_count):
		predicted = predict(dataset[row, :-1], theta)
		y = dataset[row, -1]
		if y:
			error += 1 if predicted < CV_TRESHOLD else 0
		else:
			error += 1 if predicted >= CV_TRESHOLD else 0
	return error / examples_count
	
# reading data from file and transforming string values to float
rows = []
with open(DATASET_FILE) as csv_file:
	csv_reader = csv.reader(csv_file)
	for row in csv_reader:
		rows.append(list(map(float, row)))

# data transformation to numpy matrix
dataset = np.array(rows, float)
# feature scalling with normalization
feature_scaling(dataset)
# adding ones(bias X0) to each row
ones = np.ones((len(dataset), 1), float)
dataset = np.concatenate((ones, dataset), axis=1)
# shuffle train_data
np.random.shuffle(dataset)
# choosing data for cross-validation and training
cv_to_index = int(len(dataset) * CV_PORTION)
cv_dataset = dataset[:cv_to_index]
train_dataset = dataset[cv_to_index:]

if __name__ == "__main__":
    gradient_descent_type = 'stochastic'
    if len(sys.argv) >= 2:
    	    gradient_descent_type = sys.argv[1]
    if gradient_descent_type == 'stochastic':
    	theta = get_theta_stochastic_gd(train_dataset, LEARNING_RATE, NUM_ITERATIONS)
    elif gradient_descent_type == 'batch':
    	theta = get_theta_batch_gd(train_dataset, LEARNING_RATE, NUM_ITERATIONS)
    else:
    	raise ValueError('Invalid gradient descent type')
    cv_accuracy = 1 - get_cross_validation_error(cv_dataset, theta)
    cv_cost = compute_cost(cv_dataset, theta)
    print('>>cross-validation cost={}, accuracy={}'.format(cv_cost, cv_accuracy))
