# File to take csv's from tensorboard data and make the clean accuracy plots.
# Convert image files to big : for f in *.png; do convert $f -filter point -scale 1000% big_$f; done

import matplotlib.pyplot as plt
import numpy as np
import csv

def line_side(pl, p, pr):
    """Gets the side of the line p is on.
    Args:
        pl (list[int]): Left point of line. (0 is x, 1 is y.)
        p (list[int]): Testing point.
        pr (list[int]): Right point of line.
    Returns:
        (int) 1 for one side, -1 for other, 0 for on the line.
    """
	v1 = (pr[0]-pl[0], pr[1]-pl[1])   # Vector 1
	v2 = (pr[0]-p[0], pr[1]-p[1])   # Vector 1
	xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product
	if xp > 0:
	    return 1
	elif xp < 0:
	    return -1
	else:
	    return 0

def convex_hull(objective, accuracy):
    """Removes all points where there exists two points that form a line above them.
    Args:
        objective (list[float]): List of certified area values.
        accuracy (list[float]): List of clean accuracy values.
    Returns:
        (list[float], list[float]) Filtered objective list, filtered accuracy list.
    """
	objective, accuracy = zip(*sorted(zip(objective, accuracy)))
	points = list(zip(objective, accuracy))
	ind = accuracy.index(max(accuracy))  # Start at index with maximum accuracy
	new_points = [points[ind]]
	new_add = True
	while new_add:
		new_add = False
		for new_ind in range(len(points)-1, ind, -1):
			point_sides = [line_side(points[ind], points[i], points[new_ind]) for i in range(ind+1, new_ind)]
			if sum(point_sides) == len(point_sides):  # Points are all on positive side
				new_points.append(points[new_ind])
				ind = new_ind
				new_add = True
				break
	new_objective, new_accuracy = zip(*new_points)
	return new_objective, new_accuracy

def blanket(objective, accuracy):
    """Put a blanket over points. Retain only points with highest accuracy in an objective range.
    Args:
        objective (list[float]): List of certified area values.
        accuracy (list[float]): List of clean accuracy values.
    Returns:
        (list[float], list[float]) Filtered objective list, filtered accuracy list.
    """
	accuracy, objective = zip(*sorted(zip(accuracy, objective), reverse=True))
	new_acc = list(accuracy)[0:2]
	new_obj = list(objective)[0:2]
	lb = min(new_obj)
	rb = max(new_obj)
	for i in range(2, len(objective)):
		if min(lb, rb, objective[i]) != lb or max(lb, rb, objective[i]) != rb:
			new_acc.append(accuracy[i])
			new_obj.append(objective[i])
			lb = min(new_obj)
			rb = max(new_obj)
	new_obj, new_acc = zip(*sorted(zip(new_obj, new_acc)))
	min_ind = new_acc.index(max(new_acc))  # Remove all points with an objective and accuracy lower than the highest accuracy point
	new_obj = new_obj[min_ind:]
	new_acc = new_acc[min_ind:]
	return new_obj, new_acc

def plot_line(objective_file, accuracy_file, label, fmt, print_index=False):
    """Plot a clean accuracy line
    Args:
        objective_file (str): Filepath to the csv file of certified area from tensorboard data.
        accuracy_file (str): Filepath to the csv file of acuracy from tensorboard data.
        label (str): Label to be used for the line in the legend.
        fmt (str): Format string for the line.
        print_index (bool): Whether to print out the indexes of the points plotted.
    """
	accuracy = []
	objective = []
	with open(accuracy_file) as csvfile:
		next(csvfile) # skip header line
		reader = csv.reader(csvfile)
		for row in reader:
			accuracy.append(float(row[2]))
	with open(objective_file) as csvfile:
		next(csvfile) # skip header line
		reader = csv.reader(csvfile)
		for row in reader:
			objective.append(float(row[2]))
	# n_objective, n_accuracy = zip(*sorted(zip(objective, accuracy)))
	# n_objective, n_accuracy = convex_hull(objective, accuracy)
	n_objective, n_accuracy = blanket(objective, accuracy)
	if print_index:
		print([objective.index(o) for o in n_objective])
	plt.plot(n_objective, n_accuracy, fmt, label=label)

# CUSTOM: Uncomment sections for different plots, or add your own.

# # MNIST cert area
# acc_file = 'MNIST/csv_files/mnist_acc.csv'
# obj_file = 'MNIST/csv_files/mnist_area.csv'
# orig_acc_file = 'MNIST/csv_files/orig_mnist_acc.csv'
# orig_obj_file = 'MNIST/csv_files/orig_mnist_area.csv'
# plot_rob = False
# plt.ylim(0.9, 1)
# plt.xlim(-200, 500)

# # Fashion MNIST cert area
# acc_file = 'Fashion_MNIST/csv_files/fmnist_acc.csv'
# obj_file = 'Fashion_MNIST/csv_files/fmnist_area.csv'
# orig_acc_file = 'Fashion_MNIST/csv_files/orig_fmnist_acc.csv'
# orig_obj_file = 'Fashion_MNIST/csv_files/orig_fmnist_area.csv'
# plot_rob = False
# plt.ylim(0.2, 1)
# plt.xlim(-800, 200)

# # CIFAR10 cert area
# acc_file = 'CIFAR10/csv_files/cifar10_acc.csv'
# obj_file = 'CIFAR10/csv_files/cifar10_area.csv'
# rob_acc_file = 'CIFAR10/csv_files/cifar10robust_acc.csv'
# rob_obj_file = 'CIFAR10/csv_files/cifar10robust_area.csv'
# orig_acc_file = 'CIFAR10/csv_files/orig_cifar10_acc.csv'
# orig_obj_file = 'CIFAR10/csv_files/orig_cifar10_area.csv'
# orig_rob_acc_file = 'CIFAR10/csv_files/orig_cifar10robust_acc.csv'
# orig_rob_obj_file = 'CIFAR10/csv_files/orig_cifar10robust_area.csv'
# plot_rob = True
# plt.ylim(0.4, 1)
# plt.xlim(-4500, -1000)

# # Imagenet cert area
# acc_file = 'Imagenet/csv_files/imagenet_acc.csv'
# obj_file = 'Imagenet/csv_files/imagenet_area.csv'
# rob_acc_file = 'Imagenet/csv_files/imagenetrobust_acc.csv'
# rob_obj_file = 'Imagenet/csv_files/imagenetrobust_area.csv'
# orig_acc_file = 'Imagenet/csv_files/orig_imagenet_acc.csv'
# orig_obj_file = 'Imagenet/csv_files/orig_imagenet_area.csv'
# orig_rob_acc_file = 'Imagenet/csv_files/orig_imagenetrobust_acc.csv'
# orig_rob_obj_file = 'Imagenet/csv_files/orig_imagenetrobust_area.csv'
# plot_rob = True
# plt.ylim(0.2, 0.9)
# plt.xlim(left=-100000, right=-40000)

# plt.rc('font', size=20)          # controls default text sizes
# plt.rc('axes', titlesize=20)     # fontsize of the axes title
# plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
# plt.rc('legend', fontsize=12)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

if plot_rob:
	plot_line(obj_file, acc_file, 'Low Robustness (Ours)', '-r', print_index=True)
	plot_line(orig_obj_file, orig_acc_file, 'Low Robustness (Cohen et. al.)', '--b')
	plot_line(rob_obj_file, rob_acc_file, 'High Robustness (Ours)', '-m', print_index=True)
	plot_line(orig_rob_obj_file, orig_rob_acc_file, 'High Robustness (Cohen et. al.)', '--c')
else:
	plot_line(obj_file, acc_file, 'Ours', '-r', print_index=True)
	plot_line(orig_obj_file, orig_acc_file, 'Cohen et. al.', '--b')

plt.grid()
plt.ylabel("Clean Accuracy", fontsize=15)
plt.yticks(fontsize=12)
plt.xlabel('Certified Area', fontsize=15)
plt.xticks(fontsize=12)
plt.legend(fontsize=12)

plt.show()
