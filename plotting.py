def smooth_curve(points, factor=0.8):
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous * factor + point * (1 - factor))
		else:
			smoothed_points.append(point)
	return smoothed_points


def showPlots(history, acc_mode):
	import matplotlib.pyplot as plt
	acc = history.history[acc_mode]
	val_acc = history.history["val_"+acc_mode]
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.plot(epochs, smooth_curve(val_acc), 'r', label="Smoothed Validation acc")
	plt.title('Training and validation acc')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.plot(epochs, smooth_curve(val_loss), 'r',
	         label="Smoothed Validation loss")
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()


def plot_confusion_matrix(matrix, target_names):
    import matplotlib.pyplot as plt
    import numpy as np


    fig, ax = plt.subplots()
    ax.pcolor(matrix, cmap="Reds")
    # ax.set_title("Confusion Matrix")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(0.5, len(matrix), 1))
    ax.set_xticklabels(target_names)
    ax.set_yticks(np.arange(0.5, len(matrix), 1))
    ax.set_yticklabels(target_names)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    from matplotlib import cm
    color_map = cm.get_cmap("Reds", 8)
    for j, category in enumerate(matrix):
        for i, value in enumerate(category):
            plt.text(i+0.5, j+0.5, str(value), color=color_map(1 - (value/np.max(matrix))),
            fontsize=20, horizontalalignment="center", verticalalignment="center")
    plt.show()
