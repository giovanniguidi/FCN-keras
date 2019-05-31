import numpy as np

def score_prediction(y_true_one_hot, y_pred_one_hot, n_classes):
    
    y_true = np.argmax(y_true_one_hot, axis=3)
    y_pred = np.argmax(y_pred_one_hot, axis=3)

    t_i = np.zeros((n_classes,))
    n_ii = np.zeros((n_classes,))
    n_ji = np.zeros((n_classes,))
    
    for i in range(n_classes):
#    for i in range(1):

        for j in range(len(y_true_one_hot)):
            y_true_flattened = y_true[j, ...].flatten()
            y_pred_flattened = y_pred[j, ...].flatten()

            t_i[i] += len(y_true_flattened[y_true_flattened == i])
            
            class_index = np.where(y_true_flattened == i)[0]
            n_ii[i] += len(np.where(y_true_flattened[class_index] == y_pred_flattened[class_index])[0])
            
            n_ji[i] += len(np.where(y_pred_flattened == i)[0])

    #pixel accuracy
    pixel_accuracy = np.sum(n_ii)/np.sum(t_i)
    
    #select only classes that are present (otherwise Nan)
    index = np.where(t_i > 0)
    
    #mean_accuracy
    tot_true_pixels = n_ii[index]/t_i[index]   
    mean_accuracy = np.average(tot_true_pixels)


    #mean IoU
    union = (t_i + n_ji - n_ii)
    IoU = n_ii[index] / union[index]
    mean_IoU = np.average(IoU)
    
    #f-weighted mean IoU
    freq_weighted_mean_IoU = ( 1./ np.sum(t_i[index]) )  * np.sum(t_i[index] * IoU )    

    return pixel_accuracy, mean_accuracy, mean_IoU, freq_weighted_mean_IoU

   