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

#            n_pixel = 
            t_i[i] += len(y_true_flattened[y_true_flattened == i])
            
            class_index = np.where(y_true_flattened == i)[0]
#            n_pixel_true = np.where(y_true_flattened[class_index] == y_pred_flattened[class_index])[0]
            n_ii[i] += len(np.where(y_true_flattened[class_index] == y_pred_flattened[class_index])[0])
            
#            class_index_pred = np.where(y_pred_flattened == i)[0]        
            n_ji[i] += len(np.where(y_pred_flattened == i)[0])
            
    
    index = np.where(t_i > 0)
    tot_true_pixels = n_ii[index]/t_i[index]   
    
    pixel_accuracy = np.sum(n_ii)/np.sum(t_i)
    mean_accuracy = np.average(tot_true_pixels)

    den = (t_i + n_ji - n_ii)
    IoU = n_ii[index] / den[index]
    mean_IoU = np.average(IoU)
    
#    freq_weighted_mean_IoU = ( 1./ np.sum(t_i[index]) )  * np.average(t_i[index] * IoU )    

    freq_weighted_mean_IoU = ( 1./ np.sum(t_i[index]) )  * np.sum(t_i[index] * IoU )    

    
    return pixel_accuracy, mean_accuracy, mean_IoU, freq_weighted_mean_IoU


    

def mean_accuracy(y_true_one_hot, y_pred_one_hot, n_classes):

    y_true = np.argmax(y_true_one_hot, axis=3)
    y_pred = np.argmax(y_pred_one_hot, axis=3)
    
    #print(y_true.shape)
    
    t_i = np.zeros((n_classes,))
    n_ii = np.zeros((n_classes,))

    for i in range(n_classes):
#    for i in range(1):

        for j in range(len(y_true_one_hot)):
            y_true_flattened = y_true[j, ...].flatten()
            y_pred_flattened = y_pred[j, ...].flatten()

            n_pixel = y_true_flattened[y_true_flattened == i]
            t_i[i] += len(n_pixel)
            
            class_index = np.where(y_true_flattened == i)[0]
            n_pixel_true = np.where(y_true_flattened[class_index] == y_pred_flattened[class_index])[0]
            n_ii[i] += len(n_pixel_true)
    
    index = np.where(t_i > 0)
    tot_true_pixels = n_ii[index]/t_i[index]    
    mean_accuracy = np.average(tot_true_pixels)

    return mean_accuracy


def pixel_accuracy(y_true_one_hot, y_pred_one_hot, n_classes):

    y_true = np.argmax(y_true_one_hot, axis=3)
    y_pred = np.argmax(y_pred_one_hot, axis=3)
    
    tot_t_i = 0.
    tot_n_ii = 0.

    for j in range(len(y_true_one_hot)):
#        for j in range(1):

        y_true_flattened = y_true[j, ...].flatten()
        y_pred_flattened = y_pred[j, ...].flatten()

        tot_t_i += len(y_true_flattened)

        true_pixels = np.where(y_pred_flattened == y_true_flattened)[0]
        tot_n_ii += len(true_pixels)
    
    pixel_accuracy = tot_n_ii/tot_t_i
    
    return pixel_accuracy


def mean_IoU(y_true_one_hot, y_pred_one_hot, n_classes):

    y_true = np.argmax(y_true_one_hot, axis=3)
    y_pred = np.argmax(y_pred_one_hot, axis=3)
        
    t_i = np.zeros((n_classes,))
    n_ii = np.zeros((n_classes,))
    n_ji = np.zeros((n_classes,))
    
        
    for i in range(n_classes):

        for j in range(len(y_true_one_hot)):
            y_true_flattened = y_true[j, ...].flatten()
            y_pred_flattened = y_pred[j, ...].flatten()

            #t_i
            n_pixel = y_true_flattened[y_true_flattened == i]
            t_i[i] += len(n_pixel)

            #n_ii
            class_index = np.where(y_true_flattened == i)[0]
            n_pixel_true = np.where(y_true_flattened[class_index] == y_pred_flattened[class_index])[0]
            n_ii[i] += len(n_pixel_true)

            #n_ji
#            n_pixel_true_pred = np.where(y_true_flattened[class_index] == y_pred_flattened[class_index])[0]
            class_index_pred = np.where(y_pred_flattened == i)[0]
            #print(class_index_pred)
        
            n_ji[i] += len(class_index_pred)

    den = (t_i + n_ji - n_ii)
    index = np.where(t_i > 0)

    IoU = n_ii[index] / den[index]
    
#    print(IoU)
    
#    index = np.where(IoU > 0)
#    print(IoU[index])
#    tot_true_pixels = n_ii[index]/t_i[index]    
    
#    mean_accuracy = np.average(tot_true_pixels)

    mean_IoU = np.average(IoU)

    return mean_IoU
    
    
def freq_weighted_mean_IoU(y_true_one_hot, y_pred_one_hot, n_classes):

    y_true = np.argmax(y_true_one_hot, axis=3)
    y_pred = np.argmax(y_pred_one_hot, axis=3)
        
    t_i = np.zeros((n_classes,))
    n_ii = np.zeros((n_classes,))
    n_ji = np.zeros((n_classes,))
    
        
    for i in range(n_classes):

        for j in range(len(y_true_one_hot)):
            y_true_flattened = y_true[j, ...].flatten()
            y_pred_flattened = y_pred[j, ...].flatten()

            #t_i
            n_pixel = y_true_flattened[y_true_flattened == i]
            t_i[i] += len(n_pixel)

            #n_ii
            class_index = np.where(y_true_flattened == i)[0]
            n_pixel_true = np.where(y_true_flattened[class_index] == y_pred_flattened[class_index])[0]
            n_ii[i] += len(n_pixel_true)

            #n_ji
#            n_pixel_true_pred = np.where(y_true_flattened[class_index] == y_pred_flattened[class_index])[0]
            class_index_pred = np.where(y_pred_flattened == i)[0]
            #print(class_index_pred)
        
            n_ji[i] += len(class_index_pred)
            
    den = (t_i + n_ji - n_ii)
    index = np.where(t_i > 0)

    IoU = n_ii[index] / den[index]


    F_W_mean_IoU = ( 1./ np.sum(t_i[index]) )  * np.average(t_i[index] * IoU )

    return F_W_mean_IoU
    
    