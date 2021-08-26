# Predict raw datasets, Make array of filenames and predictions for a whole day from compressed file

def MassPredictor(tar_files_path, predictions_path, model, excludelist=[]):
    """Predicts classification of compressed TSI photos
    
    If 'excludelist' isn't defined, no files are excluded from prediction.
    
    Parameters
    ----------
    tar_files_path : str
        The filepath to the compressed TSI photos. Photos are expected to be of
        .jpg format compressed with a .tar.gz ending

    predictions_path : str
        The filepath to where the CSV files containing the predictions are saved

    model : tf.keras.Model() class
        The model to be used in predicting the photos

    excludelist : str, list of str, optional
        Filenames of files that should not be predicted, e.g. other files in folder, files already 
        predicted or corrupted files (default is none)
    """
    import tarfile
    import re
    import os
    import pathlib
    import numpy as np
    import shutil

    classes = ['fogbow', 'iceoptics', 'no_optics']
    jpg = re.compile(r'.*?jpg.*?')

    tar_list = os.listdir(tar_files_path)

    for exclude in excludelist:
        tar_list.remove(exclude)

    print("Predicting on",len(tar_list), "dataset(s): \n",tar_list)

    temp_path = "temp/temporary_files_78654567654"
    
    try:
        shutil.rmtree(temp_path)
    except FileNotFoundError:
        pass
    
    pathlib.Path(temp_path).mkdir(parents=True, exist_ok=True)

    for i, tar_file in enumerate(tar_list):
        print(i+1,"/", len(tar_list),": Predicting ", tar_file)
        try:
            tar = tarfile.open(os.path.join(tar_files_path,tar_file), "r:gz")
            tar.extractall(temp_path, members=[m for m in tar.getmembers() if jpg.search(m.name)])

            img_list = os.listdir(temp_path)
            prediction_list = np.zeros((len(img_list),6))

            if len(img_list) > 0:
                for n, img in enumerate(img_list):
                    img_path = os.path.join(temp_path, img)
                    pred_raw = PredictImage(img_path, model)
                    prediction = np.argmax(pred_raw)
                    prediction_list[n,0] = img[:14]
                    prediction_list[n,1] = prediction
                    prediction_list[n,2] = pred_raw[0][prediction]
                    prediction_list[n,3] = pred_raw[0][0]
                    prediction_list[n,4] = pred_raw[0][1]
                    prediction_list[n,5] = pred_raw[0][2]

                prediction_list = prediction_list[prediction_list[:, 0].argsort()]

                np.savetxt(os.path.join(predictions_path, tar_file[3:11] + '.csv'), prediction_list, delimiter=',',fmt='%s')
            else:
                continue
        except:
            print('File currupted/wrong format, ignoring...')
           
     





# Lookup predictions and filter out the ones with highest confidence
def FilterPredictions(predicted_path, lower_limit, higher_limit=1.0, excludelist=[]):
    """Reads predictions off CSV files and output the predicted files within defined 
    confidence limits.
    
    If 'higher_limit' isn't defined, the higher limit is set to 1.0, i.e. the highest 
    possible limit. 

    If 'excludelist' isn't defined, no files are excluded from reading.
    
    Parameters
    ----------
    predicted_path : str
        The filepath to where the CSV files containing the predictions are saved

    lower_limit : float
        The lower confidence limit of the files included, should normally be between 0 and 1.

    higher_limit : float, optional
        The higher confidence limit of the files included, should be above 0 and max 1 (default is 1.0)

    excludelist : str, list of str, optional
        Filenames of files that should not be included, e.g. other files in folder, or corrupted files 
        (default is none)

    Outputs
    ----------
    fog_detections, ice_detections, no_detections

    fog_detections : list of str
        List of images with fog detected within defined limits 

    ice_detections : list of str
        List of images with ice detected within defined limits 

    no_detections : list of str
        List of images with none detected within defined limits 
    """
    import csv
    import os
    import numpy as np

    predicted_list = os.listdir(predicted_path)

    for exclude in excludelist:
        predicted_list.remove(exclude)

    fog_detections = np.empty((0,6), int)
    ice_detections = np.empty((0,6), int)
    no_detections  = np.empty((0,6), int)

    confidence_cutoff = lower_limit
    confidence_max = higher_limit

    for predicted in predicted_list:
        pred_csv = np.genfromtxt(os.path.join(predicted_path,predicted), delimiter=',')
        data_cutoff = pred_csv[(pred_csv[:,2] >= confidence_cutoff) & (pred_csv[:,2] <= confidence_max)]
        data_fog = np.where(data_cutoff[:,1]==0)
        fog_detections = np.append(fog_detections, data_cutoff[data_fog[0]], axis=0)
        data_ice = np.where(data_cutoff[:,1]==1)
        ice_detections = np.append(ice_detections, data_cutoff[data_ice[0]], axis=0)
        data_no  = np.where(data_cutoff[:,1]==2)
        no_detections = np.append(no_detections, data_cutoff[data_no[0]], axis=0)

    print('Number of detections with threshold between', confidence_cutoff*100,'% and ', confidence_max*100, '%: \n'
          'Fogbows:', len(fog_detections),'\n'
          'Ice optics:', len(ice_detections),'\n'
          'Nothing:', len(no_detections))

    return fog_detections, ice_detections, no_detections



# moving detections to inspection folder
def ExportFilteredFiles(tar_files_path, export_path, fog_detections,ice_detections,no_detections):
    """Copies the selected files from compressed tar.gz files to a new folder, sorted according 
    to classification.

    Parameters
    ----------
    tar_files_path : str
        The filepath to the compressed TSI photos. Photos are expected to be of
        .jpg format compressed with a .tar.gz ending
    
    export_path : str
        The filepath where the photos are copied to. Folder should already exist. 

    fog_detections : list of str
        The list of fog detections outputted by FilterPredictions

    ice_detections : list of str
        The list of ice detections outputted by FilterPredictions

    no_detections : list of str
        The list of none detections outputted by FilterPredictions
    """

    import re
    import tarfile
    import pathlib
    import os
    import shutil

    temp_path = "temp/temporary_files"
    pathlib.Path(temp_path).mkdir(parents=True, exist_ok=True)

    jpg = re.compile(r'.*?jpg.*?')

    pathlib.Path(os.path.join(export_path, 'no_det')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(export_path, 'ice_det')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(export_path, 'fog_det')).mkdir(parents=True, exist_ok=True)

    no_det_datelist = []
    no_det_filelist = []
    fog_det_datelist = []
    fog_det_filelist = []
    ice_det_datelist = []
    ice_det_filelist = []

    for i in range(len(no_detections[:,0])):
        no_det_datelist.append('tsi' + str(no_detections[:,0][i])[:8] + '.tar.gz')
        no_det_filelist.append(str(no_detections[:,0][i])[:-2] + '.jpg')
    no_det_datelist = list(dict.fromkeys(no_det_datelist))

    for i in range(len(fog_detections[:,0])):
        fog_det_datelist.append('tsi' + str(fog_detections[:,0][i])[:8] + '.tar.gz')
        fog_det_filelist.append(str(fog_detections[:,0][i])[:-2] + '.jpg')
    fog_det_datelist = list(dict.fromkeys(fog_det_datelist))

    for i in range(len(ice_detections[:,0])):
        ice_det_datelist.append('tsi' + str(ice_detections[:,0][i])[:8] + '.tar.gz')
        ice_det_filelist.append(str(ice_detections[:,0][i])[:-2] + '.jpg')
    ice_det_datelist = list(dict.fromkeys(ice_det_datelist))


    for date in no_det_datelist:
        tar = tarfile.open(os.path.join(tar_files_path,date), "r:gz")
        tar.extractall(temp_path, members=[m for m in tar.getmembers() if jpg.search(m.name)])
        inspection_files = [s for s in no_det_filelist if date[3:11] in s]
        for file in inspection_files:
            os.rename(os.path.join(temp_path, file), os.path.join(export_path, 'no_det', file))
    print('no_detections done')    

    for date in ice_det_datelist:
        tar = tarfile.open(os.path.join(tar_files_path,date), "r:gz")
        tar.extractall(temp_path, members=[m for m in tar.getmembers() if jpg.search(m.name)])
        inspection_files = [s for s in ice_det_filelist if date[3:11] in s]
        for file in inspection_files:
            os.rename(os.path.join(temp_path, file), os.path.join(export_path, 'ice_det', file))
    print('ice_detections done')

    for date in fog_det_datelist:
        tar = tarfile.open(os.path.join(tar_files_path,date), "r:gz")
        tar.extractall(temp_path, members=[m for m in tar.getmembers() if jpg.search(m.name)])
        inspection_files = [s for s in fog_det_filelist if date[3:11] in s]
        for file in inspection_files:
            os.rename(os.path.join(temp_path, file), os.path.join(export_path, 'fog_det', file))
    print('fog_detections done')
    shutil.rmtree(temp_path) 


def TrainingProgress(history, epochs, filename_path):
    """Shows and saves graph of training progress

    Parameters
    ----------
    history : model.fit output
        The output from the training of a model
    
    epochs : int
        The number of epochs trained in training session

    filename_path : str
        The filepath to where the graph is saved
    """
    import matplotlib.pyplot as plt

    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(filename_path)
    plt.show()


def PredictImage(path, Model):
    """Predicts the classification of a single image

    Parameters
    ----------
    path : str
        The filepath to the image to predict on
    
    Model : tf.keras.Model() model
        The model used to predict on image

    Outputs
    ----------
    prediction

    prediction : list of float
        List of prediction confidence of each class
    """
    from tensorflow import keras
    import numpy as np
    import tensorflow as tf

    x = keras.preprocessing.image.load_img(path)
    x = keras.preprocessing.image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = Model.predict(x)
    prediction = tf.nn.softmax(x)
    return prediction

def ShowPrediction(path, prediction, class_names):
    """Shows the prediction of a single image

    Parameters
    ----------
    path : str
        The filepath to the image to predict on
    
    prediction : PredictImage output
        The prediction made on the image

    class_names : list of str
        A list of the class names in correct order (use train_generator.class_indices 
        to see the right order)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    print(
        "This image most likely belongs to \n {} with a {:.2f} % confidence."
        .format(list(class_names.keys())[np.argmax(prediction)], 100 * np.max(prediction))
    )
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()