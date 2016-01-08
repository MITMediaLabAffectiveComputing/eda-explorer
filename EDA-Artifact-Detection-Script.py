import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal as scisig
import imp
import pywt
import os


matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True



def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y

def interpolateEmptyValues(data):
    cols = data.columns.values
    for c in cols:
        data[c] = data[c].interpolate()

    return data

def interpolateDataTo8Hz(data,sample_rate,startTime):

    if sample_rate<8:
        # Upsample by linear interpolation
        if sample_rate==2:
            data.index = pd.DatetimeIndex(start=startTime,periods = len(data),freq='500L')
        elif sample_rate==4:
            data.index = pd.DatetimeIndex(start=startTime,periods = len(data),freq='250L')
        data = data.resample("125L")
    else:
        if sample_rate>8:
            # Downsample
            idx_range = range(0,len(data))
            data = data.iloc[idx_range[0::sample_rate/8]]
        # Set the index to be 8Hz
        data.index = pd.DatetimeIndex(start=startTime,periods = len(data),freq='125L')

    # Interpolate all empty values
    data = interpolateEmptyValues(data)
    return data



def loadData_Qsensor(filepath):
    '''
    This function loads the Q sensor data, uses a lowpass butterworth filter on the EDA signal, and computes the wavelet coefficients
    Note: currently assumes sampling rate of 8hz, 16hz, 32hz; if sampling rate is 16hz or 32hz the signal is downsampled

    INPUT:
        filepath:       string, path to input file

    OUTPUT:
        data:           DataFrame, index is a list of timestamps at 8Hz, columns include AccelZ, AccelY, AccelX, Temp, EDA, filtered_eda
    '''
    # Get header info
    try:
        header_info = pd.io.parsers.read_csv(filepath, nrows=5)
    except IOError:
        print "Error!! Couldn't load file, make sure the filepath is correct and you are using a csv from the q sensor software"
        print 
        print
        return

    # Get sample rate
    sampleRate = int((header_info.iloc[3,0]).split(":")[1].strip())

    # Get the raw data
    data = pd.io.parsers.read_csv(filepath, skiprows=7)
    data = data.reset_index()

    # Reset the index to be a time and reset the column headers
    data.columns = ['AccelZ','AccelY','AccelX','Battery','Temp','EDA']

    # Get Start Time
    startTime = pd.to_datetime(header_info.iloc[4,0][12:-10])
    
    # Make sure data has a sample rate of 8Hz
    data = interpolateDataTo8Hz(data,sampleRate,startTime)
    
    # Remove Battery Column
    data = data[['AccelZ','AccelY','AccelX','Temp','EDA']]

    # Get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:8hz, order:6)
    data['filtered_eda'] =  butter_lowpass_filter(data['EDA'], 1.0, 8, 6)

    return data

def loadData_E4(filepath):
    # Load data
    data = pd.DataFrame.from_csv(os.path.join(filepath,'EDA.csv'))
    data.reset_index(inplace=True)
    
    # Get the startTime and sample rate
    startTime = pd.to_datetime(float(data.columns.values[0]),unit="s")
    sampleRate = float(data.iloc[0][0])
    data = data[data.index!=0]
    data.index = data.index-1
    
    # Reset the data frame assuming 4Hz samplingRate
    data.columns = ['EDA']
    if sampleRate !=4:
        print 'ERROR, NOT SAMPLED AT 4HZ. PROBLEMS WILL OCCUR\n'
    data.index = pd.DatetimeIndex(start=startTime,periods = len(data),freq='250L')

    # Make sure data has a sample rate of 8Hz
    data = interpolateDataTo8Hz(data,sampleRate,startTime)

    # Get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:8hz, order:6)
    data['filtered_eda'] =  butter_lowpass_filter(data['EDA'], 1.0, 8, 6)
    
    return data

def loadData_getColNames(data_columns):
    print "Here are the data columns of your file: "
    print data_columns

    # Find the column names for each of the 5 data streams
    colnames = ['EDA data','Temperature data','Acceleration X','Acceleration Y','Acceleration Z']
    new_colnames = ['','','','','']

    for i in range(len(new_colnames)):
        new_colnames[i] = raw_input("Column name that contains "+colnames[i]+": ")
        while (new_colnames[i] not in data_columns):
            print "Column not found. Please try again"
            print "Here are the data columns of your file: "
            print data_columns

            new_colnames[i] = raw_input("Column name that contains "+colnames[i]+": ")

    # Get user input on sample rate
    sampleRate = raw_input("Enter sample rate (must be an integer power of 2): ")
    while (sampleRate.isdigit()==False) or (np.log(int(sampleRate))/np.log(2) != np.floor(np.log(int(sampleRate))/np.log(2))):
        print "Not an integer power of two"
        sampleRate = raw_input("Enter sample rate (must be a integer power of 2): ")
    sampleRate = int(sampleRate)

    # Get user input on start time
    startTime = pd.to_datetime(raw_input("Enter a start time (format: YYYY-MM-DD HH:MM:SS): "))
    while type(startTime)==str:
        print "Not a valid date/time"
        startTime = pd.to_datetime(raw_input("Enter a start time (format: YYYY-MM-DD HH:MM:SS): "))


    return sampleRate, startTime, new_colnames



def loadData_misc(filepath):
    # Load data
    data = pd.DataFrame.from_csv(filepath)

    # Get the correct colnames
    sampleRate, startTime, new_colnames = loadData_getColNames(data.columns.values)

    data.rename(columns=dict(zip(new_colnames,['EDA','Temp','AccelX','AccelY','AccelZ'])), inplace=True)
    data = data[['AccelZ','AccelY','AccelX','Temp','EDA']]

    # Make sure data has a sample rate of 8Hz
    data = interpolateDataTo8Hz(data,sampleRate,startTime)

    # Get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:8hz, order:6)
    data['filtered_eda'] =  butter_lowpass_filter(data['EDA'], 1.0, 8, 6)

    return data

def getWaveletData(data):
    '''
    This function computes the wavelet coefficients

    INPUT:
        data:           DataFrame, index is a list of timestamps at 8Hz, columns include EDA, filtered_eda

    OUTPUT:
        wave1Second:    DateFrame, index is a list of timestamps at 1Hz, columns include OneSecond_feature1, OneSecond_feature2, OneSecond_feature3 
        waveHalfSecond: DateFrame, index is a list of timestamps at 2Hz, columns include HalfSecond_feature1, HalfSecond_feature2 
    '''
    startTime = data.index[0]

    # Create wavelet dataframes
    oneSecond = pd.DatetimeIndex(start=startTime, periods=len(data), freq='1s')
    halfSecond = pd.DatetimeIndex(start=startTime, periods=len(data), freq='500L')

    # Compute wavelets
    cA_n, cD_3, cD_2, cD_1 = pywt.wavedec(data['EDA'], 'Haar', level=3) #3 = 1Hz, 2 = 2Hz, 1=4Hz
    
    # Wavelet 1 second window
    N = int(len(data)/8)
    coeff1 = np.max(abs(np.reshape(cD_1[0:4*N],(N,4))), axis=1)
    coeff2 = np.max(abs(np.reshape(cD_2[0:2*N],(N,2))), axis=1)
    coeff3 = abs(cD_3[0:N])
    wave1Second = pd.DataFrame({'OneSecond_feature1':coeff1,'OneSecond_feature2':coeff2,'OneSecond_feature3':coeff3})
    wave1Second.index = oneSecond[:len(wave1Second)]
    
    # Wavelet Half second window
    N = int(np.floor((len(data)/8.0)*2))
    coeff1 = np.max(abs(np.reshape(cD_1[0:2*N],(N,2))),axis=1)
    coeff2 = abs(cD_2[0:N])
    waveHalfSecond = pd.DataFrame({'HalfSecond_feature1':coeff1,'HalfSecond_feature2':coeff2})
    waveHalfSecond.index = halfSecond[:len(waveHalfSecond)]

    return wave1Second,waveHalfSecond

def getDerivatives(eda):
    deriv = (eda[1:-1] + eda[2:])/ 2. - (eda[1:-1] + eda[:-2])/ 2.
    second_deriv = eda[2:] - 2*eda[1:-1] + eda[:-2]
    return deriv,second_deriv

def get3MaxDerivatives(eda,num_max=3):
    deriv, second_deriv = getDerivatives(eda)
    d = copy.deepcopy(deriv)
    d2 = copy.deepcopy(second_deriv)
    max_indices = []
    for i in range(num_max):
        maxd_idx = np.nanargmax(abs(d))
        max_indices.append(maxd_idx)
        d[maxd_idx] = 0
        max2d_idx = np.nanargmax(abs(d2))
        max_indices.append(max2d_idx)
        d2[max2d_idx] = 0
    
    return max_indices, abs(deriv), abs(second_deriv)

def getDerivStats(eda):
    deriv, second_deriv = getDerivatives(eda)
    maxd = max(deriv)
    mind = min(deriv)
    maxabsd = max(abs(deriv))
    avgabsd = np.mean(abs(deriv))
    max2d = max(second_deriv)
    min2d = min(second_deriv)
    maxabs2d = max(abs(second_deriv))
    avgabs2d = np.mean(abs(second_deriv))
    
    return maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d

def getStats(data):
    eda = data['EDA'].as_matrix()
    filt = data['filtered_eda'].as_matrix()
    maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d = getDerivStats(eda)
    maxd_f,mind_f,maxabsd_f,avgabsd_f,max2d_f,min2d_f,maxabs2d_f,avgabs2d_f = getDerivStats(filt)
    amp = np.mean(eda)
    amp_f = np.mean(filt)
    return amp, maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d,amp_f,maxd_f,mind_f,maxabsd_f,avgabsd_f,max2d_f,min2d_f,maxabs2d_f,avgabs2d_f

def computeWaveletFeatures(waveDF):
    maxList = waveDF.max().tolist()
    meanList = waveDF.mean().tolist()
    stdList = waveDF.std().tolist()
    medianList = waveDF.median().tolist()
    aboveZeroList = (waveDF[waveDF>0]).count().tolist()

    return maxList,meanList,stdList,medianList,aboveZeroList

def getWavelet(wave1Second,waveHalfSecond):
    max_1,mean_1,std_1,median_1,aboveZero_1 = computeWaveletFeatures(wave1Second)
    max_H,mean_H,std_H,median_H,aboveZero_H = computeWaveletFeatures(waveHalfSecond)
    return max_1,mean_1,std_1,median_1,aboveZero_1,max_H,mean_H,std_H,median_H,aboveZero_H

def getFeatures(data,w1,wH):
    # Get DerivStats
    amp,maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d,amp_f,maxd_f,mind_f,maxabsd_f,avgabsd_f,max2d_f,min2d_f,maxabs2d_f,avgabs2d_f = getStats(data)
    statFeat = np.hstack([amp,maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d,amp_f,maxd_f,mind_f,maxabsd_f,avgabsd_f,max2d_f,min2d_f,maxabs2d_f,avgabs2d_f])

    # Get Wavelet Features
    max_1,mean_1,std_1,median_1,aboveZero_1,max_H,mean_H,std_H,median_H,aboveZero_H = getWavelet(w1,wH)
    waveletFeat = np.hstack([max_1,mean_1,std_1,median_1,aboveZero_1,max_H,mean_H,std_H,median_H,aboveZero_H])

    all_feat = np.hstack([statFeat,waveletFeat])
    
    if np.Inf in all_feat:
        print "Inf"
    
    if np.NaN in all_feat:
        print "NaN"

    return list(all_feat)

def createFeatureDF(data):
    '''
    INPUTS:
        filepath:           string, path to input file  
    OUTPUTS:
        features:           DataFrame, index is a list of timestamps for each 5 seconds, contains all the features
        data:               DataFrame, index is a list of timestamps at 8Hz, columns include AccelZ, AccelY, AccelX, Temp, EDA, filtered_eda
    '''
    # Load data from q sensor
    wave1sec,waveHalf = getWaveletData(data)
    
    # Create 5 second timestamp list
    timestampList = data.index.tolist()[0::40]
    
    # feature names for DataFrame columns
    allFeatureNames = ['raw_amp','raw_maxd','raw_mind','raw_maxabsd','raw_avgabsd','raw_max2d','raw_min2d','raw_maxabs2d','raw_avgabs2d','filt_amp','filt_maxd','filt_mind',
        'filt_maxabsd','filt_avgabsd','filt_max2d','filt_min2d','filt_maxabs2d','filt_avgabs2d','max_1s_1','max_1s_2','max_1s_3','mean_1s_1','mean_1s_2','mean_1s_3',
        'std_1s_1','std_1s_2','std_1s_3','median_1s_1','median_1s_2','median_1s_3','aboveZero_1s_1','aboveZero_1s_2','aboveZero_1s_3','max_Hs_1','max_Hs_2','mean_Hs_1',
        'mean_Hs_2','std_Hs_1','std_Hs_2','median_Hs_1','median_Hs_2','aboveZero_Hs_1','aboveZero_Hs_2']

    # Initialize Feature Data Frame
    features = pd.DataFrame(np.zeros((len(timestampList),len(allFeatureNames))),columns=allFeatureNames,index=timestampList)
    
    # Compute features for each 5 second epoch
    for i in range(len(features)-1):
        start = features.index[i]
        end = features.index[i+1]
        this_data = data[start:end]
        this_w1 = wave1sec[start:end]
        this_w2 = waveHalf[start:end]
        features.iloc[i] = getFeatures(this_data,this_w1,this_w2)
    return features

def classifyEpochs(features,featureNames,svmClassifierPath):
    '''
    This function takes the full features DataFrame and classifies each 5 second epoch into artifact, questionable, or clean

    INPUTS:
        features:           DataFrame, index is a list of timestamps for each 5 seconds, contains all the features
        featureNames:       list of Strings, subset of feature names needed for classification
        svmClassifierPath:  string, path to pickled SVM 

    OUTPUTS:
        labels:             Series, index is a list of timestamps for each 5 seconds, values of -1, 0, or 1 for artifact, questionable, or clean
    '''
    # Only get relevant features
    features = features[featureNames]
    
    # Load classifier
    classifier = svm.SVM()
    classifier.loadClassifierFromFile(svmClassifierPath)
    
    # 
    X = features[featureNames].as_matrix()
    
    # Classify each 5 second epoch and put into DataFrame
    featuresLabels = classifier.predict(X)
    return featuresLabels

def getSVMPickle(key):
    '''
    This returns the name of the pickledSVM and the list of relevant features

    INPUT:
        key:                string, either "Binary" or "Multiclass"

    OUTPUT:
        svmPickleName:      string, filename to pickled SVM 
        featureList:        list of Strings, subset of feature names needed for classification
    '''
    if key == "Binary":
        return "SVMBinary.p",['raw_amp','raw_maxabsd','raw_max2d','raw_avgabs2d','filt_amp','filt_min2d','filt_maxabs2d','max_1s_1',
                                'mean_1s_1','std_1s_1','std_1s_2','std_1s_3','median_1s_3']
    elif key == "Multiclass":
        return "SVMMulticlass.p",['filt_maxabs2d','filt_min2d','std_1s_1','raw_max2d','raw_amp','max_1s_1','raw_maxabs2d','raw_avgabs2d',
                                    'filt_max2d','filt_amp']
    else:
        print 'Error!! Invalid key, choose "Binary" or "Multiclass"'
        print 
        print 
        return
    
def classify(filepath,classifierList,pickleDirectory,loadDataFunction):
    '''
    This function wraps other functions in order to load, classify, and return the label for each 5 second epoch of Q sensor data.

    INPUT:
        filepath:               string, path to input file          
        classifierKey:          list of strings, either "Binary" or "Multiclass"
        pickleDirectory:        string, path to pickle directory
        loadDataFunction:       function, loads sensor data and returns data at 8Hz in a pandas DataFrame indexed by timestamp and at least has 'EDA' column and 'filtered_eda' column
    OUTPUT:
        featureLabels:          Series, index is a list of timestamps for each 5 seconds, values of -1, 0, or 1 for artifact, questionable, or clean
        data:                   DataFrame, only output if fullFeatureOutput=1, index is a list of timestamps at 8Hz, columns include AccelZ, AccelY, AccelX, Temp, EDA, filtered_eda
    '''
    # Constants
    oneHour = 8*60*60 # 8(samp/s)*60(s/min)*60(min/hour) = samp/hour
    fiveSec = 8*5

    # Load data
    data = loadDataFunction(filepath)

    # Get pickle List and featureNames list
    pickleNameList = ['']*len(classifierList)
    featureNameList = [[]]*len(classifierList)
    for i in range(len(classifierList)):
        pickleName, featureNames = getSVMPickle(classifierList[i])
        pickleNameList[i]=pickleName
        featureNameList[i]=featureNames

    # Get the number of data points, hours, and labels
    rows = len(data)
    num_labels = int(np.ceil(float(rows)/fiveSec))
    hours = int(np.ceil(float(rows)/oneHour))

    # Initialize labels array
    labels = -1*np.ones((num_labels,len(classifierList)))

    for h in range(hours):
        # Get a data slice that is at most 1 hour long
        start = h*oneHour
        end = min((h+1)*oneHour,rows)
        cur_data = data[start:end]

        features = createFeatureDF(cur_data)

        for i in range(len(classifierList)):
            # Get correct feature names for classifier
            pickleName = pickleNameList[i]
            featureNames = featureNameList[i]
            
            # Label each 5 second epoch
            temp_labels = classifyEpochs(features,featureNames,os.path.join(pickleDirectory,pickleName))
            labels[(h*12*60):(h*12*60+temp_labels.shape[0]),i] = temp_labels

    return labels,data

def plotData(data,labels,classifierList,filteredPlot=0,secondsPlot=0):
    '''
    This function plots the Q sensor EDA data with shading for artifact (red) and questionable data (grey). 
        Note that questionable data will only appear if you choose a multiclass classifier

    INPUT:
        data:                   DataFrame, indexed by timestamps at 8Hz, columns include EDA and filtered_eda
        labels:                 array, each row is a 5 second period and each column is a different classifier
        filteredPlot:           binary, 1 for including filtered EDA in plot, 0 for only raw EDA on the plot, defaults to 0
        secondsPlot:            binary, 1 for x-axis in seconds, 0 for x-axis in minutes, defaults to 0

    OUTPUT:
        [plot]                  the resulting plot has N subplots (where N is the length of classifierList) that have linked x and y axes 
                                    and have shading for artifact (red) and questionable data (grey)

    '''
    
    # Initialize x axis
    if secondsPlot:
        scale = 1.0
    else:
        scale = 60.0
    time_m = np.arange(0,len(data))/(8.0*scale)
    
    # Initialize Figure
    plt.figure(figsize=(10,5))

    # For each classifier, label each epoch and plot
    for k in range(np.shape(labels)[1]):
        key = classifierList[k]
        
        # Initialize Subplots
        if k==0:
            ax = plt.subplot(len(classifierList),1,k+1)
        else:
            ax = plt.subplot(len(classifierList),1,k+1,sharex=ax,sharey=ax)

        # Plot EDA
        ax.plot(time_m,data['EDA'])

        # For each epoch, shade if necessary
        for i in range(0,len(labels)-1):
            if labels[i,k]==-1:
                # artifact
                start = i*40/(8.0*scale)
                end = start+5.0/scale
                ax.axvspan(start, end, facecolor='red', alpha=0.7, edgecolor ='none')
            elif labels[i,k]==0:
                # Questionable
                start = i*40/(8.0*scale)
                end = start+5.0/scale
                ax.axvspan(start, end, facecolor='.5', alpha=0.5,edgecolor ='none')

        # Plot filtered data if requested
        if filteredPlot:
            ax.plot(time_m-.625/scale,data['filtered_eda']) 
            plt.legend(['Raw SC','Filtered SC'],loc=0)

        # Label and Title each subplot
        plt.ylabel('$\mu$S')
        plt.title(key)
    
    # Only include x axis label on final subplot
    if secondsPlot:
        plt.xlabel('Time (s)')    
    else:
        plt.xlabel('Time (min)')

    # Display the plot
    plt.show()
    return

if __name__ == "__main__":
    pickleDirectory = raw_input('Pickle Directory (type ./ for current directory): ')

    # Load SVM Directory
    svmFilePath = os.path.join(pickleDirectory,'classify.py')
    print "Loading SVM file from "+ svmFilePath
    svm = imp.load_source('SVM',svmFilePath)

    numClassifiers = int(raw_input('Would you like 1 classifier (Binary or Multiclass) or both (enter 1 or 2): '))

    # Create list of classifiers
    if numClassifiers==1:
        classifierList= [raw_input("Name of classifier (Binary or Multiclass): ")]
    else:
        classifierList = ['Binary','Multiclass']

    # Classify the data
    dataType = raw_input("Data Type (e4 or q or misc): ")
    if dataType=='q':
        filepath = raw_input("Filepath: ")
        print "Classifying data for " + filepath
        labels,data = classify(filepath,classifierList,pickleDirectory,loadData_Qsensor)
    elif dataType=='e4':
        filepath = raw_input("Path to E4 directory: ")
        print "Classifying data for " + os.path.join(filepath,"EDA.csv")
        labels,data = classify(filepath,classifierList,pickleDirectory,loadData_E4)
    elif dataType=="misc":
        filepath = raw_input("Filepath: ")
        print "Classifying data for " + filepath
        labels,data = classify(filepath,classifierList,pickleDirectory,loadData_misc)
    else:
        print "We currently don't support that type of file."
         



    # Plotting the data
    plotDataInput = raw_input('Do you want to plot the labels? (y/n): ')

    if plotDataInput=='y':
        # Include filter plot?
        filteredPlot = raw_input('Would you like to include filtered data in your plot? (y/n): ')
        if filteredPlot=='y':
            filteredPlot=1
        else:
            filteredPlot=0

        # X axis in seconds?
        secondsPlot = raw_input('Would you like the x-axis to be in seconds or minutes? (sec/min): ')
        if secondsPlot=='sec':
            secondsPlot=1
        else:
            secondsPlot=0

        # Plot Data
        plotData(data,labels,classifierList,filteredPlot,secondsPlot)

        print "Remember! Red is for epochs with artifact, grey is for epochs that are questionable, and no shading is for clean epochs"
    

    # Saving the data
    saveDataInput = raw_input('Do you want to save the labels? (y/n): ')
    
    if saveDataInput=='y':
        outputPath = raw_input('Output directory: ')
        outputLabelFilename= raw_input('Output filename: ')

        # Save labels
        fullOutputPath = os.path.join(outputPath,outputLabelFilename)
        if fullOutputPath[-4:] != '.csv':
            fullOutputPath = fullOutputPath+'.csv'

        featureLabels = pd.DataFrame(labels,index=pd.DatetimeIndex(start=data.index[0],periods=len(labels),freq='5s'),columns=classifierList)

        featureLabels.to_csv(fullOutputPath)

        print "Labels saved to "+ fullOutputPath
        #print "Remember! The first column is timestamps and the second column is the labels (-1 for artifact, 0 for questionable, 1 for clean)"
    

    print '--------------------------------'
    print "Please also cite this project:"
    print "Taylor, S., Jaques, N., Chen, W., Fedor, S., Sano, A., & Picard, R. Automatic identification of artifacts in electrodermal activity data. In Engineering in Medicine and Biology Conference. 2015"
    print '--------------------------------'






