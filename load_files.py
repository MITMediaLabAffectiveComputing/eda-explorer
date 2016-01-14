import pandas as pd
import scipy.signal as scisig

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

def interpolateEmptyValues(data):
    cols = data.columns.values
    for c in cols:
        data[c] = data[c].interpolate()

    return data

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