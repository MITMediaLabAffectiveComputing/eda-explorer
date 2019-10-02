import numpy as np
import pandas as pd
import scipy.signal as scisig
import os
import matplotlib.pyplot as plt

from load_files import getInputLoadFile, getOutputPath, get_user_input

DEBUG = True

SAMPLING_RATE = 8

ONE_MINUTE_S = 60
THIRTY_MIN_S = ONE_MINUTE_S*30
SECONDS_IN_DAY = 24*60*60

STILLNESS_MOTION_THRESHOLD = .1
PERCENT_STILLNESS_THRESHOLD = .95

STEP_DIFFERENCE_THRESHOLD = 0.3



def computeAllAccelerometerFeatures(data, time_frames):
    if DEBUG: print("\t\tcomputing motion...")
    motion = computeMotion(data['AccelX'], data['AccelY'], data['AccelZ'])

    if DEBUG: print("\t\tcomputing steps...")
    steps = computeSteps(motion)

    if DEBUG: print("\t\tcomputing stillness...")
    stillness = computeStillness(motion)

    features = []

    for time_frame in time_frames:
        start = time_frame[0]
        end = time_frame[1]
        start1Hz = int(start / SAMPLING_RATE)
        end1Hz = end if end == -1 else int(end / SAMPLING_RATE)
        if DEBUG: print("\t\tcomputing features for time frame. Start index: "+ str(start)+ " end index: "+ str(end))

        time_frame_feats = computeAccelerometerFeaturesOverOneTimeFrame(motion[start:end],
                                                                        steps[start:end],
                                                                        stillness[start1Hz:end1Hz])
        features.append(time_frame_feats)

    return features, steps, motion

def computeMotion(acc1, acc2, acc3):
    '''Aggregates 3-axis accelerometer signal into a single motion signal'''
    return np.sqrt(np.array(acc1)**2 + np.array(acc2)**2 + np.array(acc3)**2)

def computeSteps(motion):
    '''Determines the location of steps from the aggregated accelerometer signal.
    Signal is low-pass filtered, then minimums are located in the signal. For each
    min, if the max absolute derivative (first difference) immediately surrounding
    it is greater than a threshold, it is counted as a step.

    Args:
        motion:		root mean squared 3 axis acceleration
    Returns:
        steps:		binary array at 8Hz which is 1 everywhere there is a step'''

    filtered_signal = filterSignalFIR(motion, 2, 256)
    diff = filtered_signal[1:]-filtered_signal[:-1]

    mins = scisig.argrelextrema(filtered_signal, np.less)[0]

    steps = [0] * len(filtered_signal)
    for m in mins:
        if m <= 4 or m >= len(diff) - 4:
            continue
        if max(abs(diff[m-4:m+4])) > STEP_DIFFERENCE_THRESHOLD:
            steps[m] = 1.0

    return steps

def filterSignalFIR(eda, cutoff=0.4, numtaps=64):
    f = cutoff/(SAMPLING_RATE/2.0)
    FIR_coeff = scisig.firwin(numtaps,f)

    return scisig.lfilter(FIR_coeff,1,eda)

def computeStillness(motion):
    '''Locates periods in which the person is still or motionless.
    Total acceleration must be less than a threshold for 95 percent of one
    minute in order for that minute to count as still

    Args:
        motion:		an array containing the root mean squared acceleration
    Returns:
        A 1Hz array that is 1 for each second belonging to a still period, 0 otherwise
    '''
    diff = motion[1:]-motion[:-1]
    momentary_stillness = diff < STILLNESS_MOTION_THRESHOLD
    np.append(momentary_stillness,0) # to ensure list is the same size as the full day signal
    num_minutes_in_day = 24*60

    #create array indicating whether person was still or not for each second of the day
    #to be still the momentary_stillness signal must be true for more than 95% of the minute
    #containing that second
    second_stillness = [0]*SECONDS_IN_DAY

    for i in range(num_minutes_in_day):
        hours_start = int(i / 60)
        mins_start = i % 60
        hours_end = int((i+1) / 60)
        mins_end = (i+1) % 60

        start_idx = getIndexFromTimestamp(hours_start, mins_start)
        end_idx = getIndexFromTimestamp(hours_end, mins_end)

        this_minute = momentary_stillness[start_idx:end_idx]
        minute_stillness = sum(this_minute) > PERCENT_STILLNESS_THRESHOLD*(60*SAMPLING_RATE)

        second_idx = int(start_idx/8)
        for si in range(second_idx,second_idx+60):
            second_stillness[si] = float(minute_stillness)

    return second_stillness

def computeAccelerometerFeaturesOverOneTimeFrame(motion, steps, stillness):
    ''' Computes all available features for a time period. Incoming signals are assumed to be from
    only that time period.

    Args:
        motion:						8Hz		root mean squared 3 axis acceleration
        steps:						8Hz		binary signal that is 1 if there is a step
        stillness:					1Hz		1 if the person was still during this second, 0 otherwise
    Returns:
        A list of features containing (in order):
        -Step count 								number of steps detected
        -mean step time during movement				average number of samples between two steps (aggregated first to 1 minute,
                                                    then we take the mean of only the parts of this signal occuring during movement)
        -percent stillness 							percentage of time the person spent nearly motionless
    '''

    features = []

    features.extend(computeStepFeatures(steps,stillness))
    features.append(countStillness(stillness))

    return features

def computeStepFeatures(steps,stillness):
    '''Counts the total number of steps over a given period,
    as well as the average time between steps (meant to approximate walking speed)

    Args:
        steps:	an binary array at 8 Hz that is 1 every time there is a step
    Returns:
        sum: 			the number of steps in a period
        median time: 	average number of samples between two steps'''

    sum_steps = float(sum(steps))

    step_indices = np.nonzero(steps)[0]
    diff = step_indices[1:]-step_indices[:-1]

    #ensure length of step difference array is the same so we can get the actual locations of step differences
    timed_step_diff = np.empty(len(steps)) * np.nan
    timed_step_diff[step_indices[:len(diff)]] = diff

    signal_length_1s = len(stillness)
    signal_length_1min = int(signal_length_1s / 60)

    # if there aren't enough steps during this period, cannot accurately compute mean step diff
    if len(timed_step_diff) < signal_length_1min:
        return [sum_steps, np.nan]

    agg_stillness = aggregateSignal(stillness, signal_length_1min, 'max')
    agg_step_diff = aggregateSignal(timed_step_diff, signal_length_1min, 'mean')

    movement_indices = [i for i in range(len(agg_stillness)) if agg_stillness[i] == 0.0]
    step_diff_during_movement = agg_step_diff[movement_indices]

    return [sum_steps,round(np.nanmean(step_diff_during_movement),10)]

def countStillness(stillness):
    '''Counts the total percentage of time spent still over a period

    Args:
        stillness:	an binary array at 1Hz that is 1 if that second is part of a still period
    Returns:
        the percentage time spent still over a period'''

    return float(sum(stillness)) / float(len(stillness))

def aggregateSignal(signal, new_signal_length, agg_method='sum'):
    new_signal = np.zeros(new_signal_length)
    samples_per_bucket = int(len(signal) / new_signal_length)

    #the new signal length must be large enough that there is at least 1 sample per bucket
    assert(samples_per_bucket > 0)

    for i in range(new_signal_length):
        if agg_method == 'sum':
            new_signal[i] = np.nansum(signal[i*samples_per_bucket:(i+1)*samples_per_bucket])
        elif agg_method == 'percent':
            new_signal[i] = np.nansum(signal[i*samples_per_bucket:(i+1)*samples_per_bucket]) / samples_per_bucket
        elif agg_method == 'mean':
            new_signal[i] = np.nanmean(signal[i*samples_per_bucket:(i+1)*samples_per_bucket])
        elif agg_method == 'max':
            new_signal[i] = np.nanmax(signal[i*samples_per_bucket:(i+1)*samples_per_bucket])
    return new_signal

def getIndexFromTimestamp(hours, mins=0): 
    return ((hours * 60) + mins) * 60 * SAMPLING_RATE

def inputTimeFrames():
    '''Allows user to choose the time frames over which they compute accelerometer features.'''

    time_frames = []
    print("Accelerometer features can be extracted over different time periods.")
    cont = get_user_input("If you would like to enter a time period over which to compute features, enter 'y', or press enter to compute features over the entire file.")
    while cont == 'y' or cont == 'Y':
        start = int(get_user_input("Enter the starting hour of the time period (hour 0 is when the file starts):"))
        end = int(get_user_input("Enter the ending hour of the time period (hour 0 is when the file starts; use -1 for the end of the file):"))
        start = getIndexFromTimestamp(int(start))
        if end != -1:
            end = getIndexFromTimestamp(int(end))
        time_frames.append([start,end])
        print("Great! Now computing features for the following time periods:"+ str(time_frames))
        cont = get_user_input("To add another time period, enter 'y'. To finish, press enter.")

    if len(time_frames) == 0:
        time_frames = [[0,-1]] # the whole file

    return time_frames

def saveFeaturesToFile(features, time_frames, output_file):
    of = open(output_file, 'w')
    of.write("Time period start hour, Time period end hour, Step count, Mean step time during movement, Percent stillness\n")
    tf_i = 0
    for tf in time_frames:
        output_str = str(tf[0]) + ' , ' + str(tf[1])
        for feat in features[tf_i]:
            output_str += ' , ' + str(feat)
        tf_i += 1
        of.write(output_str + '\n')
    of.close()
    print("Saved features to file"+ output_file)

# draws a graph of the data with the peaks marked on it
# assumes that 'data' dataframe already contains the 'peaks' column
def plotSteps(data, x_seconds, sampleRate = SAMPLING_RATE):
    if x_seconds:
        time_m = np.arange(0,len(data))/float(sampleRate)
        realign = 128/(sampleRate)
    else:
        time_m = np.arange(0,len(data))/(sampleRate*60.)
        realign = 128/(sampleRate*60.)

    data_min = data['motion'].min()
    data_max = data['motion'].max()

    #Plot the data with the Peaks marked
    plt.figure(1,figsize=(20, 5))

    plt.plot(time_m,data['motion'])

    for i in range(len(data)):
        if data.iloc[i]["steps"]==1:
            x_loc = time_m[i] - realign
            plt.plot([x_loc,x_loc],[data_min,data_max],"k")
    step_height = data_max * 1.15
    #data['steps_plot'] = data['steps'] * step_height
    #plt.plot(time_m,data['steps_plot'],'k')

    plt.xlim([0,time_m[-1]])
    plt.ylim([data_min-.1,data_max+.1])
    plt.title('Motion with Detected "Steps" marked')
    plt.ylabel('g')
    if x_seconds:
        plt.xlabel('Time (s)')
    else:
        plt.xlabel('Time (min)')

    plt.show()

if __name__ == "__main__":
    print("This script will extract features related to accelerometer data.")

    data, filepath_confirm = getInputLoadFile()

    output_path = getOutputPath()

    time_frames = inputTimeFrames()

    features, steps, motion = computeAllAccelerometerFeatures(data, time_frames)

    data["steps"] = steps
    data["motion"] = motion

    saveFeaturesToFile(features, time_frames, output_path)

    print("")
    plot_ans = get_user_input("Do you want to plot the detected steps? (y/n): ")
    if 'y' in plot_ans:
        secs_ans = get_user_input("Would you like the x-axis to be in seconds or minutes? (sec/min): ")
        if 'sec' in secs_ans:
            x_seconds=True
        else:
            x_seconds=False
        plotSteps(data, x_seconds)
    else:
        print("\tOkay, script will not produce a plot")

