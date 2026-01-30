import os
import numpy as np
from datetime import datetime, timedelta

# Using sliding window approach to partition model inputs and ground truth data
if __name__ == "__main__":
    # Set the path for the MeteoNet npy file and the output file path
    examples_root = '/path/to/meteonet/data/nw/reflectivity_npy/'
    out_root = '/path/to/meteonet/data/nw/reflectivity_split/'

    # Set the start and end times; the time periods vary by year
    begin_time = datetime(2016, 1, 1, 0, 0)
    # begin_time = datetime(2017, 1, 1, 0, 0)
    # begin_time = datetime(2018, 1, 1, 0, 0)
    end_time = datetime(2016, 12, 31, 23, 55)
    # end_time = datetime(2017, 12, 31, 23, 55)
    # end_time = datetime(2018, 10, 31, 23, 55)

    # The task is set to predict 15 frames (75 minutes) based on 5 frames
    # with the time incrementing by 75 minutes each time.
    while begin_time + timedelta(minutes=75) <= end_time:
        # item_time records the time when the file was last read.
        item_time = begin_time
        print("begin_time: ", begin_time)

        file_name = begin_time.strftime("%Y%m%d_%H%M")
        flag = 0
        radar_data = []
        # 每个文件一共包含20帧
        for j in range(20):
            if os.path.exists(examples_root + item_time.strftime("%Y%m%d_%H%M") + '.npy'):
                d = np.load(examples_root + item_time.strftime("%Y%m%d_%H%M") + '.npy')

                # If the mean value of Frame 6 (the starting frame) is less than the precipitation threshold, skip that time point
                if j == 5 and d.mean() < 0.001:
                    flag = 1
                    break

                radar_data.append(d)
                # Add 5 minutes each time
                item_time = item_time + timedelta(minutes=5)
            # If the file is still missing after adding 5 minutes, it indicates data loss at that time point; skip that time point
            else:
                flag = 1
                break
        # If terminated early, the sliding window shifts by 5 frames (25 minutes).
        if flag:
            # Note: The start time needs to be updated here.
            begin_time = begin_time + timedelta(minutes=25)
            continue

        # Save as npy file
        file_npy = np.stack(radar_data, axis=0)
        base = out_root + file_name
        np.save(os.path.join(out_root, f"{base}.npy"), file_npy)

        begin_time = begin_time + timedelta(minutes=75)