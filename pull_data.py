import numpy as np
import pandas as pd


def pull_data(comma_count, df_resolved_refined, df_averaged_refined, df_hr, df_voxel, voxel, vessel, final_voxel_list, resolved_vessel_list, resolved_value_list, averaged_vessel_list, averaged_value_list, fpath2):
    # account for index shift in df of -2
    index_shift = 1
    window = 2

    # calculate upper and lower bounds
    if comma_count == 0:
        # no commas i.e. only one voxel value input so get a contiguous voxel range
        lower_bound = int(voxel) - window - index_shift
        upper_bound = int(voxel) + window - index_shift
    elif comma_count > 1:
        # at least one comma, so parse the input to define the range
        split_words = voxel.split(', ')
        lower_bound = int(split_words[0])
        upper_bound = int(split_words[-1])

    # check if the chosen voxel range is in the final refined list
    # compare lists and list any missing voxels
    selected_voxel_list = []
    while lower_bound <= upper_bound:
        selected_voxel_list.append(lower_bound)
        lower_bound = lower_bound + 1
    missing_voxel_list = list(set(selected_voxel_list) - set(final_voxel_list))

####

    if len(missing_voxel_list) = 0:
        # no missing voxels
        # calc flow/min for time resolved files
        row_count = len(df.index)
        cols = ['Flow']
        df_flow = pd.DataFrame(columns=cols, index=range(row_count))

        # calculate hr
        hr = 60 / df_hr.iat[0, 20]

        # setup absolute hr vectors
        v1 = np.array(df_hr.iloc[0][2:21].astype(float))
        v2 = np.array(df_hr.iloc[0][1:20].astype(float))

        for row in range(0, row_count):
            v3 = np.array(df.iloc[row][2:21].astype(float))
            v4 = np.array(df.iloc[row][1:20].astype(float))
            v5 = v1 - v2
            v6 = v3 + v4
            v7 = v5 * v6
            flow_beat = sum(v7) * 0.5
            flow = flow_beat * hr
            df_flow.loc[row].Flow = flow


        df_flow.insert(loc=0, column='Voxel', value=final_voxel_list)
        df_flow.to_csv(str(vessel[0][0]) + '_time_resolved_analyzed.csv', sep=",", index=False)

        resolved_value_list.extend([np.mean(df_flow.iloc[lower_bound:upper_bound, 1])])
        resolved_vessel_list.append(vessel[0][0])

        # time averaged
        averaged_value_list.extend([np.mean(df_averaged_refined.loc[lower_bound:upper_bound][1:], axis=0)])
        averaged_vessel_list.append(vessel[0][0])

        return {'resolved_vessel_list': resolved_vessel_list, 'resolved_value_list': resolved_value_list,
                'averaged_vessel_list': averaged_vessel_list, 'averaged_value_list': averaged_value_list}
    else:
        # missing voxels
        # write any voxels that need to be adjusted to a .txt file
        str1 = str(vessel[0][0]) + ' voxel selection is not in the refined list of voxels. '
        str2 = 'Do not include the following voxels: ' + str(missing_voxel_list)
        msg = str1 + str2 + '\n'
        with open(fpath2, 'a') as f:
            f.write(msg)
        return None
