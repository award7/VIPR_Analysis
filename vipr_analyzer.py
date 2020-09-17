import argparse
import csv
import errno
import glob
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
import shutil
from xlrd import open_workbook


# TODO: install new version of python 3.8

def organize(vipr_path, target_dirs):
    # mv and copy files to new dirs
    raw_dir = 'raw'
    proc_dir = 'proc'
    graph_dir = 'graphs'
    os.makedirs(os.path.join(vipr_path, raw_dir), exist_ok=True)
    os.makedirs(os.path.join(vipr_path, proc_dir), exist_ok=True)
    os.makedirs(os.path.join(vipr_path, graph_dir), exist_ok=True)

    file_list = glob.glob(os.path.join(vipr_path, 'Summary.xls')) + glob.glob(os.path.join(vipr_path, 'voxels.csv'))
    for file in file_list:
        shutil.move(os.path.join(vipr_path, file), os.path.join(vipr_path, raw_dir))


def missing_voxels(voxel, comma_count, final_voxel_list):
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
    lower_bound_itr = lower_bound
    while lower_bound_itr <= upper_bound:
        selected_voxel_list.append(lower_bound)
        lower_bound_itr += 1
    missing_voxel_list = list(set(selected_voxel_list) - set(final_voxel_list))
    return missing_voxel_list, lower_bound, upper_bound


def noncontig_voxels(windows):
    # remove non-contiguous voxels
    list1 = []
    for lis in windows:
        window_total = lis[0] + lis[1] + lis[2]
        if lis[0] + (lis[0] + 1) + (lis[0] + 2) == window_total:
            list1.append(lis)
    list2 = [lis[0] for lis in list1]
    list3 = [lis[1] for lis in list1]
    list4 = [lis[2] for lis in list1]
    list5 = list2 + list3 + list4
    revised_list = list(dict.fromkeys(list5))
    revised_list.sort()
    return revised_list


def refined_to_csv(vipr_path, df_raw, input_file, final_voxel_list):
    # write refined df to new csv file
    df_refined = df_raw[df_raw['Voxel'].isin(final_voxel_list)]
    rename_file = input_file.replace("_raw.csv", "_refined.csv")
    rename_file = os.path.join(vipr_path, rename_file)
    df_refined.to_csv(rename_file, sep=",", index=False)
    return df_refined


def rm_outliers(df):
    threshold = 3
    df = df[(np.abs(stats.zscore(df)) < threshold).all(axis=1)]
    df_rm_outliers = df[(df > 0).all(1)]
    return df_rm_outliers


def sliding_window(seq, n=3):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def split_workbook(vipr_path):
    # split summary.xls tabs into individual .csv files
    wb = open_workbook(os.path.join(vipr_path, 'Summary.xls'))
    for i in range(3, wb.nsheets):
        sheet = wb.sheet_by_index(i)
        fname = "%s_raw.csv" %(sheet.name.replace(" ","_"))
        with open(os.path.join(vipr_path, fname), "w", newline = '') as file:
            writer = csv.writer(file, delimiter = ",")
            header = [cell.value for cell in sheet.row(0)]
            writer.writerow(header)
            for row_idx in range(1, sheet.nrows):
                row = [cell.value for cell in sheet.row(row_idx)]
                writer.writerow(row)


def sumproduct(df, df_hr):
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
    return df_flow


def time_averaged_plots(save_path, vessel, df_averaged_raw, df_averaged_refined):
    #will create a combined plot of the raw and refined data for each variable

    #loop over each variable



    for i in range(1,8):
        plt.figure()

        # raw plot
        var_column = i
        df_averaged_raw_plot = df_averaged_raw.iloc[:, var_column]
        df_averaged_raw_plot = df_averaged_raw_plot.T
        df_averaged_raw_plot.plot(legend=False)

        # final refined plot
        df_averaged_refined_plot = df_averaged_refined.iloc[:,var_column]
        df_averaged_refined_plot = df_averaged_refined_plot.T
        df_averaged_refined_plot.plot(legend=False)

        # save plot
        fname = '{}_{}.{}'.format(vessel, 'time_averaged', 'png')
        fname = os.path.join(save_path, fname)
        plt.savefig(fname)
        plt.clf()
        plt.close()

def time_resolved_plots(save_path, vessel, df_resolved_raw, df_resolved_refined):
    # time resolved plots

    plt.figure()

    # raw plot
    df_resolved_raw_plot = df_resolved_raw.iloc[:, 1:]
    df_resolved_raw_plot = df_resolved_raw_plot.T
    plt.subplot()
    plt.xlabel('Voxel')
    plt.ylabel('Flow (mL/s)')
    df_resolved_raw_plot.plot(legend=False)

    # final refined plot
    df_resolved_refined_plot = df_resolved_refined.iloc[:, 1:]
    df_resolved_refined_plot = df_resolved_refined_plot.T
    plt.subplot()
    plt.xlabel('Voxel')
    plt.ylabel('Flow (mL/s)')
    df_resolved_refined_plot.plot(legend=False)

    # save plot
    fname = '{}_{}.{}'.format(vessel, 'time_resolved', 'png')
    fname = os.path.join(save_path, fname)
    plt.savefig(fname)
    plt.clf()
    plt.close()


def main(src_dir):
    # TODO: remove src_dir statement
    src_dir = 'C:/Users/Aaron Ward/Box/Data_Analysis_Transforms/Practice_Files/subdir'
    # get list of subj folders
    subj_list = next(os.walk(src_dir))[1]
    # get list of vipr folders
    for subj_folder in subj_list:
        vipr_list = next(os.walk(os.path.join(src_dir, subj_folder)))[1]
        for vipr_folder in vipr_list:
            vipr_path_root = os.path.join(src_dir, subj_folder, vipr_folder)
            vipr_path_raw = os.path.join(vipr_path_root, 'raw')
            vipr_path_proc = os.path.join(vipr_path_root, 'proc')
            vipr_path_graphs = os.path.join(vipr_path_root, 'graphs')

            target_dirs = [
                # TODO: add paths to save e.g. Box, HDD
            ]

            organize(vipr_path_root, target_dirs)

            # split wb
            split_workbook(vipr_path_raw)

            # make a list of all vessel files
            file_list = [os.path.basename(file) for file in glob.glob(os.path.join(vipr_path_raw, "*time_resolved_raw.csv"))]
            file_set = {file.replace('_time_resolved_raw.csv', '') for file in file_list}
            file_list = list(file_set)
            file_list.sort()

            # make a df of the selected voxels
            df_voxel = pd.read_csv(os.path.join(vipr_path_raw, 'voxels.csv'))

            # make a list of the vessels, voxel, and comma counts in the voxel df
            vessel_list = df_voxel['Vessel'].tolist()
            voxel_list = df_voxel['Voxel'].tolist()
            df_voxel['Voxel'] = df_voxel['Voxel'].astype(str)
            comma_count_list = df_voxel['Voxel'].str.count(',').tolist()
            combined_list = sorted(zip(vessel_list, voxel_list, comma_count_list))

            # return revised list of vessels that have a .csv file
            # and have a voxel measurement
            revised_list = []
            for i in file_list:
                output = list(filter(lambda x: i in x, combined_list))
                revised_list.append(output)
            revised_list = list(filter(None, revised_list))

            # make list of time resolved column names
            column_names = ['Voxel']
            i = 5
            while i <= 100:
                mystr = 'Cardiac Time ' + str(i) + '%'
                column_names.append(mystr)
                i = i + 5

            #######################
            resolved_voxel_list = []
            averaged_voxel_list = []
            resolved_value_list = []
            averaged_value_list = []
            resolved_vessel_list = []
            averaged_vessel_list = []
            count = 0
            window_size = 3

            for vessel in revised_list:

                # assign voxel number
                voxel = vessel[0][1]

                # assign comma count
                comma_count = vessel[0][2]

                # import raw time resolved csv file
                time_resolved_file = os.path.join(vipr_path_raw, str(vessel[0][0]) + "_time_resolved_raw.csv")
                df_resolved_raw = pd.read_csv(time_resolved_file, skiprows=[0, 1], header=None)
                # import raw time averaged csv file
                time_averaged_file = os.path.join(vipr_path_raw, str(vessel[0][0]) + "_time_averaged_raw.csv")
                df_averaged_raw = pd.read_csv(time_averaged_file)

                # make hr df only 1 time
                while count < 1:
                    df_hr = pd.read_csv(time_resolved_file, nrows=1, header=None)
                    count = count + 1

                # convert voxel column to type integer
                df_resolved_raw[0] = df_resolved_raw[0].astype(int)
                df_averaged_raw['Point along Vessel'] = df_averaged_raw['Point along Vessel'].astype(int)

                # rename columns
                df_resolved_raw.columns = column_names
                df_averaged_raw.rename(columns={'Point along Vessel': 'Voxel'}, inplace=True)

                # make refined df by calling remove outliers fcn
                df_resolved_rm_outliers = rm_outliers(df_resolved_raw)
                df_averaged_rm_outliers = rm_outliers(df_averaged_raw)

                # make series from df
                resolved_seq = df_resolved_rm_outliers.iloc[:, 0]
                averaged_seq = df_averaged_rm_outliers.iloc[:, 0]

                # perform sliding window function to find contiguous voxels, then remove non-contiguous voxels
                # time resolved
                windows = list(sliding_window(resolved_seq, window_size))
                resolved_voxel_list = noncontig_voxels(windows)
                #time averaged
                windows = list(sliding_window(averaged_seq, window_size))
                averaged_voxel_list = noncontig_voxels(windows)

                # return the final list of voxels
                final_voxel_list = set(resolved_voxel_list).intersection(averaged_voxel_list)
                final_voxel_list = list(final_voxel_list)

                # calculate the yield of acceptable voxels and write to file
                voxel_yield = len(final_voxel_list) / len(resolved_voxel_list)
                fname = "voxel_yield.txt"
                try:
                    os.remove(os.path.join(vipr_path_proc, fname))
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise
                with open(os.path.join(vipr_path_proc, fname), 'a') as f:
                    f.write('{} voxel yield: {:01.3f}\n'.format(str(vessel[0][0]), voxel_yield))

                # write new time resolved and time averaged refined dataframes and files
                df_resolved_refined = refined_to_csv(vipr_path_proc, df_resolved_raw, time_resolved_file, final_voxel_list)
                df_averaged_refined = refined_to_csv(vipr_path_proc, df_averaged_raw, time_averaged_file, final_voxel_list)

                # get missing voxels, lower bound, and upper bound of voxels
                missing_voxel_list, lower_bound, upper_bound = missing_voxels(voxel, comma_count, final_voxel_list)

                # perform flow calculations
                if len(missing_voxel_list) == 0:
                    df_flow = sumproduct(df=df_resolved_refined, df_hr=df_hr)
                    df_flow.insert(loc=0, column='Voxel', value=final_voxel_list)
                    fname_analyzed = str(vessel[0][0]) + '_time_resolved_analyzed.csv'
                    save_path = os.path.join(vipr_path_proc, fname_analyzed)
                    df_flow.to_csv(save_path, sep=',', index=False)
                    # time resolved
                    resolved_value_list.extend([np.mean(df_flow.iloc[lower_bound:upper_bound, 1])])
                    resolved_vessel_list.append(vessel[0][0])

                    # time averaged
                    averaged_value_list.extend([np.mean(df_averaged_refined.loc[lower_bound:upper_bound][1:], axis=0)])
                    averaged_vessel_list.append(vessel[0][0])
                else:
                    # missing voxels
                    # write any voxels that need to be adjusted to a .txt file
                    fname = 'missing_voxels.txt'
                    try:
                        os.remove(os.path.join(vipr_path_proc, fname))
                    except OSError as e:
                        if e.errno != errno.ENOENT:
                            raise
                    msg = str(vessel[0][0]) + ' voxel selection is not in the refined list of voxels. '
                    msg = msg + 'Do not include the following voxels: ' + str(missing_voxel_list)
                    msg = msg + '\n'
                    with open(os.path.join(vipr_path_proc, fname), 'a') as f:
                        f.write(msg)

                # make graphs
                time_averaged_plots(vipr_path_graphs, str(vessel[0][0]), df_averaged_raw,  df_averaged_refined)
                time_resolved_plots(vipr_path_graphs, str(vessel[0][0]), df_resolved_raw, df_resolved_refined)

            # write aggregated time resolved data
            resolved_columns = ['Vessel', 'Flow']
            df_resolved_agg = pd.DataFrame(list(zip(resolved_vessel_list, resolved_value_list)), columns=resolved_columns)
            fname = 'agg_time_resolved.csv'
            df_resolved_agg.to_csv(os.path.join(vipr_path_proc, fname), sep=",", index=False)

            # write aggregated time averaged data
            df_averaged_agg = pd.DataFrame(averaged_value_list)
            column_values = pd.Series(averaged_vessel_list)
            del df_averaged_agg['Voxel']
            df_averaged_agg.insert(loc=0, column='Vessel', value=column_values)
            fname = 'agg_time_averaged.csv'
            df_averaged_agg.to_csv(os.path.join(vipr_path_proc, fname), sep=",", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze VIPPR summary.xls files')
    parser.add_argument('-p', '--path', help='Path to VIPR files', type=str, default=None)
    args = parser.parse_args()
    if args.path is not None:
        src_dir = args.path
    else:
        src_dir = os.getcwd()
    main(src_dir)
