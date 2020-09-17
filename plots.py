import matplotlib.pyplot as plt

def time_resolved_plots(df_resolved_raw, df_resolved_refined):
    #time resolved plots

    plt.figure()
    
    #raw plot
    df_resolved_raw_plot = df_resolved_raw.iloc[:,1:]
    df_resolved_raw_plot = df_resolved_raw_plot.T
    plt.subplot()
    plt.xlabel('Voxel')
    plt.ylabel('Flow (mL/s)')
    df_resolved_raw_plot.plot(legend = False)
    
    #final refined plot
    df_resolved_refined_plot = df_resolved_refined.iloc[:,1:]
    df_resolved_refined_plot = df_resolved_refined_plot.T
    plt.subplot()
    plt.xlabel('Voxel')
    plt.ylabel('Flow (mL/s)')
    df_resolved_refined_plot.plot(legend = False)
    
    #save plots?
	
def time_averaged_plots(df_averaged_raw, df_averaged_refined):
    #will create a combined plot of the raw and refined data for each variable

    #loop over each variable
    for i in range(1,8):
        plt.figure()

        #raw plot
        var_column = i
        df_averaged_raw_plot = df_averaged_raw.iloc[:, var_column]
        df_averaged_raw_plot = df_averaged_raw_plot.T
        df_averaged_raw_plot.plot(legend = False)

        #final refined plot
        df_averaged_refined_plot = df_averaged_refined.iloc[:,var_column]
        df_averaged_refined_plot = df_averaged_refined_plot.T
        df_averaged_refined_plot.plot(legend = False)

        #save plots?