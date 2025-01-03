import MC_Tutorial_Functions as mc
import pandas as pd
import os

if __name__ == '__main__':

    path = os.getcwd()
    path += "/Simulation Results" #My data is additionally stored in a folder called "Simulation Results" Not necessary if data is saved directly into current working directory
    filename = path + '/tutorial_subset.csv'
    mainframe = mc.load_mainframe(filename)

    row_min = 0
    row_max = len(mainframe)
    analyzed_mainframe = mc.analyze_dataset_numbernorm(mainframe, row_min, row_max)
    title = 'analyzed_tutorial_subset'
    analyzed_mainframe = mc.save_mainframe(analyzed_mainframe,title)