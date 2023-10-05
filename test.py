import os
import glob
baselines = ["reinforce-robust"]
tasks = ['tf-bind-10']
folder_origin = '/results/SIRO'

# os.system(f'rm -rf {folder_origin}/{baseline}/{task}/make_table/log.txt')
for baseline in baselines:
    for task in tasks:
        os.system(f'rm -rf {folder_origin}/{baseline}/{task}/make_table')
        os.system(f'mkdir -p {folder_origin}/{baseline}/{task}/make_table')
        for folder in glob.glob(f'{folder_origin}/{baseline}/{task}/*'):
            # os.system(f'rm -rf {folder_origin}/{baseline}/{task}/make_table')
            # os.system(f'mkdir -p {folder_origin}/{baseline}/{task}/make_table')
            for subfolder in os.listdir(folder):
                path_subfolder = os.path.join(folder, subfolder)
                if os.path.isdir(path_subfolder):
                        os.system(f'cp -R {path_subfolder} {folder_origin}/{baseline}/{task}/make_table')
                # for subsubfolder in os.listdir(path_subfolder):
                #     path_subsubfolder = os.path.join(path_subfolder, subsubfolder)
                #     if os.path.isdir(path_subsubfolder):
                #         os.system(f'cp -R {path_subsubfolder} ./make_table/{baseline}/{task}/{baseline}-{task}')
        with open(f'{folder_origin}/{baseline}/{task}/make_table/log.txt', 'a') as txt_file:
            txt_file.write(folder)
            txt_file.write('\n')
        os.system(f'design-baselines make-table --dir {folder_origin}/{baseline}/{task}/make_table --group A --percentile 100th')
        
    
# design-baselines make-table --dir results/SIRO --group A --percentile 100th
