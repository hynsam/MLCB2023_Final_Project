#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-11:00
#SBATCH -p short
#SBATCH --mem=8G
#SBATCH -o /n/groups/marks/users/samhuang/projects/MLCB2023_Final_Project/sandbox/Sam/download_HF_data_moad.out
#SBATCH -e /n/groups/marks/users/samhuang/projects/MLCB2023_Final_Project/sandbox/Sam/download_HF_data_moad.err
#SBATCH --mail-user=yininghuang@hms.harvard.edu
#SBATCH --mail-type=FAIL

wget --no-check-certificate -P /n/groups/marks/users/samhuang/projects/MLCB2023_Final_Project/sandbox/Sam https://bindingmoad.org/files/biou/every_part_a.zip
wget --no-check-certificate -P /n/groups/marks/users/samhuang/projects/MLCB2023_Final_Project/sandbox/Sam https://bindingmoad.org/files/biou/every_part_b.zip
wget --no-check-certificate -P /n/groups/marks/users/samhuang/projects/MLCB2023_Final_Project/sandbox/Sam https://bindingmoad.org/files/biou/nr_bind.zip
wget --no-check-certificate -P /n/groups/marks/users/samhuang/projects/MLCB2023_Final_Project/sandbox/Sam https://bindingmoad.org/files/biou/new_for_2020.zip

