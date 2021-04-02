clc; clear all;
att = 'B';
txt_file = 'SHHA_result.txt';
dataset ='B'
att = 'train';

dataset_name = ['shanghaitech_part_' dataset];
path =            ['E:/Crowd_Count_DataSet/ShanghaiTech_Crowd_Counting_Dataset/part_' dataset '_final/' att '_data/images/'];
output_path = 'E:/Crowd_Count_DataSet/SHHB/';
gt_path = ['E:/Crowd_Count_DataSet/ShanghaiTech_Crowd_Counting_Dataset/part_' dataset '_final/' att '_data/ground_truth/'];

train_path_img = strcat(output_path, dataset_name,'/', att, '/img/');
train_path_den = strcat(output_path,'/', att, '/den/');
SHHB_label =  'E:/Crowd_Count_DataSet/SHHB/train/den/';

label = load(strcat(gt_path, 'GT_IMG_',num2str(1),'.mat')) ;
img_label = importdata([SHHB_label num2str(1) '.csv']);
train_path_den
data = importdata([train_path_den num2str(1) '.csv']);


f = fopen(txt_file, 'w');
for index = 1:1
    label = load(strcat(gt_path, 'GT_IMG_',num2str(index),'.mat')) ;
    img_label = importdata([SHHB_label num2str(index) '.csv']);
    data = importdata([train_path_den num2str(index) '.csv']);
    figure(3)
    imagesc(data)
    figure(4)
    imshow(img_label*200)
    annPoints =  label.image_info{1}.location;
    gt_cnt = length(annPoints);
    gaussian_15_s4 =  sum(sum(img_label));
    error = gt_cnt - gaussian_15_s4;
    fprintf(1,'%d   %.2f   %.3f\n ', gt_cnt, gaussian_15_s4,error );
end
fclose(f);

