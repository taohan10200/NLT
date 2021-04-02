clc; clear all;
dataset = 'B';
att ='test'
standard_size = [480,640];

path = ['E:/Crowd_Count_DataSet/mall_dataset/frames/'];

output_path = 'E:Crowd_Count_DataSet/mall_dataset/k15-s4/';
train_path_img = strcat(output_path , att, '/img/');
train_path_den = strcat(output_path,  att,'/den/');

gt_path = ['E:/Crowd_Count_DataSet/mall_dataset/'];

mkdir(output_path)
mkdir(train_path_img);
mkdir(train_path_den);
%This is a read example
load('mall_gt.mat');
example_path = ['E:/Crowd_Count_DataSet/mall_dataset/frames/seq_%.6d.jpg'];
img_name = example_path

img_index = 970;
im = imread(sprintf(img_name,img_index));
XY=frame{img_index}.loc;
imshow(im); hold on;
plot(XY(:,1),XY(:,2),'r*');
%====================
num_images = 2000;
for idx = 801:num_images
    i = idx;
    if (mod(idx,10)==0)
        fprintf(1,'Processing %3d/%d files\n', idx, num_images);
    end
   % load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat')) ;
    load(strcat(gt_path,'mall_gt.mat'));
    input_img_name = strcat(path,'seq_%.6d.jpg');
    im = imread(sprintf(input_img_name,i));  
    [h, w, c] = size(im);
    annPoints =  frame{i}.loc;


    rate = standard_size(1)/h;
    rate_w = w*rate;
    if rate_w>standard_size(2)
        rate = standard_size(2)/w;
    end
    rate_h = double(int16(h*rate))/h;
    rate_w = double(int16(w*rate))/w;
    im = imresize(im,[int16(h*rate),int16(w*rate)]);
    annPoints(:,1) = annPoints(:,1)*double(rate_w);
    annPoints(:,2) = annPoints(:,2)*double(rate_h);
    
    im_density = get_density_map_gaussian(im,annPoints,15,4); 
    im_density = im_density(:,:,1);
    
    imwrite(im, [train_path_img num2str(idx) '.jpg']);
    csvwrite([train_path_den num2str(idx) '.csv'], im_density);
end
