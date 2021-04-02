clc; clear all;
dataset = 'A';
standard_size = [768,1024];

att = 'test';

dataset_name = ['shanghaitech_part_' dataset];
path =            ['E:/Crowd_Count_DataSet/ShanghaiTech_Crowd_Counting_Dataset/part_' dataset '_final/' att '_data/images/'];
output_path = 'E:/Crowd_Count_DataSet/ShanghaiTech_Crowd_Counting_Dataset/';

train_path_img = strcat(output_path, dataset_name,'/', att, '/img/');
train_path_den = strcat(output_path, dataset_name,'/', att, '/den/');

gt_path = ['E:/Crowd_Count_DataSet/ShanghaiTech_Crowd_Counting_Dataset/part_' dataset '_final/' att '_data/ground_truth/'];

% mkdir(output_path)
mkdir(train_path_img);
mkdir(train_path_den);

if (dataset == 'A')
    num_images = 300;
else
    num_images = 400;
end
min_w = 10000;
min_h  = 10000;
for idx = 1:300
    i = idx;
    if (mod(idx,10)==0)
        fprintf(1,'Processing %3d/%d files\n', idx, num_images);
    end
    load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat')) ;
    input_img_name = strcat(path,'IMG_',num2str(i),'.jpg');
    im = imread(input_img_name);  
    [h, w, c] = size(im);
%     if h<min_h
%         min_h = h;
%     end
%     if w < min_w
%         min_w = w;
%     end
%     fprintf(1,'h %3d w %d c %d  \n', h, w,c);
    annPoints =  image_info{1}.location;
%     disp(annPoints)

    exp_w = w + mod(-w, 8);
    exp_h  = h + mod(-h, 8);
    rate_h = exp_h/h;
    rate_w = exp_w/w;
    rate_h = double(int16(h*rate_h))/h;
    rate_w = double(int16(w*rate_w))/w;
    fprintf(1,'h %.3f w %.3f  \n', rate_h, rate_w);
    im = imresize(im,[int16(h*rate_h),int16(w*rate_w)]);
    size(im)
    annPoints(:,1) = annPoints(:,1)*double(rate_w);
    annPoints(:,2) = annPoints(:,2)*double(rate_h);
    
    im_density = get_density_map_gaussian(im,annPoints,15,4); 
    im_density = im_density(:,:,1);

    imwrite(im, [train_path_img num2str(idx) '.jpg']);
    csvwrite([train_path_den num2str(idx) '.csv'], im_density);
end
%min_w, min_h
