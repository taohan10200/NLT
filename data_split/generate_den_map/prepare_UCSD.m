clc; clear all;
standard_size = [480,720];

RootPath = 'E:\Crowd_Count_DataSet\UCSD\';
DstPath = 'E:\Crowd_Count_DataSet\UCSD\Mall_blurred';


%% Train/Test set
path = [RootPath 'ucsdpeds/vidf/']
gt_path = [RootPath '/vidf-cvpr/'];
output_path = [RootPath '/UCSD_480_k15_4/']; 
mkdir(output_path);

train_path_img = strcat(output_path,'/train/', 'img/');
train_path_den = strcat(output_path,'/train/', 'den/');
test_path_img = strcat(output_path,'/test/', 'img/');
test_path_den = strcat(output_path,'/test/', 'den/');
mkdir(train_path_img);
mkdir(train_path_den);
mkdir(test_path_img);
mkdir(test_path_den);


folder_list =  dir(fullfile(path,'vidf1_33_00*'))
gt_list = dir(fullfile(gt_path,'*_frame_full.mat')); 
for i_folder = 1:10

folder_path = folder_list(i_folder).name;
    
img_list = dir(fullfile([path,folder_path],'*.png')); 
load([gt_path, '/', gt_list(i_folder).name]);% people

for idx = 1:size(img_list,1)
    filename = img_list(idx,1).name;
    filename_no_ext = regexp(filename, '.png', 'split');
    filename_no_ext = filename_no_ext{1,1};
      
    i = idx;
    if (mod(idx,10)==0)
        fprintf(1,'Train: Processing %3d/%d files\n', idx, size(img_list,1));
    end

    input_img_name = strcat(path,folder_path,'/',filename);
    im = imread(input_img_name);  
     [h, w, c] = size(im);
    im = imresize(im, standard_size);
   
    point_position = frame{1,idx}.loc;
    if isempty(point_position)
        continue;
    end
    annPoints = point_position;
    annPoints(:,1) = annPoints(:,1)/w*standard_size(2);
    annPoints(:,2) = annPoints(:,2)/h*standard_size(1);
    
%     %% ROI mask
%     BW = mask;       
%     back_img = im;
%     back_img(:,:,1) = back_img(:,:,1).*uint8(~BW);
%     back_img(:,:,2) = back_img(:,:,2).*uint8(~BW);
%     back_img(:,:,3) = back_img(:,:,3).*uint8(~BW); 
%     %% blur the region out of interest region
%     H = fspecial('disk',5);
%     back_img = imfilter(back_img,H,'replicate');
%     back_img(:,:,1) = back_img(:,:,1).*uint8(~BW);
%     back_img(:,:,2) = back_img(:,:,2).*uint8(~BW);
%     back_img(:,:,3) = back_img(:,:,3).*uint8(~BW);  
%     %% keep ROI 
%     im(:,:,1) = im(:,:,1).*uint8(BW);
%     im(:,:,2) = im(:,:,2).*uint8(BW);
%     im(:,:,3) = im(:,:,3).*uint8(BW);
%     %% restore img
%     final_img = back_img+im;
    %% generation  
    im_density = get_density_map_gaussian(im,annPoints,15,4); 
    im_density = im_density(:,:,1);
    if isempty(annPoints)
        continue;
    end
    %% save
    imRGB = insertShape(im,'FilledCircle',[annPoints(:,1),annPoints(:,2),5*ones(size(annPoints(:,1)))],'Color', {'red'});
    figure(1),imshow(imRGB);
     figure(2),imagesc(im_density);
if i_folder>=6 && i_folder<=9
    imwrite(im, [train_path_img '/' filename_no_ext '.png']);
    csvwrite([train_path_den  '/' filename_no_ext '.csv'], im_density);
else
        imwrite(im, [test_path_img '/' filename_no_ext '.png']);
    csvwrite([test_path_den  '/' filename_no_ext '.csv'], im_density);
end
end
end