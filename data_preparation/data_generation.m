clear; clc;
aim_path = '~/dataset/a_matte_multi_focus_dataset_by_VOC2012_1';
sigma = 1; % 0.5 0.75 1 1.5 1.75 2 3 4 5 diff sigma for selection
%% focus map generation
% segmentation_path = [root_path , '/SegmentationClass'];
% focus_map_path = [root_path, '/focus_map_png'];
% 
% % read the file name
% fileFolder1=fullfile(segmentation_path);
% dirOutput1=dir(fullfile(fileFolder1,'*.png'));
% seg_Names={dirOutput1.name};
% M = size(seg_Names);
% for i = 1:M(2)
%     seg_path = [segmentation_path,'\', seg_Names{i}];
%     I = imread(seg_path);
%     I(I~=0) = 255;
%     Fucus_map_path = [focus_map_path,'\', seg_Names{i}];
%     imwrite(I,Fucus_map_path);
% end

%% A and B generation
root_path = '~/dataset/a_matte_multi_focus_dataset_by_VOC2012';
image_path = [root_path , '/image_jpg'];
focus_map_path = [root_path, '/focus_map_png'];

output_folder_A = [aim_path,  '/A_jpg'];
output_folder_B = [aim_path, '/B_jpg'];
output_folder_FM = [aim_path, '/blurred_focus_map_png'];
if exist(output_folder_A,'dir')==0
    mkdir(output_folder_A);
    mkdir(output_folder_B);
    mkdir(output_folder_FM);
end

% read the file name
fileFolder0=fullfile(image_path);
dirOutput0=dir(fullfile(fileFolder0,'*.jpg'));
imag_Names={dirOutput0.name};

fileFolder2=fullfile(focus_map_path);
dirOutput2=dir(fullfile(fileFolder2,'*.png'));
mask_Names={dirOutput2.name};

M = size(imag_Names);
parfor i = 1:M(2)
    img_path = [image_path,'\', imag_Names{i}];
    I = imread(img_path);
    BI = imgaussfilt(I,sigma);
    I = double(I)/255;
    BI = double(BI)/255;
    
    mask_path = [focus_map_path,'\', mask_Names{i}];
    matte = imread(mask_path);
    blurred_focusmap = imgaussfilt(matte,sigma);
    
    matte = double(matte)/255;
    blurred_matte = double(blurred_focusmap)/255;

    S1 = I .* matte + BI .* (1 - matte);
    S2 = BI .* blurred_matte + I .*(1 - blurred_matte); 
    
    A_path = [output_folder_A,'\A_', imag_Names{i}];
    B_path = [output_folder_B,'\B_', imag_Names{i}];
    output_path = [output_folder_FM,'\blurred_', mask_Names{i}];
    
    imwrite(S1,A_path);
    imwrite(S2,B_path);
    imwrite(blurred_focusmap,output_path);
end
