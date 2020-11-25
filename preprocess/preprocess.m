% Get a folder containing the images to be forward processed
folderInput = uigetdir('','Folder for the input images');
if isnumeric(folderInput); return; end % User cancelled

% Ask for the output folder
folderOutput1 = uigetdir('','Folder1 for the output images');
if isnumeric(folderOutput); return; end % User cancelled

%% scale factor
sf = 2;

%% Search for all the PNG images
allFiles = dir(fullfile(folderInput, '*.png'));

% Go through all the images
for fileID = 1:length(allFiles)
    
    % Open the image
    imageData = imread([folderInput '/' allFiles(fileID).name]);
    h = size(imageData, 1);
    w = size(imageData, 2);
    h = h - rem(h, sf);
    w = w - rem(w, sf);
    imageData = imageData(1:h, 1:w, :);
    % Downscale the image by 2x2
    imageData_LR = imresize(imageData,1/sf,'bicubic');
%     % Add noise
%     sigma = 10;
%     NoisyImage = double(imageData_LR)+randn(size(imageData_LR))*sigma;
%     % Mosaic, pixel order RGGB
%     BayerImage = mosaicrut(NoisyImage);
    
    % Save the image as 8 bit PNG
    imwrite(uint8(imageData_LR),[folderOutput '/' allFiles(fileID).name(1:end-4) '.png']);
    
    % Show progress
    fprintf(1,'%s\n',allFiles(fileID).name);
end
