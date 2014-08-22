
addpath 'C:\OCR\sample_images\';
addpath 'C:\OCR\StepsOutput\';
addpath 'C:\OCR\vsprojects\face_reco\haar\';
colorImage1 = imread('_10_2_roi.png');
colorImage2 = imread('_6_1_roi.png');
colorImage3 = imread('_7_1_roi.png');
colorImage4 = imread('_10_2_roi.png');

colorImageArray = {colorImage1, colorImage2, colorImage3, colorImage4};
resizedImageArray = cell(1, 4);

for n = 1:numel(colorImageArray)
    resizedImageArray{n} = imresize(colorImageArray{n}, 0.5);
%      figure; imshow(resizedImageArray{n}); title(sprintf('Image %f', n));
end

% convert to gray
grayImageArray = cell(1, 4);

for n = 1:numel(colorImageArray)
    grayImageArray{n} = rgb2gray(resizedImageArray{n});
%       figure; imshow(grayImageArray{n}); title(sprintf('Gray Image %f', n));
end

% Run the edge detector
edgeImageArray = cell(1, 4);

for n = 1:numel(colorImageArray)
    edgeImageArray{n} = edge(grayImageArray{n}, 'Canny', [0.1 0.3]);
     figure; imshow(edgeImageArray{n}); title(sprintf('Edge Image %f', n));
end

[dx, dy] = gradient(double(grayImageArray{1}));

SWTImage = SWT(grayImageArray{1}, edgeImageArray{1}, dx, dy, true);

