function [SWTImage] = SWT(grayImage, edgeImage, dx, dy, dark_on_light)

%im = rgb2gray(im);
%edgeImage = edge(grayImage,'canny');
% [dx, dy] = gradient(double(grayImage));
% figure; imshow(dx); title(sprintf('Dx Image'));
% figure; imshow(dy); title(sprintf('Dy Image'));

% img = im2single(grayImage);
% img = imfilter( img, fspecial('gauss',[5 5], 0.3*(2.5-1)+.8) );
% gx = imfilter( img, fspecial('prewitt')' ); %//'
% gy = imfilter( img, fspecial('prewitt') );
% gx = single(medfilt2( gx, [3 3] ));
% gy = single(medfilt2( gy, [3 3] ));
% figure; imshow(img); title(sprintf('Gauss Image'));
% figure; imshow(gx); title(sprintf('Dx2 Image'));
% figure; imshow(gy); title(sprintf('Dy2 Image'));

% figure; imshow(SWTImage); title(sprintf('SWT Image'));

% get rows and column sizes
[imageHeight, imageWidth] = size(grayImage);
prec = 0.05;
flag = 0;
cnt=1;

%set the image to all -1
SWTImage = zeros(imageHeight,imageWidth)-1;

for row = 1:imageHeight
    for col = 1:imageWidth
        
        % next pixel if the pixel is not an edge
        if( edgeImage( row, col) < 1 )
            continue;
        end
        
        p.x = col;
        p.y = row;
        r.p = p;
        k = 1;
        clear pointsList;
        pointsList(k) = p;
        
        curX = col + 0.5;
        curY = row + 0.5;
        curPixX = col;
        curPixY = row;
        
        G_x = dx(row,col);
        G_y = dy(row,col);
        %         if (G_x~=0&&G_y~=0)
        
        % normalize the gradient vector so that we only have the direction
        mag = sqrt((G_x * G_x) + (G_y * G_y));
        if(dark_on_light)
            G_x = -G_x/mag;
            G_y = -G_y/mag;
        else
            G_x = G_x/mag;
            G_y = G_y/mag;
        end
        
        % Traverse in the direction of the vector
        while(flag==0)
            curX = curX + G_x * prec;
            curY = curY + G_y * prec;
            
            % If we found a new pixel
            if ( (uint16(curX) ~= uint16(curPixX)) || (uint16(curY) ~= uint16(curPixY)) )
                curPixX = double(uint16(curX));
                curPixY = double(uint16(curY));
  
                % If pixel is out of bounds then leave
                if (curPixX <=0||curPixX>imageWidth||curPixY <=0 ||curPixY >imageHeight)
                    break
                end
                
                % Store the new found pixel
                pnew.x = curPixX;
                pnew.y = curPixY;
                k = k + 1;
                pointsList(k) = pnew;
                
                % If the new found pixel is an edge
                if (edgeImage(curPixY,curPixX)>0)
                    
                    % Set the ray's ending point
                    r.q = pnew;
                    % Get the new point's gradient direction and normalize %it
                    G_xt = dx(curPixY,curPixX);
                    G_yt = dy(curPixY,curPixX);                    
                    mag = sqrt(G_xt*G_xt + G_yt*G_yt);
                    if(dark_on_light)
                        G_xt = -G_xt/mag;
                        G_yt = -G_yt/mag;
                    else
                        G_xt = G_xt/mag;
                        G_yt = G_yt/mag;
                    end
                    
                    % If the direction of the new pixel is roughly opposite
                    % to the original pixel
                    if(acos(G_x * -G_xt + G_y * -G_yt) < pi/2)
                        % Calculate the length of the ray
                        len = sqrt(double((r.q.x - r.p.x)*(r.q.x - r.p.x) + (r.q.y - r.p.y)*(r.q.y - r.p.y)));
                        
                        % Set all the found pixels of the ray to the ray's
                        % length
                        for m=1:k
                            if (SWTImage(pointsList(m).y,pointsList(m).x)<0)
                                SWTImage(pointsList(m).y,pointsList(m).x) = len;
                            else
                                SWTImage(pointsList(m).y,pointsList(m).x) = min(len,SWTImage(pointsList(m).y,pointsList(m).x));
                            end
                        end
                        
                        % Add the pointsList to the ray object
                        r.pointsList = pointsList;
                        % Add the ray object to the rays list
                        raysList(cnt)=r; cnt=cnt+1;
                    end
                    break
                end
            end
        end
    end
end

% subplot(1,3,1);
% imagesc(edgeImage);
% subplot(1,3,2);
% imagesc(SWTImage);

% For each ray, go through each of the pointsList and find the median length.
% Set each point to the median length
for i=1:length(raysList)-1          %each rays
    cnt1 = 1;
    for j=1:length(raysList(i).pointsList)  %each pointsList on the rays
        xxx(cnt1) = SWTImage(raysList(i).pointsList(j).y,raysList(i).pointsList(j).x);
        cnt1 = cnt1+1;
    end
    raysList(i).med = median(xxx);
    
    for j=1:length(raysList(i).pointsList)  %each pointsList on the rays
        SWTImage(raysList(i).pointsList(j).y,raysList(i).pointsList(j).x)= min(SWTImage(raysList(i).pointsList(j).y,raysList(i).pointsList(j).x),raysList(i).med);
    end    
end
%SWTImage(SWTImage == -1) = max( SWTImage(:) );
%SWTImage(SWTImage == -1) = 0;

%subplot(1,3,3);
%imagesc(SWTImage);colorbar;
figure,imagesc(SWTImage); colorbar;
figure,imshow(SWTImage);
