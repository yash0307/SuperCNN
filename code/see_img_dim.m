srcFiles_img = dir('../data/images/*.png');
for i = 1 : length(srcFiles_img)
    filename = strcat('../data/images/',srcFiles_img(i).name);
    im = imread(filename);
    size = size(im)
    pause(1)
end