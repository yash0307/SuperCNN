load('all_L.mat')
load('out.mat')
srcFiles_img = dir('../data/images/*.png');

for a=1:size(all_L,2)
    filename = strcat('../data/images/',srcFiles_img(a).name);
    im = imread(filename);
    subplot(1,2,1),
    imshow(im);
    L = all_L{1,a};
    BW = boundarymask(L);
    imshow(imoverlay(im,BW,'black'),'InitialMagnification',100)

    L_predicted = zeros(size(L));
    labels_predicted = out_mat(a,:);
    label_idx = label2idx(L);
    label_idx = label_idx';
    for j = 1:size(label_idx,1)
        label_idx_j = label_idx{j};
        L_predicted(label_idx_j) = labels_predicted(1,j);
    end
    subplot(1,2,2)
    imshow(L_predicted);
end
    
        
        
    