close all
%read dataset train
P = './Dataset/train';
D = dir(fullfile(P,'*.pgm'));
C = cell(size(D));
U = cell(size(D));
for k = 1:numel(D)
    C{k} = imread(fullfile(P,D(k).name));
end
X = cast(reshape(cell2mat(cellfun(@(x) reshape(x, [], 1), C, 'un',0)), max(size(C{1}).^2), numel(D)), 'double');

%read dataset test
P = './Dataset/test';
D = dir(fullfile(P,'*.pgm'));
C = cell(size(D));
U = cell(size(D));
for k = 1:numel(D)
    C{k} = imread(fullfile(P,D(k).name));
end

S_mask = ones(19, 19);
S_mask(:, 11:19) = bsxfun(@minus, S_mask(:, 11:19), 0.05*(1:9));

masked = cast(C{k}, 'double') .* S_mask;

rank_max = 100;

error_store_svd = zeros(rank_max, 1);
error_store_nmf = zeros(rank_max, 1);

%SVD of X
[U,S,V] = svd(X);
for r=1:rank_max
    error_svd = zeros(numel(D), 1);
    error_nmf = zeros(numel(D), 1);
    %r_nmf
    r_nmf = r;
    %%r_svd
    r_svd = r;
    %NMF
    [W,H] = nnmf(X,r_nmf);
    for k=1:numel(D)
        %add mask
        masked = cast(C{k}, 'double') .* S_mask;
        %vectorize noisy image
        masked_vectorized = reshape(masked, [], 1);
        %%compute accumulated energy
        e = cumsum(diag(S).^2);
        %reconstruct y_svd
        y_bar_svd = U(:, 1:r_svd) * (transpose(U(:, 1:r_svd)) * masked_vectorized);
        %reconstruct y_nmf
        beta = inv(transpose(W)*W)*transpose(W)*masked_vectorized;
        y_bar_nmf = W(:, 1:r_nmf) * beta;
        reconstructed_image_nmf = reshape(y_bar_nmf, 19, 19);
        reconstructed_image_svd = reshape(y_bar_svd, 19, 19);
        if k == 10 && r == 30
           figure
           imshow(imresize(cast(reconstructed_image_nmf, 'uint8'), 5))
           truesize
           figure
           imshow(imresize(cast(reconstructed_image_svd, 'uint8'), 5))
           figure
           imshow(imresize(cast(masked, 'uint8'), 5)) 
        end
        error_svd(k) = norm(cast(C{k}, 'double') - reconstructed_image_svd, 'fro');
        error_nmf(k) = norm(cast(C{k}, 'double') - reconstructed_image_nmf, 'fro');
    end
    error_store_svd(r) = sum(error_svd)/472;
    error_store_nmf(r) = sum(error_nmf)/472;
end

figure
plot([1:rank_max], error_store_svd);
title('SVD error vs. rank for masked images');

figure
plot([1:rank_max], error_store_nmf);
title('NMF error vs. rank for masked images');
