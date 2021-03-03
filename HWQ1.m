close all
%read dataset train
P = './Dataset/train';
D = dir(fullfile(P,'*.pgm'));
C = cell(size(D));
U = cell(size(D));
for k = 1:numel(D)
    U{k} = im2double(imread(fullfile(P,D(k).name)));
end
X = reshape(cell2mat(cellfun(@(x) reshape(x, [], 1), U, 'un',0)), max(size(U{1}).^2), numel(D));
%use SVD to factorize X
[U,S,V] = svd(X);
diag(S.^2);
%plot singular values
figure
plot([1:361], diag(S));
title('Singular Values vs. \Sigma [k]');
ylabel('k^{th} Singular Value');
xlabel('\Sigma [k]');
%%compute accumulated energy
e = cumsum(diag(S).^2);
%%plot normalized accumulated energy
figure
plot([1:361], e/max(e));
title('Normalized Accumulated Energy');
ylabel('e[k]') 
xlabel('k') 
%identify indices I90, I95, I99
I90 = find(e/max(e)>=0.9, 1)
I95 = find(e/max(e)>=0.95, 1)
I99 = find(e/max(e)>=0.99, 1)
%display first I90 columns of U as an image
figure
imshow(imresize(reshape(U(:, 1), sqrt(max(size(X(:,1)))), sqrt(max(size(X(:,1))))), 5), [min(U(:, 1)), max(U(:, 1))]);
title('Reconstructed image from the first I90 columns of U');
figure
for i = 1:30
    subplot(6,5,i), imshow(imresize(reshape(U(:, i), sqrt(max(size(X(:,i)))), sqrt(max(size(X(:,i))))), 5), [min(U(:, i)), max(U(:, i))]);
end
%nmf with HALS
% X = WH
number_of_iterations = 100;
r_nmf = 30;
[W,H, error] = Two_Block_Coordinate_Descent(X,r_nmf,number_of_iterations);
final_error = norm(X - W*H, 'fro')
figure
plot([1:number_of_iterations], error);
title("Average Error vs. Iterations");
ylabel('||X - WH||_F / ||X||_F');
xlabel('Iteration Number');
figure
for i = 1:30
    subplot(6,5,i), imshow(imresize(reshape(W(:, i), sqrt(max(size(X(:,i)))), sqrt(max(size(X(:,i))))), 5), [min(W(:, i)), max(W(:, i))]);
end