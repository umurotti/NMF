function [W,H, error] = Two_Block_Coordinate_Descent(X,r,N)
    %initialization
    [W,H] = init_TBCD(X,r);
    %update
    W_t = W;
    H_t = H;
    error = zeros(N,1);
    for t = 1:N
        W_t_prev = W_t;
        H_t_prev = H_t;
        error(t) = norm(X - W_t*H_t, 'fro') / norm(X, 'fro');
        W_t = HALS_update(X, H_t_prev, W_t_prev);
        H_t = transpose(HALS_update(transpose(X), transpose(W_t), transpose(H_t_prev)));
    end
    W = W_t;
    H = H_t;
end

