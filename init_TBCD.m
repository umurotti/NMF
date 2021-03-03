function [W,H] = init_TBCD(X,r)
    %initialization
    [U,S,V] = svd(X);
    for k = 1:r
        %calculation of Mk+
        u_k_plus = max(U(:, k), 0);
        v_k_plus = max(V(:, k), 0);
        M_k_plus = u_k_plus * transpose(v_k_plus);
        %calculation of Mk-
        u_k_neg = max(U(:, k).*-1, 0);
        v_k_neg = max(V(:, k).*-1, 0);
        M_k_neg = u_k_neg * transpose(v_k_neg);
        %initialization of columns of W and H
        if norm(M_k_plus,'fro') >= norm(M_k_neg,'fro')
            W(:,k)= u_k_plus / norm(u_k_plus);
            H(k, :)= transpose(S(k, k) * norm(u_k_plus) * v_k_plus);
        else
            W(:,k)= u_k_neg / norm(u_k_neg);
            H(k, :)= transpose(S(k, k) * norm(u_k_neg) * v_k_neg);
        end
    end
end

