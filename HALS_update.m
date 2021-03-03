function [T] = HALS_update(X,H,W)
    [m,n] = size(W);
    for l = 1:n
        C = 0;
        for k = 1:n
            if k ~= l
                A = H(k,:)*transpose(H(l,:));
                B = W(:,k);
                C = C + B * A;
            end
        end
        D = X * transpose(H(l,:));
        E = norm(H(l,:)) ^ 2;
        F = (D - C) / E;
        W(:,l) = max(F(:,1),0);
    end
    T = W;
end

