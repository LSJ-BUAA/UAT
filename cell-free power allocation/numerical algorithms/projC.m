function Cout = projC(Pmax,C)
    [L, K] = size(C);
    C = max(1e-6,C);

    row_norm = sum(C.^2, 2);
    scale = min(1, sqrt(Pmax ./ row_norm));

    Cout = C .* scale;
end