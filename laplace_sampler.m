function Y = laplace_sampler(X, sigma)
    % sample based on laplace distribution on SPD manifold with mean X is 
    % 2x2 SPD matrix and sigma
    % https://www.mdpi.com/1099-4300/18/3/98
    m = size(X, 1);
    assert(m ==2);
    
    gamma21 = sqrt(pi) * gamma(1) * gamma(0.5);
    omegam = pi^(m^2/2)/(gamma21);
    cm = 1/factorial(m) * omegam * 8^(m*(m-1)/4);
    
    % sample r
    pdf = @rpdf;
    proprnd = @(r) r + randn(1,m);
    rv = mhsample([0.1 0.2], 1, 'pdf',pdf, 'proprnd', proprnd, 'symmetric',1, 'burnin', 10000, 'thin', 10);
    
    U = randn(m,m);
    [U, ~] = qr(U);
    W = U' * diag([exp(rv(1)) exp(rv(2))]) * U;
    
    Xhalf = sqrtm(X);
    
    Y = Xhalf * W * Xhalf;

    
    function q = rpdf(r)
        assert(length(r) == 2);
        r1 = r(1);
        r2 = r(2);
        q = cm * exp(-sqrt(r1^2+r2^2)/sigma) * sinh(abs(r1-r2)/2);
    end

end

