function mean_spd()
    %% 
    clear;
    clc;
    rng(42);
    
    
    %% 
    % set parameters
    r = 2;
    d = r*(r+1)/2;
    %n = 200;
    
    tol = 1; % diameter
    L0 = tol;
    eps = 0.1;
    delta = 1e-3;  
    
    
    M = sympositivedefinitefactory(r);

    repeats = 20;
    ns = 1:15;
    ns = ns * 20;
    
    outgap_prgd = zeros(length(ns), repeats); 
    for in = 1:length(ns)
        n = ns(in);
        
        T = n;
        sigma = T*log(1/delta) * L0^2/(n^2*eps^2);
        sigma = sqrt(sigma);
        %keyboard;
    
        
        % generate samples 
        X = zeros(r,r,n);
        k = 1;
        while true
            temp = wishrnd(eye(r)/r,r);
            if M.dist(eye(r),temp) < tol
                X(:,:,k) = temp;
                k = k + 1;
                if k > n; break; end
            end
        end
        
        
        % find optimal solution using RGD
        eta = 1;
        W_rgd = M.rand();
        for iter = 1:10
            rgrad = 0;
            for i = 1 : n
                rgrad = rgrad - M.log(W_rgd, X(:, :, i));
            end 
            rgrad = rgrad./n;
            normgrad = M.norm(W_rgd, rgrad);
            fprintf('Iter [%0.4d]:\t norm rgrad %e\n', iter, normgrad);
            if normgrad < 1e-14
                break;
            end
            W_rgd = M.exp(W_rgd, rgrad, -eta);
        end
        % compute optimal cost
        optfn = cost(W_rgd);
        
        
        for irep = 1:repeats  
            W0 = M.rand();
            
            W_prgd = W0;
            eta = 0.01;
            for iter = 1:T
                %optgap = abs(cost(W_prgd) - optfn)
                
                % compute rgrad
                rgrad = 0;
                for i = 1 : n
                    rgrad = rgrad - M.log(W_prgd, X(:, :, i));
                end 
                rgrad = rgrad./n;

                % sample using MHsample
                pdf = @gpdf;
                proprnd = @(xi) xi + randn(1,d);  
                [noise, accept] = mhsample(zeros(1,d), 1, 'pdf',pdf, 'proprnd', proprnd, 'symmetric',1, 'burnin', 10000, 'thin', 10);
                noise = vec2mat(noise);
                %keyboard;

                rgrad = rgrad + noise;
                W_prgd = M.exp(W_prgd, rgrad, -eta);
            end 
            outgap_prgd(in,irep) = abs(cost(W_prgd) - optfn);
            
        end        
    end
    
    %save('mean_spd.mat', 'ns', 'outgap_prgd');
    %}
    
    %% baseline FM-LAP
    ns = 1:15;
    ns = ns * 20; 
    outgap_base = zeros(length(ns), repeats); 
    for in = 1:length(ns)
        n = ns(in);
        
        sigma = 2/(n*eps);

        % generate samples 
        X = zeros(r,r,n);
        k = 1;
        while true
            temp = wishrnd(eye(r)/r,r);
            if M.dist(eye(r),temp) < tol
                X(:,:,k) = temp;
                k = k + 1;
                if k > n; break; end
            end
        end
        
        
        % find optimal solution using RGD
        eta = 1;
        W_rgd = M.rand();
        for iter = 1:10
            rgrad = 0;
            for i = 1 : n
                rgrad = rgrad - M.log(W_rgd, X(:, :, i));
            end 
            rgrad = rgrad./n;
            normgrad = M.norm(W_rgd, rgrad);
            fprintf('Iter [%0.4d]:\t norm rgrad %e\n', iter, normgrad);
            if normgrad < 1e-14
                break;
            end
            W_rgd = M.exp(W_rgd, rgrad, -eta);
        end
        % compute optimal cost
        optfn = cost(W_rgd);
        
        for irep = 1:repeats
            
           W_base = laplace_sampler(W_rgd, sigma);
            
           outgap_base(in, irep) = abs(cost(W_base) - optfn);
        end
        
    end
    %save('mean_spd_base.mat', 'ns', 'outgap_base');
    %}
    
    %%
    % K-norm gradient norm FM-KNG
    ns = 1:15;
    ns = ns * 20; 
    epsilon = 0;
    outgap_kng = zeros(length(ns), repeats); 
    for in = 1:length(ns)
        n = ns(in);
        
        sigma = 4/(n*eps);

        % generate samples 
        X = zeros(r,r,n);
        k = 1;
        while true
            temp = wishrnd(eye(r)/r,r);
            if M.dist(eye(r),temp) < tol
                X(:,:,k) = temp;
                k = k + 1;
                if k > n; break; end
            end
        end
        
        
        % find optimal solution using RGD
        eta = 1;
        W_rgd = M.rand();
        for iter = 1:10
            rgrad = 0;
            for i = 1 : n
                rgrad = rgrad - M.log(W_rgd, X(:, :, i));
            end 
            rgrad = rgrad./n;
            normgrad = M.norm(W_rgd, rgrad);
            fprintf('Iter [%0.4d]:\t norm rgrad %e\n', iter, normgrad);
            if normgrad < 1e-14
                break;
            end
            W_rgd = M.exp(W_rgd, rgrad, -eta);
        end
        % compute optimal cost
        optfn = cost(W_rgd);
        
        for irep = 1:repeats
            W_base = kng_sampler(W_rgd, sigma, 5000, 50000);
            outgap_kng(in, irep) = abs(cost(W_base) - optfn);
        end
        
    end
    save('mean_spd_kng.mat', 'ns', 'outgap_kng');
    
    
    
    
   
    
    
    
    %% helper functions
    function mycost = cost(W_prgd)
        mycost = 0;
        for ii = 1 : n
            mycost = mycost + (M.dist(W_prgd, X(:, :, ii)))^2;
        end
        mycost = mycost/n;
    end
    
    function q = gpdf(xi)
        % evaluate the pdf without normalizing constant
        Xi = vec2mat(xi);
        q = exp(-1/(2*sigma^2) * (M.norm(W_prgd, Xi))^2);
    end
    
    
    function u = mat2vec(U)
        u = zeros(d,1);
        k = 1;
        for ii = 1:r
            for jj = ii:r
                u(k) = U(ii,jj);
                k = k + 1;
            end
        end
        assert(k == d+1);
    end

    function U = vec2mat(u)
        U = zeros(r,r);
        k = 1;
        for ii = 1:r
            for jj = ii:r
                U(ii,jj) = u(k);
                U(jj,ii) = u(k);
                k = k + 1;
            end
        end
        assert(k == d+1);
    end

    function xnext = kng_sampler(x0, sigma, burnin, maxepoch)
        xcur = x0;
        rgrad = 0;
        for jj = 1 : n
            rgrad = rgrad - M.log(xcur, X(:, :, jj));
        end 
        g_cur = exp(-1/sigma * M.norm(xcur, rgrad)/n);
        
        accept_count = 0;
        for ii = 1: maxepoch
            v = unifrnd(-0.5, 0.5, 3,1);
            V = vec2mat(v);
            xnext = M.exp(xcur, 0.1*sigma*V) + epsilon*eye(r);
            rgrad = 0;
            for jj = 1 : n
                rgrad = rgrad - M.log(xnext, X(:, :, jj));
            end 
            g_next = exp(-1/sigma * M.norm(xnext, rgrad)/n);
            
            accept_prob = g_next / g_cur;
            xrand = rand;
            if xrand < accept_prob
                accept_count = accept_count + 1;
                xcur = xnext;
                g_cur = g_next;
            end
            if accept_count >= burnin
                break
            end
            if ii == maxepoch
                disp('Max iteration reached!');
            end
            
        end
        
    end

    function q = kngpdf(xi)
        Xi = vec2mat(xi);
        Xi = expm(Xi);
        rgrad = 0;
        for ii = 1 : n
            rgrad = rgrad - M.log(Xi, X(:, :, ii));
        end 
        q = exp(-1/sigma *  M.norm(Xi, rgrad)/n);
    end
    
    

end

