function [] = pca_sphere()
    
    clear;
    clc;
    rng(9);
    
    
    
    %% set parameters
    %n = 200;
    d = 50;
    nu = 1e-3;
    % privacy parameters
    eps = 0.1;
    delta = 1e-3;    
    
    M = spherefactory(d);    
    
    
    
    %%
    
    repeats = 20;
    ns = 1:20;
    ns = ns * 100;
    
    outgap_prgd = zeros(length(ns), repeats);
    outgap_ppgd = zeros(length(ns), repeats);
    for in = 1:length(ns)
        n = ns(in);
        
        X = genZ(n,d,nu); 
        
        % estimate Lipschitz constant 
        L0 = 0;
        for i = 1:n
            temp = X(i,:)* X(i,:)';
            if temp >= L0
                L0 = temp;
            end
        end
        
        A = (X'*X);
        
        T = log(n^2*eps^2/(d*L0^2*log(1/delta))); % max epoch
        T = ceil(T);
        sigma = T *log(1/delta) *L0^2/(n^2*eps*2); %std
        sigma = sqrt(sigma);
       
            
        for irep = 1:repeats  
            w0 = M.rand();

            % PRGD: perturbed Riemannian gradient descent
            w_prgd = w0;
            eta = 0.2;
            for ii = 1:T
                rgrad = M.egrad2rgrad(w_prgd, -2*A*w_prgd);
                noise = normrnd(0, sigma, [d-1,1]);
                noise = vec2tangent(w_prgd, noise);
                rgrad = rgrad + noise;
                w_prgd = M.exp(w_prgd, rgrad, -eta);
            end
            outgap_prgd(in, irep) = abs(w_prgd'*A*w_prgd - 1);

            % PPGD: perturbed projected gradient descent
            w_ppgd = w0;
            eta = 0.2;
            for ii = 1:T
                egrad = -2*A*w_ppgd;
                noise = normrnd(0, sigma, [d,1]);
                egrad = egrad + noise;
                w_ppgd = w_ppgd - eta * egrad;
                w_ppgd = w_ppgd/norm(w_ppgd);        
            end
            outgap_ppgd(in, irep) = abs(w_ppgd'*A*w_ppgd - 1);
            

        end
    end
    
    %save('pca_results.mat', 'ns', 'outgap_prgd', 'outgap_ppgd');
    
    %% plot 
    
    fs = 22;
    
    mean_prgd = mean(outgap_prgd, 2);
    mean_ppgd = mean(outgap_ppgd, 2);
    std_prgd = std(outgap_prgd, 0, 2);
    std_ppgd = std(outgap_ppgd, 0, 2);
    
    
    h = figure();
    x = ns';
    y = mean_ppgd;
    %dy = std_ppgd;
    %fill([x;flipud(x)],[y-dy;flipud(y+dy)],[173, 216, 230]/255,'linestyle','none');
    semilogy(x,y,'color','#0072BD','LineWidth',3); hold on;
    y = mean_prgd;
    %dy = std_prgd;
    %fill([x;flipud(x)],[y-dy;flipud(y+dy)],[254 216 177]/255,'linestyle','none');
    semilogy(x,y, 'color','#EDB120','LineWidth',3); hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize', fs);
    set(gca,'fontname','Arial');
    legend('DP-PGD', 'DP-RGD', 'fontsize', fs-2);
    xlabel('Sample size', 'fontsize', fs);
    ylabel('Empirical excess risk', 'fontsize', fs);
    
    
    %% helper functions
    function B = genbasis(w)
        % create a basis for tangent space at w

        [~, ~, V] = svd(reshape(w, [1, length(w)]));
        %V = V';
        B = V(:,2:end);    
        assert(size(B,1) == d);
        assert(size(B,2) == d-1);
    end

    function u = vec2tangent(w, v)
        % convert d-1 dim euclidean vector to tangent vector of dim d at w
        assert(length(v) == d-1);
        B = genbasis(w);
        u = B * reshape(v, [d-1,1]);
    end
    


    % generate synthetic samples
    function X = genZ(n,d,nu)
        % the eigenvalues are (1, 1-nu, 1-1.1nu,...1-1.4nu, ... |g|/d...)
        if d > 5
            D = [1 1-nu 1-1.1*nu 1- 1.2*nu 1- 1.3*nu 1- 1.4*nu];
            D_ = (normrnd(0,1, [1, d - 6]))/d;
            D_ = abs(D_);
            D = [D D_];
        else
            error("Not implemented for d < 6");
        end
    
        A = randn(n,d);
        U = orth(A);
        assert(rank(U) == d);
    
        A = randn(d,d);
        V = orth(A);
        assert(rank(V) == d);
    
        X= U * diag(D) * V;
    end


end