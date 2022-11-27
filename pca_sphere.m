function pca_sphere_data()
    clear;
    clc;
    rng(42);
    
    % Notice here, we don't divide by n, for scale issues
    % Whether divide by n has no impact on DP-RGD/DP-PGD except for the
    % stepsize
    % For GMIP, the sensitivity is simply L0 without dividing by n
    
    
    %% set parameters
    % privacy parameters
    delta = 1e-3;     
    
    data = load('./data/data_ijcnn.mat');
    X = data.ijcnn_data;
    
    %data = load('./data/data_satimage.mat');
    %X = data.satimage;
    
    [n, d] = size(X);
    X = zscore(X);
    X = X/norm(X);
    A = X'*X;    
    
    eigenv = eig(A);
    optsol = eigenv(end);
    
    
    % estimate Lipschitz constant 
    L0 = 0;
    for i = 1:n
        temp = 2 *X(i,:)* X(i,:)';
        if temp >= L0
            L0 = temp;
        end
    end
    M = spherefactory(d);
    
    
    repeats = 20;
    
    epss = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7];
    
    outgap_prgd = zeros(length(epss), repeats);
    outgap_ppgd = zeros(length(epss), repeats);
    outgap_gaus = zeros(length(epss), repeats);
    
    for ieps = 1:length(epss)
        eps = epss(ieps);
        
        T = log(n^2*eps^2/(d*L0^2*log(1/delta))); % max epoch
        T = ceil(T);
        sigma = T *log(1/delta) *L0^2/(n^2*eps*2); %std
        sigma = sqrt(sigma);
        
        sigma_gaus =  L0 * sqrt(2 * log(1.25/delta)) /(eps);
        
        for irep = 1:repeats   
            w0 = M.rand();

            % PRGD: perturbed Riemannian gradient descent
            w_prgd = w0;
            eta = 0.9;
            for ii = 1:T
                rgrad = M.egrad2rgrad(w_prgd, -2*A*w_prgd);
                noise = normrnd(0, sigma, [d-1,1]);
                noise = vec2tangent(w_prgd, noise);
                rgrad = rgrad + noise;
                w_prgd = M.exp(w_prgd, rgrad, -eta);
            end
            outgap_prgd(ieps, irep) = abs(w_prgd'*A*w_prgd - optsol);
            %}


            % PPGD: perturbed projected gradient descent
            w_ppgd = w0;
            eta = 10;
            for ii = 1:T
                egrad = -2*A*w_ppgd;
                noise = normrnd(0, sigma, [d,1]);
                egrad = egrad + noise;
                w_ppgd = w_ppgd - eta * egrad;
                w_ppgd = w_ppgd/norm(w_ppgd);        
            end
            outgap_ppgd(ieps, irep) = abs(w_ppgd'*A*w_ppgd - optsol);
            %}
            
            % 
            % Gaussian mechanism input perturbation
            noise = normrnd(0, sigma_gaus, [d,d]);
            noise = triu(noise);
            noise = noise + noise' - diag(diag(noise));
            Anoise = A + noise;            
            
            [w_gaus, D] = eigs(Anoise, 1, 'largestreal');
            outgap_gaus(ieps, irep) = abs(w_gaus'*A*w_gaus - optsol);
           
        end
    end
    
    
    %%
    fs = 22;
    
    mean_prgd = mean(outgap_prgd, 2);
    mean_ppgd = mean(outgap_ppgd, 2);
    mean_gaus = mean(outgap_gaus, 2);
    std_prgd = std(outgap_prgd, 0, 2);
    std_ppgd = std(outgap_ppgd, 0, 2);
    
    
    h = figure();
    x = epss';
    y = mean_gaus;
    semilogy(x,y,'color','b','LineWidth',3); hold on;
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
    legend('DP-GMIP', 'DP-PGD', 'DP-RGD', 'fontsize', fs-2);
    xlabel('Eps', 'fontsize', fs);
    ylabel('Empirical excess risk', 'fontsize', fs);
    
    
    %% helper
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
end
