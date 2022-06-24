function lrmc_grass()

    clc; 
    close all; 
    clear
    rng(42);
    
    %% Select case
    N = 10000; d = 30; r = 5;
    condition_number = 20;            

    %% Define parameters
    over_sampling = 2;
    noiseFac = 1e-8;
    
    NumEntries_train = over_sampling*r*(N + d -r);      
    NumEntries_test = over_sampling*r*(N + d -r);      

    
    %% Generate data
    fprintf('generating data ... \n');    
    % Generate well-conditioned or ill-conditioned data
    M = over_sampling*r*(N + d -r); % total entries
    
    % The left and right factors which make up our true data matrix Y.
    YL = randn(d, r);
    YR = randn(N, r);
    
    % Condition number
    if condition_number > 0
        YLQ = orth(YL);
        YRQ = orth(YR);
        
        s1 = 1000;
        %     step = 1000; S0 = diag([s1:step:s1+(r-1)*step]*1); % Linear decay
        S0 = s1*diag(logspace(-log10(condition_number),0,r)); % Exponential decay
        
        YL = YLQ*S0;
        YR = YRQ;
        
        fprintf('Creating a matrix with singular values...\n')
        for kk = 1: length(diag(S0))
            fprintf('%s \n', num2str(S0(kk, kk), '%10.5e') );
        end
        singular_vals = svd(YL'*YL);
        condition_number = sqrt(max(singular_vals)/min(singular_vals));
        fprintf('Condition number is %f \n', condition_number);
    end
    cn = floor(condition_number);
    
    % Select a random set of M entries of Y = YL YR'.
    idx = unique(ceil(N*d*rand(1,(10*M))));
    idx = idx(randperm(length(idx)));
    
    [I, J] = ind2sub([d, N],idx(1:M));
    [J, inxs] = sort(J); I=I(inxs)';
    
    % Values of Y at the locations indexed by I and J.
    S = sum(YL(I,:).*YR(J,:), 2);
    S_noiseFree = S;
    
    % Add noise.
    noise = noiseFac*max(S)*randn(size(S));
    S = S + noise;
    
    values = sparse(I, J, S, d, N);
    indicator = sparse(I, J, 1, d, N);
    

    % Creat the cells
    samples(N).colnumber = []; % Preallocate memory.
    for k = 1 : N
        % Pull out the relevant indices and revealed entries for this column
        idx = find(indicator(:, k)); % find known row indices
        values_col = values(idx, k); % the non-zero entries of the column
        
        samples(k).indicator = idx;
        samples(k).values = values_col;
        samples(k).colnumber = k;
    end 
    
    % Test data
    idx_test = unique(ceil(N*d*rand(1,(10*M))));
    idx_test = idx_test(randperm(length(idx_test)));
    [I_test, J_test] = ind2sub([d, N],idx_test(1:M));
    [J_test, inxs] = sort(J_test); I_test=I_test(inxs)';
    
    % Values of Y at the locations indexed by I and J.
    S_test = sum(YL(I_test,:).*YR(J_test,:), 2);
    values_test = sparse(I_test, J_test, S_test, d, N);
    indicator_test = sparse(I_test, J_test, 1, d, N);
    
    samples_test(N).colnumber = [];
    for k = 1 : N
        % Pull out the relevant indices and revealed entries for this column
        idx = find(indicator_test(:, k)); % find known row indices
        values_col = values_test(idx, k); % the non-zero entries of the column
        
        samples_test(k).indicator = idx;
        samples_test(k).values = values_col;
        samples_test(k).colnumber = k;
    end
    
    % for grouse
    data_ls.rows = I;
    data_ls.cols = J';
    data_ls.entries = S;
    data_ls.nentries = length(data_ls.entries);

    data_test.rows = I_test;
    data_test.cols = J_test';
    data_test.entries = S_test;
    data_test.nentries = length(data_test.entries);        
    fprintf('done.\n');    
    
    
    %% Set manifold
    problem.M = grassmannfactory(d, r);
    problem.ncostterms = N;
    problem.d = d;    
    batchsize = 10;
    checkperiod = 50;
    
    % == parameters ==
    d_mfd = r * (d - r);
    eps = 0.1;
    delta = 1e-3;  
    maxiter = 100* N * eps/sqrt(d_mfd * log(1/delta));
    maxiter = ceil(maxiter);
    sigma = 100 * maxiter * log(1/delta)/(N^2 * eps^2);
    sigma = sqrt(sigma);
    maxiter
    sigma
    

    
    % Define problem definitions
    problem.cost = @cost;
    function f = cost(U)
        W = mylsqfit(U, samples);
        f = 0.5*norm(indicator.*(U*W') - values, 'fro')^2;
        f = f/N;
    end
    
    problem.egrad = @egrad;
    function g = egrad(U)
        W = mylsqfit(U, samples);
        g = (indicator.*(U*W') - values)*W;
        g = g/N;
    end
    
    problem.partialegrad = @partialegrad;
    function g = partialegrad(U, idx_batch)
        g = zeros(d, r);
        m_batchsize = length(idx_batch);
        for ii = 1 : m_batchsize
            colnum = idx_batch(ii);
            w = mylsqfit(U, samples(colnum));
            indicator_vec = indicator(:, colnum);
            values_vec = values(:, colnum);
            g = g + (indicator_vec.*(U*w') - values_vec)*w;
        end
        g = g/m_batchsize;
    end

    function g = partialegrad_noise(U, idx_batch)
        g = zeros(d, r);
        m_batchsize = length(idx_batch);
        for ii = 1 : m_batchsize
            colnum = idx_batch(ii);
            w = mylsqfit(U, samples(colnum));
            indicator_vec = indicator(:, colnum);
            values_vec = values(:, colnum);
            g = g + (indicator_vec.*(U*w') - values_vec)*w;
        end
        g = g/m_batchsize;
        
        % sample noise using MHsample
        pdf = @gpdf;
        proprnd = @(xi) xi + randn(1,d*r);  
        noise = mhsample(zeros(1,d*r), 1, 'pdf',pdf, 'proprnd', proprnd, 'symmetric',1, 'burnin', 1000, 'thin', 1);
        noise = reshape(noise, [d, r]);
        noise = problem.M.proj(U, noise);
        g = g + noise;
        
        function q = gpdf(xi)
            Xi = reshape(xi, [d, r]);
            Xi = problem.M.proj(U, Xi);
            q = exp(-1/(2*sigma^2) * (problem.M.norm(U, Xi))^2);
        end
    end

    
    function stats = mc_mystatsfun(problem, U, stats)
        W = mylsqfit(U, samples);
        f_test = 0.5*norm(indicator_test.*(U*W') - values_test, 'fro')^2;
        f_test = f_test/N;
        stats.cost_test = f_test;
    end


    function W = mylsqfit(U, currentsamples)
        W = zeros(length(currentsamples), size(U, 2));
        for ii = 1 : length(currentsamples)
            % Pull out the relevant indices and revealed entries for this column
            IDX = currentsamples(ii).indicator;
            values_Omega = currentsamples(ii).values;
            U_Omega = U(IDX,:);
            
            % Solve a simple least squares problem to populate W.
            %OmegaUtUOmega = U_Omega'*U_Omega;
            OmegaUtUOmega = U_Omega'*U_Omega + 1e-10*eye(r);            
            W(ii,:) = (OmegaUtUOmega\(U_Omega'*values_Omega))';

        end
    end



    
%% Run algorithms    
    % Initialize
    Uinit = problem.M.rand();
    
    repeats = 5;
    train_sgd = [];
    train_psgd = [];
    test_sgd = [];
    test_psgd = [];
    for i_rep = 1 : repeats
        % R-SGD 
        clear options;
        options.batchsize = batchsize; 
        options.maxiter = maxiter; 
        options.checkperiod = checkperiod;
        options.verbosity = 1;
        options.stepsize_type = 'fix'; 
        options.stepsize_init = 1e-2;
        options.transport = 'ret_vector';   
        options.statsfun = @mc_mystatsfun;
        [~, infos_sgd, options_sgd] = RSGD(problem, Uinit, options);   
        %keyboard;
        train_sgd = [train_sgd; infos_sgd.cost];
        test_sgd = [test_sgd; infos_sgd.cost_test];

        % R-PSGD
        problem.partialegrad = @partialegrad_noise;
        [~, infos_psgd, options_psgd] = RSGD(problem, Uinit, options);
        train_psgd = [train_psgd; infos_psgd.cost];
        test_psgd = [test_psgd; infos_psgd.cost_test];
    end 
     
    iters = [infos_sgd.iter];
    save('lrmc_grass.mat', 'train_sgd', 'test_sgd', 'train_psgd', 'test_psgd', 'iters');
    

    %% Plots
    
    %{
    fs = 20;    
    % Train MSE 
    figure;
    semilogy([infos_sgd.iter], [infos_sgd.cost] * 2 * N / NumEntries_train,'-*','LineWidth',3,'Color', [76, 153, 0]/255);  hold on;
    semilogy([infos_psgd.iter], [infos_psgd.cost] * 2 * N / NumEntries_train, '-x', 'LineWidth',3,'Color', [255, 128, 0]/255); hold on;        hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Iteration','FontName','Arial','FontSize',fs);
    ylabel(ax1,'Train MSE','FontName','Arial','FontSize',fs);
    legend('RSGD', 'DP-RSGD');     
    
    % Test MSE 
    figure;
    semilogy([infos_sgd.iter], [infos_sgd.cost_test]  * 2 * N / NumEntries_test,'-*','LineWidth',3,'Color', [76, 153, 0]/255);  hold on;
    semilogy([infos_psgd.iter], [infos_psgd.cost_test]  * 2 * N / NumEntries_test,'-x','LineWidth',3,'Color', [255, 128, 0]/255);  hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Iteration','FontName','Arial','FontSize',fs);
    ylabel(ax1,'Test MSE','FontName','Arial','FontSize',fs);
    legend('RSGD', 'DP-RSGD');
    %}
    
  
end
