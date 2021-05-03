function [Copt, gammaopt, Wopt, infos] = COT_with_MOT(S, T, options)
% Implements Co-optimal tranpsport with MOT + Manopt.
%
% S is the source data matrix of size d1 by m, i.e., 
% m data points of dimension d1.
%
% T is the target data matrix of size d2 by n, i.e.,
% n data points of dimension d2.
% 
% gamma is the sample-sample transport plan of size m by n.
% W is the feature-feature transport plan of size d1 by d2.
%
% If you use the code, please cite the following along with the Manopt paper.
%
% Co-optimal transport is based on the paper
% @inproceedings{NEURIPS2020_cc384c68,
%  author = {Titouan, Vayer and Redko, Ievgen and  Flamary, R\'{e}mi and Courty, Nicolas},
%  booktitle = {Advances in Neural Information Processing Systems},
%  title = {CO-Optimal Transport},
%  year = {2020}
% }
%
% @techreport{mishra2021manifold,
% title={Manifold optimization for optimal transport},
% author={Mishra, Bamdev and Dev, NTV and Kasai, Hiroyuki and Jawanpuria, Pratik},
% institution={arXiv preprint arXiv:2103.00902},
% year={2021}
% }
%

% Original author: Bamdev Mishra, 13 April 2021.
%

	% Local defaults for options
    localdefaults.maxiter = 250; % Max iterations.
    localdefaults.verbosity = 2; % Default: show the output.
    localdefaults.lambda_samples = 1e-5; % Regularization parameter
    localdefaults.lambda_features = 1e-5; % Regularization parameter
    localdefaults.tolgradnorm = 1e-6; % Absolute tolerance on Gradnorm.
    localdefaults.maxinner = 30; % Max inner iterations for the tCG step.
    localdefaults.method = 'CG'; % Other options: TR, SD. Default solver is conjugate gradient (CG).
    localdefaults.checkgradienthessian = false; % To check gradient and Hessian correctness.

    % Options
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);

	numT = size(T, 2); % Number of target points
	numS = size(S, 2); % Number of source points

	e2 = ones(numT,1); % Vector of ones
	e1 = ones(numS,1); % Vector of ones

	d2 = size(T, 1); % dimension of target feature space
	d1 = size(S, 1); % dimension of source feature space

	% Product manifold
	elements.gamma = multinomialdoublystochasticgeneralfactory(numS, numT, e1./numS, e2./numT);
	elements.W = multinomialdoublystochasticgeneralfactory(d1, d2, ones(d1,1)./d1, ones(d2,1)./d2);
	problem.M = productmanifold(elements);

	
	% Entropy regularizations
	lambda1 = options.lambda_samples;
	lambda2 = options.lambda_features;

	problem.cost = @cost;
	problem.egrad = @egrad;
	problem.ehess = @ehess;

	function [f, store] = cost(Z, store)
		gamma = Z.gamma;
		W = Z.W;
		if ~isfield(store, 'C')
            store.C = computeCOTcost(S', T', W);
        end
		C = store.C;
		f = gamma(:)'*C(:);
		f = f + lambda1*sum(sum(gamma.*log(gamma))) + lambda2*sum(sum(W.*log(W)));
	end

	function [g, store] = egrad(Z, store)
		gamma = Z.gamma;
		W = Z.W;
		if ~isfield(store, 'C')
            [~, store] = cost(Z, store);
        end

		C = store.C;
		M = computeCOTcost(S, T, gamma);

		g.gamma = C + lambda1*(1 + log(gamma));
		g.W = M + lambda2*(1 + log(W));
	end

	function [gdot, store] = ehess(Z, Zdot, store)
		gamma = Z.gamma;
		W = Z.W;
		if ~isfield(store, 'C')
            [~, store] = cost(Z, store);
        end

        gammadot = Zdot.gamma;
        Wdot = Zdot.W;

        Cdot = -S'*Wdot*T;
        Mdot = -S*gammadot*T';

        gdot.gamma = Cdot + lambda1*(gammadot./gamma);
        gdot.W = Mdot + lambda2*(Wdot./W);
	end

	if options.checkgradienthessian
        % Check correctness of gradient and Hessian
        checkgradient(problem);
        pause;
        checkhessian(problem);
        pause;
    end

    % Initialization point
	Zinit = [];

	if strcmpi('TR', options.method)
        % Riemannian trustregions
        [Zopt, ~, infos] = trustregions(problem, Zinit, options);  
    elseif strcmpi('SD', options.method)
        % Riemannian steepest descent
        [Zopt, ~, infos] = steepestdescent(problem, Zinit, options);
    elseif strcmpi('CG', options.method)
        %         % Riemannian conjugategradients
        %         options.beta_type = 'H-S';
        %         options.linesearch = @linesearch;
        %         options.ls_contraction_factor = .2;
        %         options.ls_optimism = 1.1;
        %         options.ls_suff_decr = 1e-4;
        %         options.ls_max_steps = 25;
        [Zopt, ~, infos] = conjugategradient(problem, Zinit, options);
    end
    
	gammaopt = Zopt.gamma;
	Wopt = Zopt.W;
	Copt = computeCOTcost(S', T', Wopt);
	obj = gammaopt(:)'*Copt(:);

	fprintf('Final transport objective value sans regularizations: %e \n', obj);
end

function C = computeCOTcost(A, B, P)
	[m, d1] = size(A);
	[n, d2] = size(B);

	% P is of size d1 by d2

	e1 = ones(d1, 1);
	e2 = ones(d2, 1);

	p1 = e1./d1;
	p2 = e2./d2;

	Const = ((A.^2)*p1)*ones(1,n) + ones(m,1)*(p2'*(B.^2)');

	C =  Const./2 - A*P*B';
end

