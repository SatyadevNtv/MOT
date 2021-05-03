function [C, gamma, W, infos] = COT(Y, X, itermax, lambdaOT1, lambdaOT2)
	numX = size(X, 2);
	numY = size(Y, 2);

	e2 = ones(numX,1);
	e1 = ones(numY,1);

	d2 = size(X, 1);
	d1 = size(Y, 1);

	W = doubly_stochastic_general(rand(d1, d2), ones(d1,1)./d1, ones(d2,1)./d2);
	gamma = doubly_stochastic_general(rand(numY, numX), e1./numY, e2./numX);
	Wold = W;
	gammaold = gamma;

	t0 = tic();
	M0 = computeCOTcost1(Y, X, gamma);
	cost0 = W(:)'*M0(:) + lambdaOT1*sum(sum(gamma.*log(gamma))) + lambdaOT2*sum(sum(W.*log(W)));
	time0 = toc(t0);

	infos.cost = cost0;
	infos.time = time0;

	tstart = tic();
	for ii = 1:itermax

	  	C = computeCOTcost1(Y', X', W);
	  	gamma = doubly_stochastic_general(exp(-C/lambdaOT1), e1./numY, e2./numX, 1000);
		
	  	M = computeCOTcost1(Y, X, gamma);
	  	W = doubly_stochastic_general(exp(-M/lambdaOT2), ones(d1,1)./d1, ones(d2,1)./d2, 1000);
	  
	  	reldiff = norm(W-Wold, 'fro');
        fprintf('>>>>>>> CoOT iter %d rel difference: %e \n', ii, reldiff);
        Wold = W;
        gammaold = gamma;

        mycost = W(:)'*M(:) + lambdaOT1*sum(sum(gamma.*log(gamma))) + lambdaOT2*sum(sum(W.*log(W)));
        mytime = toc(tstart);

        infos.cost = [infos.cost, mycost];
        infos.time = [infos.time, mytime];

        if reldiff < 1e-8
        	break
        end
	end
	C = computeCOTcost1(Y', X', W);
	obj =  gamma(:)'*C(:);
	fprintf('>>>>>>> CoOT final transport objective: %e \n', obj);
end


function C = computeCOTcost1(A, B, P)
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

