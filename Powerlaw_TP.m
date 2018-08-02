function [U,V,S] = Powerlaw_TP(A_input,k,beta,l1,l2,l3,l4,l5,iter,seed,use_GPU)
% Inputs:
	% A_input: n x n adjacency matrix
	% k: dimensionality
	% beta: coefficient for high-order proximity
	% l1,l2,l3,l4,l5: regularization parameters
	% iter: number of iterations
	% seed: random seed
	% use_GPU: whether to use GPU
% Outputs:
	% U: n x k matrix
	% V: k x k matrix
	% S: n x n matrix, sparse
	
% min_{U,V,S} ||(A - U * V * U' - S)||_F^2 + l1 * ||U||_F^2 + 
% l2 * ||V||_F^2 + l3 * ||S||_F^2 + l4 * ||S||_1

% ref: Wang, Xiao, et al. "Power-law Distribution Aware Trust Prediction." IJCAI. 2018.

A = A_input + beta * A_input * A_input;  % high-order proximity
n = size(A,1);

% Initialize
rng(seed);
if use_GPU == 1
	U = gpuArray(rand(n,k) * 0.1 - 0.05);
	V = gpuArray(rand(k,k) * 0.1 - 0.05);
	S = gpuArray(rand(n,n) * 0.1);
	P = gpuArray(rand(n,n));            % auxiliary variable
else
	U = rand(n,k) * 0.1 - 0.05;
	V = rand(k,k) * 0.1 - 0.05;
	S = rand(n,n) * 0.1;
	P = rand(n,n);            % auxiliary variable
end

temp_AS = A - S;            % temporary variables

for i = 1:iter
    % Update U
    temp1 = U * V;          % avoid duplicate calculation
    temp2 = U * V';         % avoid duplicate calculation
    delta_U = 2 * ( - temp_AS * temp2 - temp_AS' * temp1 + U * (temp2' * temp2) + U * temp1' * temp1 + l1 * U);
    clear temp1 temp2;
    alpha = BLS_U(U,delta_U,temp_AS,V,l1);  % search for step size
    U = U - alpha * delta_U;
    clear delta_U;
    % Update V
    delta_V = 2 * ( - U' * temp_AS * U + (((U' * U) * V) * U') * U + l2 * V);
    alpha = BLS_V(V,delta_V,temp_AS,U,l2);  % search for step size
    V = V - alpha * delta_V;
    clear delta_V;
    % Update S
        % Update M
        temp3 = S + P;
        M = sign(temp3) .* max(abs(temp3) - l4 / l5,0);
        clear temp3;
        % Update S
        S = ((A - U * (V * U')) + l5 * (M - P)) ./ (1 + l3 + l5);
        % Update P
        P = P + S - M;
    temp_AS = A - S;
    % can be modified here to terminate if changes of variable are smaller than threshold
end
if use_GPU == 1
    U = gather(U);
    V = gather(V);
    S = gather(S);
end
end