function alpha = BLS_U(U,delta_U,A_S,V,l1)
% Backtracing line search 
temp = (A_S - U * V * U');
loss = sum(sum(temp .* temp)) + l1 * sum(sum(U .* U));  % omit other constant terms
temp_l = sqrt(sum(sum(delta_U .* delta_U)));            % length of gradient
alpha = 0.01;     % initial step size 
delta = 0.5;      % BLT parameter
gamma = 1e-4;     % BLT parameter
while 1
   U_new = U - alpha * delta_U;
   temp = (A_S - U_new * V * U_new');
   loss_new = sum(sum(temp .* temp)) + l1 * sum(sum(U_new .* U_new));
   if loss_new <= loss - gamma * alpha * temp_l
       break;
   end
   alpha = alpha * delta;
end
% disp(alpha);
end