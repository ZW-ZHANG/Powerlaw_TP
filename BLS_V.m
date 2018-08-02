function alpha = BLS_V(V,delta_V,A_S,U,l2)
% Backtracing line search 
temp = (A_S - U * V * U');
loss = sum(sum(temp .* temp)) + l2 * sum(sum(V .* V)); % omit other constant terms
temp_l = sqrt(sum(sum(delta_V .* delta_V)));           % length of gradient
alpha = 0.01;     % initial step size 
delta = 0.5;      % BLT parameter
gamma = 1e-4;     % BLT parameter
while 1
   V_new = V - alpha * delta_V;
   temp = (A_S - U * V_new * U');
   loss_new = sum(sum(temp .* temp)) + l2 * sum(sum(V_new .* V_new));
   if loss_new <= loss - gamma * alpha * temp_l
       break;
   end
   alpha = alpha * delta;
end
% disp(alpha);
end