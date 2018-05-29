function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.100;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%c_cand_arr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%sigma_cand_arr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

%min_error = Inf;
%best_c_cand = Inf;
%best_sigma_cand = Inf;

%for c_cand=c_cand_arr,
 %   for sigma_cand=sigma_cand_arr,
  %      model = svmTrain(X, y, c_cand, @(x1, x2) gaussianKernel(x1, x2, sigma_cand));
   %     predictions = svmPredict(model, Xval);
    %    err = mean(double(predictions ~= yval));
     %   fprintf("VAlues: %f %f %f", err, c_cand, sigma_cand)
      %  if min_error > err;
       %     min_error = err;
        %    best_c_cand = c_cand;
         %   best_sigma_cand = sigma_cand;
        %end;
    %end;
%end;

%fprintf("Best candidates \n");
%disp(best_c_cand);
%disp(best_sigma_cand);
%C = best_c_cand;
%sigma = best_sigma_cand;


% =========================================================================

end
