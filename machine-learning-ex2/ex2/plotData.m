function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

pos_idxs = find(y == 1);
neg_idxs = find(y == 0);

% Plot examples
plot(X(pos_idxs, 1), X(pos_idxs, 2), 'k+', 'LineWidth', 3, 'MarkerSize', 7);
plot(X(neg_idxs, 1), X(neg_idxs, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);







% =========================================================================



hold off;

end