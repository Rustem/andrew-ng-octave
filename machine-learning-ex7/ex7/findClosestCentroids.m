function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
m = size(X, 1);
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i=1:rows(X),
    xi = X(i, :);
    min_sq_dist = inf;
    min_centroid = inf;
    for j=1:rows(centroids),
        mu = centroids(j, :);
        cur_sq_dist = sum((xi - mu) .^ 2);
        if min_sq_dist > cur_sq_dist,
            min_sq_dist = cur_sq_dist;
            min_centroid = j;
        end;
    end;
    idx(i) = min_centroid;
end;




% =============================================================

end

