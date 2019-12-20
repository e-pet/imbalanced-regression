function probs = oKDE_probs(data, Dth)
% 
% This uses the online KDE method [1] for estimating the data likelihood
% and estimating the data weights wi = 1/p(xi).
%
% [1] Kristan Matej, Leonardis Ales and Skocaj Danijel, "Multivariate Online Kernel Density Estimation with Gaussian Kernels", 
% Pattern Recognition, 2011.
if nargin < 2
    Dth = 0.1; % set the compression value (see the paper [1] for better idea about what value to use)
end
% we'll assume your data is in "dat", which is DxN matrix, D being the
% dimension of datapoints and N being the number of datapoints
[D, N] = size(data);
if D > N
    data = data';
    [D, N] = size(data);
end

% Note: if you have access to your datapoints in advance, or if you have a
% fair sample from your data points, you can use this to prescale your data
% prior to learning in the oKDE. This is especially convenient when the
% scale in one dimension (or subsspace) is significantly larger than in the
% other. Note that the oKDE will take care of this prescaling internally,
% but I still suggest that if you are able to provide some scaling factors
% in advance, you should do so.
% WE ASSUME DATA TO BE PRESCALED ALREADY
prescaling = 0 ;
if prescaling
    [Mu, T] = getDataScaleTransform(data);
    dat = applyDataScaleTransform(data, Mu, T) ;    
end

% initialize your KDE. Again, the oKDE has many valves to make it robust
% against poor initialization, but, if you can, it is recomended that you
% initialize it with sufficiently large sample set (N_init). A rule of thumb would
% be to initialize with more samples than twice the dimensionality of your data.

N_init = 200; % how many samples will you use for initialization?
kde = executeOperatorIKDE([], 'input_data', data(:,1:N_init), 'add_input' );

kde = executeOperatorIKDE(kde, 'compressionClusterThresh', Dth) ;

% now you can add one sample at a time...
figure(1) ; clf ;
for i = N_init+1 : size(data, 2) 
    tic
    kde = executeOperatorIKDE(kde, 'input_data', data(:,i), 'add_input') ;
    t = toc ; 
    % print out some intermediate results
    msg = sprintf('Samples: %d ,Comps: %d , Last update time: %f ms\n', i , length(kde.pdf.w), t*1000 ) ; 
    title(msg) ; drawnow ;
end 
 
% your gaussian mixture model:
pdf_out = kde.pdf ;

% if you have prescaled your data, you will have to inverse the scaling of
% the estimated oKDE to project it back into the original space of your
% data. Note that from this point on, you can not update your pdf -- you
% will have to continue to update the KDE and inverse scaling again...
if prescaling    
  pdf_out = applyInvScaleTransformToPdf( pdf_out, Mu, T );
end

probs = evaluatePointsUnderPdf(pdf_out, data)';

end


