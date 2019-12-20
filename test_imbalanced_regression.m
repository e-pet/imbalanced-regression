function test_imbalanced_regression(system_model, system_model_int, ...
    noise_model, data_dist, regression_methods, N, export_file)
%% Generate data
rng(1); % set seed

% Sample data points
xs = data_dist(N);

% Evaluate model at data points
ys = system_model(xs);

% Generate artificial measurements
zs = ys + noise_model(N);

%% Perform regression
% Perform regression
regression_models = cell(length(regression_methods), 1);
[N_reg, ~] = size(regression_methods);
times = zeros(N_reg, 1);
for ii=1:N_reg
    tic;
    regression_models{ii} = regression_methods{ii, 1}(xs, zs);
    times(ii) = toc;
end
fprintf('\nRegression execution times (%s): %s\n\n', ...
    sprintf("%s ", regression_methods{:, 2}), sprintf("%f ", times));

% Calculate ideal linear fit
ideal_model = linear_fit(system_model, system_model_int, xs);

%% Plot everything
[N, p] = size(xs);
xm = min(xs);
xM = max(xs);
figure;
if p == 1
    scatter(xs, zs, 25, 'p', 'filled');
    hold on;
    for ii=1:N_reg
        plot([xm, xM], ...
            [regression_models{ii}(xm), regression_models{ii}(xM)]);
    end
    plot([xm, xM], [ideal_model(xm), ideal_model(xM)]);
    x_eq = linspace(xm, xM);
    plot(x_eq, system_model(x_eq));
elseif p == 2
    CM = prism(N_reg+3);
    scatter3(xs(:, 1), xs(:, 2), zs, 'MarkerEdgeColor', CM(1,:));
    hold all;
%     xs_bounds = [xm(1), xm(2); ...
%                  xm(1), xM(2); ...
%                  xM(1), xm(2); ...
%                  xM(1), xM(2)];
%     X = reshape(xs_bounds(:, 1), [2,2]);
%     Y = reshape(xs_bounds(:, 2), [2,2]);
    [X, Y] = meshgrid(linspace(xm(1), xM(1), 10), linspace(xm(2), xM(2), 10));
    xs_grid = [reshape(X, [], 1), reshape(Y, [], 1)];
    for ii=1:N_reg
        zs_reg = regression_models{ii}(xs_grid);
        Z = reshape(zs_reg, [10, 10]);
        mesh(X, Y, Z, 'EdgeColor', CM(ii+1, :), 'FaceAlpha', 0);
    end
    zs_id = ideal_model(xs_grid);
    Z = reshape(zs_id, [10, 10]);
    mesh(X, Y, Z, 'EdgeColor', CM(end-1, :), 'FaceAlpha', 0);
    [X, Y] = meshgrid(linspace(xm(1), xM(1)), linspace(xm(2), xM(2)));
    xs_grid = [reshape(X, [], 1), reshape(Y, [], 1)];
    zs_system = system_model(xs_grid);
    Z = reshape(zs_system, [100, 100]);
    surf(X, Y, Z, 'FaceColor', CM(end, :), 'FaceAlpha', 1, ...
        'LineStyle', 'none');
else
    error('Cannot do plotting in more than 3 dimensions.');
end

labels = [{'Data'}, regression_methods{:, 2}, {'BLE', ...
    'f(x)'}];
legend(labels, 'Location', 'northwest');

if nargin >= 7 && ~isempty(export_file)
    [xs_sorted, idces] = sort(xs);
	export_matrix = [xs_sorted, zs(idces), nan * ones(length(xs), N_reg+1)];
	for ii = 1:N_reg
        export_matrix(1, 2+ii) = regression_models{ii}(xm);
        export_matrix(end, 2+ii) = regression_models{ii}(xM);
	end
    export_matrix(1, end) = ideal_model(xm);
    export_matrix(end, end) = ideal_model(xM);
    csvwrite(export_file, export_matrix);
end


function ideal_model = linear_fit(sys, sys_int, xs)
    xm = min(xs);
    xM = max(xs);
    lx = xM - xm;
    fM = sys(xM);
    fm = sys(xm);
    ideal_model = @(x) sum((sys_int(xM) - sys_int(xm)) ./ lx - (fM-fm)/2) ...
        + (fM-fm) ./ lx * (x-xm)';
        