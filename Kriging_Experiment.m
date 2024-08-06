% Sample size and Vectors
n = [307, 562, 701, 1019, 2129, 3001];
h = {
    [1, 42, 229, 101],
    [1, 53, 89, 221],
    [1, 82, 415, 382],
    [1, 71, 765, 865],
    [1, 766, 1281, 1906],
    [1, 174, 266, 1269]
};
row_vectors = {
    [232, 232, 232, 232],
    [551, 551, 551, 551],
    [0, 651, 142, 292],
    [0, 843, 884, 822],
    [0, 1770, 1491, 364],
    [1720, 1720, 1720, 1720]
};
s = 4;

% Preallocate results storage
results_GLP = zeros(3, 6);
results_GGLP = zeros(3, 6);

% Define function f5
f5 = @(x1, x2, x3, x4) 100*(x1.^2 - x2).^2 + (1 - x1).^2 + 90*(x4 - x3.^2).^2 + (1 - x3).^2 + 10.1*((x2 - 1).^2 + (x4 - 1).^2) + 19.8*(x2 - 1).*(x4 - 1);

% Generate new test points
rng(1); % Fix random seed for reproducibility
xnewf5 = -2 + 4 * rand(5000, 4);
ynewf5 = f5(xnewf5(:,1), xnewf5(:,2), xnewf5(:,3), xnewf5(:,4));

theta = [1 1 1 1]; 
lob = [1e-2 1e-2 1e-2 1e-2]; 
upb = [20 20 20 20];

% Loop over all datasets
for i = 1:length(n)
    % Generate GLP and GGLP data
    [Um_GLP, h_GLP] = generate_GLP(n(i), s, h{i});
    [Um_GGLP, h_GGLP] = generate_GGLP(n(i), s, h{i}, row_vectors{i});

    % Transform matrices
    Um_GLP_pracf5 = -2 + 4 / (n(i) - 1) * (Um_GLP - 1);
    Um_GGLP_pracf5 = -2 + 4 / (n(i) - 1) * (Um_GGLP - 1);

    % Calculate y values
    y_GLP = f5(Um_GLP_pracf5(:,1), Um_GLP_pracf5(:,2), Um_GLP_pracf5(:,3), Um_GLP_pracf5(:,4));
    y_GGLP = f5(Um_GGLP_pracf5(:,1), Um_GGLP_pracf5(:,2), Um_GGLP_pracf5(:,3), Um_GGLP_pracf5(:,4));

    % Check dimensions of input data
    disp(['Dataset ', num2str(i)]);
    disp(['Size of Um_GLP_pracf5: ', num2str(size(Um_GLP_pracf5))]);
    disp(['Size of y_GLP: ', num2str(size(y_GLP))]);
    disp(['Size of Um_GGLP_pracf5: ', num2str(size(Um_GGLP_pracf5))]);
    disp(['Size of y_GGLP: ', num2str(size(y_GGLP))]);

    % Fit models for different regression orders (Poly0, Poly1, and Quadratic)
    regressionOrders = {'regpoly0', 'regpoly1', 'regpoly2'};
    for j = 1:length(regressionOrders)
        % Create user options for oodacefit
        userOpts = struct('type', 'Kriging', 'regrFunc', regressionOrders{j}, 'corrFunc', @corrgauss, 'hp0', theta, 'hpBounds', [lob; upb]);

        % Display dimensions before fitting
        disp(['Regression order: ', regressionOrders{j}]);
        disp(['Size of Um_GLP_pracf5: ', num2str(size(Um_GLP_pracf5))]);
        disp(['Size of y_GLP: ', num2str(size(y_GLP))]);

        % Fit GLP model using oodacefit
        krige_GLP = oodacefit(Um_GLP_pracf5, y_GLP, userOpts);
        Yhat_GLP = predictor(xnewf5, krige_GLP);
        sMSE_GLP = sqrt(mean((ynewf5 - Yhat_GLP).^2));
        results_GLP(j, i) = sMSE_GLP;

        % Fit GGLP model using oodacefit
        krige_GGLP = oodacefit(Um_GGLP_pracf5, y_GGLP, userOpts);
        Yhat_GGLP = predictor(xnewf5, krige_GGLP);
        sMSE_GGLP = sqrt(mean((ynewf5 - Yhat_GGLP).^2));
        results_GGLP(j, i) = sMSE_GGLP;
    end
end

% Create table for results
Model = {'Poly0'; 'Poly1'; 'Quadratic'};
T_results = array2table([results_GLP, results_GGLP], 'VariableNames', {'GLP1', 'GGLP1', 'GLP2', 'GGLP2', 'GLP3', 'GGLP3', 'GLP4', 'GGLP4', 'GLP5', 'GGLP5', 'GLP6', 'GGLP6'}, 'RowNames', Model);

% Write table to Excel file
filename = 'GLP_GGLP_comparison.xlsx';
writetable(T_results, filename, 'WriteRowNames', true);

% Display message
disp(['Results saved to ', filename]);

% Generate Good Lattice Points (GLP)
function [Um, generator] = generate_GLP(n, s, generator)
    % Generate a vector from 1 to n
    nvec = 1:n;

    % Length of the generator
    m = length(generator);

    if s > m
        fprintf('ERROR: s should not be larger than the length of the generator!\n');
        Um = [];
        generator = [];
        return;
    end

    % Create the initial hypercube sample using the generator
    ih = kron(generator, nvec); % Ensure generator is an array
    Um = mod(ih, n); % Modulo operation
    Um(Um == 0) = n; % Replace 0s with n
    Um = reshape(Um, length(nvec), length(generator));
end

% Generate Generalized Good Lattice Points (GGLP)
function [Um_modified, generator] = generate_GGLP(n, s, generator, row_vector)
    % Generate a vector from 1 to n
    nvec = 1:n;

    % Length of the generator
    m = length(generator);

    if s > m
        fprintf('ERROR: s should not be larger than the length of the generator!\n');
        Um_modified = [];
        generator = [];
        return;
    end

    % Create the initial hypercube sample using the generator
    ih = kron(generator, nvec); % Ensure generator is an array
    Um = mod(ih, n); % Modulo operation
    Um(Um == 0) = n; % Replace 0s with n
    Um = reshape(Um, length(nvec), length(generator));

    % Add row_vector to each row of Um and take modulo n
    Um_modified = mod(Um + row_vector, n);
    Um_modified(Um_modified == 0) = n; % Replace 0s with n
end