function [] = RunCapacitanceSpectrum()

% Specify parameters
gamma = 0.05; % damping (relative to k_p)
eps_inf = 2; % high-freq dielectric constant

% Other parameters
xi = 0.5; % Ewald parameter

% Drude model
k = (0.01:0.01:0.8)'; % wave vectors (relative to plasma freq.)
eps_p = eps_inf-1./(k.^2+1i*k.*gamma); % dielectric function

% Specify the configuration
%x = [-1, 2.8, 0;
%     -4.6, 2.3, 0; 
%     -3.05, 1, 0; % upper left cluster
%     
%     -1.35, -0.45, 0;
%     0.4, -1.4, 0; % middle cluster
%     
%     4.3, 1.2, 0;
%     2.65, 0.1, 0;
%     4.5, -1., 0; % right cluster
%     
%     -4.2, -1.05, 0;
%     -2.75, -2.5, 0]; % lower left cluster
%box = [30, 30, 30];
x = readmatrix("Hex_Pos.txt");
box = readmatrix("Hex_Box.txt");
box(3) = 50;
N = size(x, 1);

% Assign all particles the same dielectric function
eps_in = repmat(eps_p.', N, 1);

% Output filename
outfile = sprintf('dipoles.mat'); % output file name

% Compute the capacitance spectrum
tic
[C, p] = CapacitanceTensorSpectrum(x, box, eps_in, xi);
toc

% Reshape capacitance and dipoles
C = permute([C(1,:,:), C(2,:,:), C(3,:,:)], [3,2,1]); % From 3-by-3-by-Nk to Nk-by-9
p = squeeze([p(:,:,1,:,:), p(:,:,2,:,:), p(:,:,3,:,:)]); % From N-3-3-Nk-Nframe to N-9-Nk-Nframes

% Save to file
save(outfile, 'gamma', 'eps_inf', 'N', 'box', 'x', 'k', 'eps_p', 'p', 'C')


end
