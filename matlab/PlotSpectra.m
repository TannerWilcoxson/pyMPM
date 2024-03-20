function [] = PlotSpectra()

% Plot spectra for the core size series

% Specify variable parameters
N = 10; % number of particles
eta = 0.03; % volume fraction
N_type = 1; % number of particle types

% Load particle data
load('dipoles.mat', 'k', 'eps_p', 'C');

disp(size(C))
C(1,:)

% Initializations
omega_LSPR_0 = zeros(N_type,1);
w_LSPR_0 = zeros(N_type,1);
omega_LSPR = zeros(N_type,1);
w_LSPR = zeros(N_type,1);

% Set up plot window
close all
h_ext = figure; hold on
colors = parula(N_type);

% Function for computing peak features by fitting a Gaussian
function [x_0, h, w] = peak_features(x, y)

    % Get data peak location
    [~, ind] = max(y);

    % Fit a Gaussian near the peak and extract height and location
    span = 2; % grab span number of points on either side of the peak
    fitob = fit(x(ind-span:ind+span), y(ind-span:ind+span), 'gauss1');
    x_0 = fitob.b1;
    h = fitob.a1;

    % Get the full width half max
    [~,~,w,prom] = findpeaks(y, x, 'WidthReference', 'halfheight');
    w = w(prom==max(prom)); % grab the width of the most prominent peak
    
end

i = 1;
% Single NP extinction
omega = k; % frequency (cm^-1)
k_p = 2*pi*100; % angular plasma wavenumber (m^-1)
ext_0 = 3*k_p*k.*imag((eps_p(:,i)-1)./(eps_p(:,i)+2)); % extinction cross-section per optical core volume (m^-1)
    
% Get single NP peak features
[omega_LSPR_0(i), ~, w_LSPR_0(i)] = peak_features(omega, ext_0);
   
% Gel extinction cross-section
ext = 3/(4*pi)*k_p*k.*imag(C(:,1)+C(:,5)+C(:,9))/3;
    
% Get gel peak features
[omega_LSPR(i), ~, w_LSPR(i)] = peak_features(omega, ext);
 
% Plot
figure(h_ext)
plot(omega, ext_0, '--', 'Color', colors(i,:), 'LineWidth', 3)
plot(omega, ext, 'Color', colors(i,:), 'LineWidth', 3)


end
