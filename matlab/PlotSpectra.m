function [] = PlotSpectra()

% Plot spectra for the core size series

% Specify variable parameters
N = 10; % number of particles
eta = 0.03; % volume fraction
N_type = 1; % number of particle types

% Load particle data
load('dipoles.mat', 'k', 'C');
close all

colors = parula(N_type);

    
% Gel extinction cross-section
ext = k.*imag(C(:,1)+C(:,5)+C(:,9))/3;
    
plot(k, ext, 'LineWidth', 3)
end
