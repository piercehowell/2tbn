% Dynamic Bayesian Network for Modeling Human Comfort during a Robot
% Interaction
%
% Propagation CPD parameters
% - wch_b: bias
% - wch_prev_ch:
% - wch_spd: speed
% - wch_leg: legibility
% - std_ch: addititve comfort noise (std)
%
%
% Observation of true comfort CPD
% - std_c: additive noise in true comfort estimate
classdef ComfortModel < handle
  properties (SetAccess='protected')
    params % structure containing graphical model parameters
  end
    
  % Optimization properties
  properties (Constant)
    optimize_prop_cpd_using_fmincon = true; % as opposed to linear least squares

    prop_optim_options = optimoptions('fmincon', ...
      'Algorithm', 'interior-point', ...
      'GradObj', 'on', 'Display', 'none');
    prop_optim_params_lb = [-1, -1, -1, -1, 0]; % Hand-chosen lower bounds for wch_b, wch_prev_ch, wch_spd, wch_leg, stdch
    prop_optim_params_ub = [1, 1, 1, 1, 20]; % Hand-chosen upper bounds for wch_b, wch_prev_ch, wch_spd, wch_leg, stdch
    
%     obs_dir_optim_options = optimoptions('fmincon', ...
%       'Algorithm', 'interior-point', ...
%       'GradObj', 'on', 'Display', 'none');
%     obs_dir_optim_params_lb = [1, 0, 0]; % Hand-chosen lower bounds for kd, od, bd
%     obs_dir_optim_params_ub = [1e4, 1, 1.0/3]; % Hand-chosen upper bounds for kd, od, bd

    % Hand-chosen tolerances for model parameters
    eps_wch_b = 1e-8;
    eps_wch_prev_ch = 1e-8;
    eps_wch_spd = 1e-8;
    eps_wch_leg = 1e-8;
    eps_stdch = 1e-10;
    eps_stdc = 1e-10;
  end

  methods(Static)

      % Assess whether 2 sets of parameters are sufficiently similar
      function same = compareParams(params_a, params_b, epsilon)
      if nargin < 3
        epsilon = 0;
      end
      
      same = ...
        (abs(params_a.wch_b - params_b.wch_b) < epsilon*ComfortModel.eps_wch_b) && ...
        (abs(params_a.wch_prev_ch - params_b.wch_prev_ch) < epsilon*ComfortModel.eps_wch_prev_ch) && ...
        (abs(params_a.wch_spd - params_b.wch_spd) < epsilon*ComfortModel.eps_wch_spd) && ...
        (abs(params_a.stdch - params_b.stdch) < epsilon*ComfortModel.eps_stdch) && ...
        (abs(params_a.wch_leg - params_b.wch_leg) < epsilon*ComfortModel.eps_wch_leg) && ...
        (abs(params_a.stdc - params_b.stdc) < epsilon*ComfortModel.eps_stdc);
    end % function compareParams()
    

    function params_new = optimizeParams(data_all, ch_states, params_old)
        % Hard-assignment EM
        params_new = params_old;
        num_time_steps = numel(data_all);
        if num_time_steps == 0
            error('ComfortModel:EmptyDataset', 'data_all has 0 samples')
        end

        ch_past = ch_states(1:end-1);
        ch_curr = ch_states(2:end);
        ch_diff = ch_curr - ch_past;

        % Optimize wch_b, wch_spd, stdch via constrained minimization of 
        % negative log joint prob or linear least squares
        
        spd = [data_all(:).spd]';
        leg = [data_all(:).leg]';
        
        u_curr = [ones(num_time_steps, 1), ch_past, spd, leg]; % concatenate propagation inputs

        if ComfortModel.optimize_prop_cpd_using_fmincon
            % Optimize using MATLAB's generalized non-linear function
            % minimization
            prop_optim_loss = @(params) loss_norm_cpd(ch_curr, u_curr, params);

            [prop_optim_newparams, ~, prop_optim_flag] = ...
                fmincon(prop_optim_loss, ...
                [params_old.wch_b, params_old.wch_prev_ch, params_old.wch_spd, params_old.wch_leg, params_old.stdch], ...
                [], [], [], [], ...
                ComfortModel.prop_optim_params_lb, ComfortModel.prop_optim_params_ub, ...
                [], ComfortModel.prop_optim_options);
            if prop_optim_flag < 0
                warning('ComfortModel:ParamOptimFailed',...
                    'Bounded optimization of propagate() params failed')
            % save the newly optimzied parameters
            else
                params_new.wch_b = prop_optim_newparams(1);
                params_new.wch_prev_ch = prop_optim_newparams(2);
                params_new.wch_spd = prop_optim_newparams(3);
                params_new.wch_leg = prop_optim_newparams(4);
                params_new.stdwch = prop_optim_newparams(5);
            end
        else
            error('ComfortModel:Optimization with linear least square (LLS) not implemented');
            
        end

    end %function optimizeParams()
  end % methods(Static)

  methods
    % Creates a new comfort model PGM object (constructor)
    function obj=ComfortModel(params)
        obj.params = params;
    end

    % Update parameters of robot model PGM
    function updateParams(obj, new_params)
        obj.params = new_params;
    end

    % Propagates belief of latent state to next time step
    function probs = propagate(obj, cache, data_curr, data_past)
      
        % Expect cache = struct with fields: ch_curr, ch_past
        % Expect data_curr, data_past = structs with fields: spd, leg

        % Compute linear gaussian CPD

        mu = obj.params.wch_prev_ch*cache.x_past + obj.params.wch_b + obj.params.wch_spd * data_curr.spd + ...
            obj.params.wch_leg * data_curr.leg;

        probs = boundedLinearGaussianCPD(cache.x_curr, mu, obj.params.stdch, ...
            cache.num_bins);
    
    end % function propagate()

    % Applies evidence to update belief on latent state
    function probs = observe(obj, cache, data_curr, data_past)
        % Expect cache = struct with fields: ch_curr, ch_past
        % Expect data_curr = struct with fields c
        data_past;
        probs = ones(size(cache.x_curr));
        probs = probs .* boundedLinearGaussianCPD( ...
            cache.x_curr, data_curr.c, obj.params.stdc, size(cache.x_curr, 1));
    end % function observe()


  end % methods
end % classdef ComfortModel
  