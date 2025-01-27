
clear all;
close all;


%% Generate simulated observed data
num_time_steps = 50;
init_ch = 0.4; % initial comfort
behavior = 'spd_const_leg_dec';
prop_wch_b = 0.0;
prop_wch_prev_ch = 1.00;
prop_wch_spd = 0.0;
prop_wch_leg = 0.02;
prop_stdch = 0.01;
prop_stdc = 0.01;

dataset = genComfortData(num_time_steps, init_ch, behavior, prop_wch_b, prop_wch_prev_ch, ...
    prop_wch_spd, prop_wch_leg, prop_stdch, prop_stdc);

ds = dataset.data_all;

%% Initialize histogram engine

num_em_steps = 20;
num_bins = 300;
hard_em_type = 'MAP';
em_params_eps_gain = 1; %?
params.wch_b = 0.0000;
params.wch_prev_ch = 0.5;
params.wch_spd = 0.0;
params.wch_leg = 0.0;
params.stdch = 0.01;
params.stdc = 0.01;

model = ComfortModel(params);
model_dup = ComfortModel(params);
engine = HistoEngineEM(model, params, num_bins, [], ds, hard_em_type);
engine_dup = HistoEngineEM(model_dup, params, num_bins, [], ds, hard_em_type);

[smoothing_probs_untrained, filtering_probs_untrained] = engine_dup.batchSmooth(ds, true);
map_states_untrained = engine_dup.extractMAP();

%% Run smoothing/filtering/MAP
inference_t = -1;
paramfit_t = -1;
if num_em_steps <= 0
    tic;
    [smoothing_probs, filtering_probs] = engine.batchSmooth(ds, true);
    map_states = engine.extractMAP();
    inference_t = toc;

    % Run log joint prob
    tic;
    ch_traj = map_states;
    logjointprob = engine.logJointProb(ds, ch_traj);
    ljp_t = toc;

    % Run param fitting
    tic;
    params_new = model.optimizeParams(ds, ch_traj, params);
    paramtit_t = toc;
%     model.plotHistogram(ds, engine.cache, ch_traj, ...
%         filtering_probs, smoothing_probs);

end

if num_em_steps > 0
    engine.reset()
    tic;
    engine.runEM(num_em_steps, false, em_params_eps_gain, true)
    [smoothing_probs_last, filtering_probs_last] = engine.batchSmooth(ds, true);
    map_states_trained = engine.extractMAP();
    
end

%% Report results
if num_em_steps > 0
    state_vec_matrix = repmat(engine.cache.x_vec, size(ds, 1)+1, 1);

    init_params = engine.em.params_list{1};
    final_params = engine.em.params_list{end};
    engine_dup.reset()
    engine_dup.updateParams(init_params);
    
    curr_params = init_params;
    fprintf('\n');
    fprintf('First Iter Params (vs GT | diff):\n');
    fprintf('- wch_b: %.4f (%.4f | %.4f)\n', curr_params.wch_b, prop_wch_b, abs(curr_params.wch_b - prop_wch_b));
    fprintf('- wch_prev_ch: %.4f (%.4f | %.4f\n', curr_params.wch_prev_ch, prop_wch_prev_ch, abs(curr_params.wch_prev_ch - prop_wch_prev_ch))
    fprintf('- wch_spd: %.4f (%.4f | %.4f)\n', curr_params.wch_spd, prop_wch_spd, abs(curr_params.wch_spd - prop_wch_spd));
    fprintf('- wch_leg: %.4f (%.4f | %.4f)\n', curr_params.wch_leg, prop_wch_leg, abs(curr_params.wch_leg - prop_wch_leg));
    fprintf('- stdch: %.4f (%.4f | %.4f)\n', curr_params.stdch, prop_stdch, abs(curr_params.stdch - prop_stdch));
    fprintf('- stdc: %.4f (%.4f | %.4f)\n', curr_params.stdc, prop_stdc, abs(curr_params.stdc - prop_stdc));
    
    curr_params = final_params;
    fprintf('\n');
    fprintf('Last Iter Params (vs GT | diff):\n');
    fprintf('- wch_b: %.4f (%.4f | %.4f)\n', curr_params.wch_b, prop_wch_b, abs(curr_params.wch_b - prop_wch_b));
    fprintf('- wch_prev_ch: %.4f (%.4f | %.4f\n', curr_params.wch_prev_ch, prop_wch_prev_ch, abs(curr_params.wch_prev_ch - prop_wch_prev_ch))
    fprintf('- wch_spd: %.4f (%.4f | %.4f)\n', curr_params.wch_spd, prop_wch_spd, abs(curr_params.wch_spd - prop_wch_spd));
    fprintf('- wch_leg: %.4f (%.4f | %.4f)\n', curr_params.wch_leg, prop_wch_leg, abs(curr_params.wch_leg - prop_wch_leg));
    fprintf('- stdch: %.4f (%.4f | %.4f)\n', curr_params.stdch, prop_stdch, abs(curr_params.stdch - prop_stdch));
    fprintf('- stdc: %.4f (%.4f | %.4f)\n', curr_params.stdc, prop_stdc, abs(curr_params.stdc - prop_stdc));
    
   figure;
   plot(map_states_trained)
   hold on;
   title("Estimated Comfort Trajectory (MAP Estimates)")
   plot(map_states_untrained)
   plot([ds(1:end).ch])
   legend('EM Trained Model', 'Before Training Model (guessed weights)',...
       'True Model (simulated)')



end
