
clear all;
close all;


%% Generate simulated observed data
num_time_steps = 50;
init_ch = 0.1; % initial comfort
behavior = 'spd_const_leg_const';
prop_wch_b = 0.0;
prop_wch_prev_ch = 1.03;
prop_wch_spd = 0.0;
prop_wch_leg = 0.0;
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
params.wch_b = 0.0001;
params.wch_prev_ch = 0.3;
params.wch_spd = 0.001;
params.wch_leg = 0.001;
params.stdch = 0.01;
params.stdc = 0.01;

model = ComfortModel(params);
model_dup = ComfortModel(params);
engine = HistoEngineEM(model, params, num_bins, [], ds, hard_em_type);
engine_dup = HistoEngineEM(model_dup, params, num_bins, [], ds, hard_em_type);


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
    state_vec_matrix = repmat(engine.cache.x_vec, size(ds, 1)+1, 1);
    exp_smoothed_states_last = mean(smoothing_probs_last .* state_vec_matrix, 2);
    map_states_last = engine.extractMAP();
    
end

%% Report results
if num_em_steps > 0
    state_vec_matrix = repmat(engine.cache.x_vec, size(ds, 1)+1, 1);

    init_params = engine.em.params_list{1};
    final_params = engine.em.params_list{end};
    engine_dup.reset()
    engine_dup.updateParams(init_params);
    cache = engine_dup.cache;
    cache.ds = dataset;
    [smoothing_probs_first, filtering_probs_first] = engine_dup.batchSmooth(ds, true);
    map_states_first = engine_dup.extractMAP();
    exp_smoothed_states_first = mean(smoothing_probs_first.*state_vec_matrix, 2);
    
    ljp_first = engine_dup.logJointProb(ds, exp_smoothed_states_first);
    
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
   plot(map_states_last)
   hold on;
   title("Expected Comfort Trajectory (Smoothed Inferences)")
   plot(map_states_first)
   plot([ds(1:end).ch])
   legend('After Training Model (EM)', 'Before Training Model (guessed weights)',...
       'True Model (simulated)')



end
