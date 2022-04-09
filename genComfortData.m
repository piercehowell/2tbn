function ds = genComfortData(num_time_steps, init_ch, ...
    behavior, wch_b, wch_prev_ch, wch_spd, wch_leg, stdch, stdc)
    
    % Generate time-series data for simulated comfort model
    %
    %ds.data_all: struct array, containting the fileds
    %   - ch \in [0, 1]: hidden state for comfort of human
    %   - spd \in [0, 1]: speed characteristic of the robot
    %   - leg \in [0, 1]: legibility characteristic of the robot
    %   - c \in [0, 1]: subjective response from the human

    ds.params.num_time_steps = num_time_steps;
    ds.params.init_ch = min(max(init_ch, 0.0), 1.0);
    ds.params.behavior = behavior;
    ds.params.wch_prev_ch = wch_prev_ch;
    
    ds.data_all = repmat(struct('ch', [], 'spd', [], 'leg', [], 'c', []), ds.params.num_time_steps, 1);
    ch = init_ch;

    for i=1:ds.params.num_time_steps
        if(strcmp(behavior, 'spd_const_leg_const'))
            spd = 0.0;
            leg = 0.0;
        end
        
        % Update hidden state comfort of human
        prev_ch = ch;
        ch = wch_prev_ch*ch + spd*wch_spd + leg*wch_leg + normrnd(wch_b, stdch);

        if ch >= 1
            ch=1;
        elseif ch <= 0
            ch=0;
        end

        % observe subject comfort
        c = normrnd(ch, stdc);
        if (c > 1)
            c = 1;
        elseif (c < 0)
            c = 0;
        end
        
        % collect data
        ds.data_all(i).ch = ch;
        ds.data_all(i).spd = spd;
        ds.data_all(i).leg = leg;
        ds.data_all(i).c = c; 
    end

    



end 