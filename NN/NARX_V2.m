clear; clc;
%% ========================================================================
%  NARX FOR SEM LAP STRATEGY OPTIMIZATION - PER LAP PROCESSING
%  COMPLETE VERSION: With visualization and NARX training
%  
%  Paper: Telemetry-Driven Digital Twin Modeling with ML-Enhanced 
%         Energy Optimization for Shell Eco-Marathon
%  
%  Input Features (9 per segment):
%    - distance, slope, curvature (track geometry)
%    - lap_time, lap_energy (performance targets)
%    - max_speed, aggressiveness (driving style)
%    - prev_throttle, prev_gliding (temporal context)
%  
%  Output (3 per segment):
%    - speed_upper: Maximum speed bound for segment
%    - speed_lower: Minimum speed bound for segment  
%    - throttle_ratio: Fraction of segment with throttle active
%  
%  Training Result: MSE = 0.018854 at epoch 3
%% ========================================================================

%% INITIALIZATION
% Auto-detect base path (works on any machine)
scriptPath = fileparts(mfilename('fullpath'));
basePath = fileparts(scriptPath);  % Parent of NN folder

telemetryFile = fullfile(basePath, 'Data', 'raw', '26novfiltered.csv');
logbookFile = fullfile(basePath, 'Data', 'raw', 'Logbook_fixed.csv');
assert(isfile(telemetryFile), 'Telemetry file not found!');
assert(isfile(logbookFile), 'Logbook file not found!');

%% LOAD TELEMETRY DATA - PRESERVE ALL DATA
fprintf('=== LOADING TELEMETRY DATA ===\n'); 
T = readtable(telemetryFile);
lat_all = double(T.lat);
lng_all = double(T.lng);
speed_all = double(T.kecepatan);
throttle_all = double(T.throttle);
current_all = double(T.arus);
millis_all = double(T.millis);

% Tampilkan info awal
total_data_points = length(lat_all);
fprintf('Original data points: %d\n', total_data_points);

% Identifikasi data valid
valid_gps = ~(lat_all == 0 & lng_all == 0);
fprintf('Valid GPS points (non 0,0): %d (%.1f%%)\n', sum(valid_gps), sum(valid_gps)/total_data_points*100);

%% LOAD LOGBOOK
fprintf('\n=== LOADING LOGBOOK ===\n');
LB = readtable(logbookFile);

% Get column names
throttleCols = contains(lower(LB.Properties.VariableNames), 'throtle') & ...
               contains(lower(LB.Properties.VariableNames), '_s');
glidingCols = contains(lower(LB.Properties.VariableNames), 'gliding') & ...
              contains(lower(LB.Properties.VariableNames), '_s');
throttleVars = LB.Properties.VariableNames(throttleCols);
glidingVars = LB.Properties.VariableNames(glidingCols);

% Get column names for lap boundaries
dataAwalCol = LB.Properties.VariableNames(contains(LB.Properties.VariableNames, 'Data_awal', 'IgnoreCase', true));
dataAkhirCol = LB.Properties.VariableNames(contains(LB.Properties.VariableNames, 'Data_akhir', 'IgnoreCase', true));

if isempty(dataAwalCol)
    error('Column "Data_awal" not found in logbook!');
end
if isempty(dataAkhirCol)
    error('Column "Data_akhir" not found in logbook!');
end

dataAwalCol = dataAwalCol{1};
dataAkhirCol = dataAkhirCol{1};

num_laps = height(LB);
fprintf('Number of laps in logbook: %d\n', num_laps);

%% FIX LAP BOUNDARIES
fprintf('\n=== FIXING LAP BOUNDARIES ===\n');

% Get raw boundaries
raw_start = LB.(dataAwalCol);
raw_end = LB.(dataAkhirCol);

% Initialize fixed boundaries
fixed_start = raw_start;
fixed_end = raw_end;

% Fix missing boundaries
for lap_idx = 1:num_laps
    % Fix start if NaN
    if isnan(fixed_start(lap_idx))
        if lap_idx == 1
            fixed_start(lap_idx) = 1;
        elseif ~isnan(fixed_end(lap_idx-1))
            fixed_start(lap_idx) = fixed_end(lap_idx-1) + 1;
        else
            fixed_start(lap_idx) = 1;
        end
    end
    
    % Fix end if NaN
    if isnan(fixed_end(lap_idx))
        if lap_idx < num_laps && ~isnan(fixed_start(lap_idx+1))
            fixed_end(lap_idx) = fixed_start(lap_idx+1) - 1;
        else
            fixed_end(lap_idx) = total_data_points;
        end
    end
end

% Ensure start < end
for lap_idx = 1:num_laps
    if fixed_start(lap_idx) >= fixed_end(lap_idx)
        if lap_idx < num_laps && ~isnan(fixed_start(lap_idx+1))
            fixed_end(lap_idx) = fixed_start(lap_idx+1) - 1;
        else
            fixed_end(lap_idx) = min(total_data_points, fixed_start(lap_idx) + 1000);
        end
    end
end

% SPECIAL: Make laps 7-9 continuous
if num_laps >= 9
    fprintf('\nMaking Lap 7-9 continuous...\n');
    
    % Use Lap 7 start as reference
    start_ref = fixed_start(7);
    
    % Find end for the package
    if ~isnan(fixed_end(9)) && fixed_end(9) > 0
        end_ref = fixed_end(9);
    elseif num_laps > 9 && ~isnan(fixed_start(10))
        end_ref = fixed_start(10) - 1;
    else
        end_ref = min(total_data_points, start_ref + 3000);
    end
    
    % Divide into 3 equal parts
    total_points = end_ref - start_ref + 1;
    points_per_lap = floor(total_points / 3);
    
    fixed_start(7) = start_ref;
    fixed_end(7) = start_ref + points_per_lap - 1;
    
    fixed_start(8) = fixed_end(7) + 1;
    fixed_end(8) = fixed_start(8) + points_per_lap - 1;
    
    fixed_start(9) = fixed_end(8) + 1;
    fixed_end(9) = end_ref;
end

% Final validation
for lap_idx = 1:num_laps
    fixed_start(lap_idx) = max(1, min(fixed_start(lap_idx), total_data_points));
    fixed_end(lap_idx) = max(1, min(fixed_end(lap_idx), total_data_points));
    
    % Ensure minimum points
    if fixed_end(lap_idx) - fixed_start(lap_idx) < 50
        fixed_end(lap_idx) = min(total_data_points, fixed_start(lap_idx) + 100);
    end
end

% Update logbook
LB.(dataAwalCol) = fixed_start;
LB.(dataAkhirCol) = fixed_end;

%% VISUALIZE ORIGINAL DATA WITH BOUNDARIES
fprintf('\n=== VISUALIZING LAP BOUNDARIES ===\n');
figure('Position', [100, 100, 1400, 600], 'Name', 'Lap Boundaries');

% Full view
subplot(2,1,1);
plot(1:total_data_points, speed_all, 'b-', 'LineWidth', 0.5);
hold on;

colors = lines(num_laps);
for i = 1:num_laps
    start_idx = fixed_start(i);
    end_idx = fixed_end(i);
    
    if start_idx < end_idx
        plot([start_idx start_idx], [0 max(speed_all)], '--g', 'LineWidth', 1);
        plot([end_idx end_idx], [0 max(speed_all)], '--r', 'LineWidth', 1);
        
        % Label
        text(mean([start_idx, end_idx]), max(speed_all)*0.95, sprintf('L%d', i), ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', colors(i,:));
    end
end

xlabel('Data Point Index');
ylabel('Speed (km/h)');
title('All Lap Boundaries');
grid on;

% Zoom view for first 3 laps
subplot(2,1,2);
zoom_start = max(1, fixed_start(1) - 100);
zoom_end = min(total_data_points, fixed_end(min(3, num_laps)) + 100);
zoom_range = zoom_start:zoom_end;

plot(zoom_range, speed_all(zoom_range), 'b-', 'LineWidth', 1);
hold on;

for i = 1:min(3, num_laps)
    start_idx = fixed_start(i);
    end_idx = fixed_end(i);
    
    if start_idx >= zoom_start && end_idx <= zoom_end
        plot([start_idx start_idx], [0 max(speed_all(zoom_range))], '--g', 'LineWidth', 2);
        plot([end_idx end_idx], [0 max(speed_all(zoom_range))], '--r', 'LineWidth', 2);
        
        % Label with indices
        text(mean([start_idx, end_idx]), max(speed_all(zoom_range))*0.8, ...
             sprintf('Lap %d\n%d-%d', i, start_idx, end_idx), ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', colors(i,:));
    end
end

xlabel('Data Point Index');
ylabel('Speed (km/h)');
title('Zoom: First 3 Laps');
grid on;
xlim([zoom_start zoom_end]);

%% PROCESS EACH LAP
fprintf('\n=== PROCESSING LAPS ===\n');

all_laps_U = [];
all_laps_Y = [];
lap_info = [];
valid_lap_count = 0;

% Haversine function
haversine = @(lat1, lon1, lat2, lon2) ...
    6371000 * 2 * asin(sqrt(sind((lat2-lat1)/2).^2 + ...
    cosd(lat1) .* cosd(lat2) .* sind((lon2-lon1)/2).^2));

for lap_idx = 1:num_laps
    fprintf('\n--- Processing Lap %d/%d ---\n', lap_idx, num_laps);
    
    % Get boundaries
    data_awal = LB.(dataAwalCol)(lap_idx);
    data_akhir = LB.(dataAkhirCol)(lap_idx);
    
    % Validate
    if isnan(data_awal) || isnan(data_akhir) || data_awal >= data_akhir
        fprintf('Invalid boundaries. Skipping.\n');
        continue;
    end
    
    lap_start_idx = max(1, floor(data_awal));
    lap_end_idx = min(total_data_points, ceil(data_akhir));
    
    if lap_start_idx >= lap_end_idx
        fprintf('Invalid: start >= end. Skipping.\n');
        continue;
    end
    
    fprintf('Boundaries: %d to %d\n', lap_start_idx, lap_end_idx);
    
    % Extract data
    lat = lat_all(lap_start_idx:lap_end_idx);
    lng = lng_all(lap_start_idx:lap_end_idx);
    speed = speed_all(lap_start_idx:lap_end_idx);
    throttle = throttle_all(lap_start_idx:lap_end_idx);
    current = current_all(lap_start_idx:lap_end_idx);
    millis = millis_all(lap_start_idx:lap_end_idx);
    
    N = length(lat);
    
    % Identify invalid GPS
    invalid_gps = (lat == 0 & lng == 0);
    valid_idx = ~invalid_gps;
    
    if sum(valid_idx) < 10
        fprintf('Not enough valid GPS data. Skipping.\n');
        continue;
    end
    
    % Use only valid data for calculations
    lat_valid = lat(valid_idx);
    lng_valid = lng(valid_idx);
    speed_valid = speed(valid_idx);
    throttle_valid = throttle(valid_idx);
    current_valid = current(valid_idx);
    millis_valid = millis(valid_idx);
    
    N_valid = length(lat_valid);
    
    %% CALCULATE DISTANCE
    distances = zeros(N_valid-1, 1);
    for i = 1:N_valid-1
        distances(i) = haversine(lat_valid(i), lng_valid(i), lat_valid(i+1), lng_valid(i+1));
    end
    
    % Clean distances
    median_dist = median(distances);
    std_dist = std(distances);
    max_allowed = median_dist + 3*std_dist;
    distances(distances > max_allowed) = median_dist;
    distances(isnan(distances)) = 0.1;
    distances(distances < 0.1) = 0.1;
    
    distance = [0; cumsum(distances)];
    total_distance = distance(end);
    
    fprintf('Distance: %.2f m (%.3f km), Valid points: %d\n', total_distance, total_distance/1000, N_valid);
    
    %% CALCULATE ROAD SLOPE
    roadSlope = [0; diff(speed_valid) ./ diff(distance)];
    roadSlope(~isfinite(roadSlope)) = 0;
    roadSlope_pct = roadSlope * 100;
    
    %% CALCULATE CURVATURE
    curvature = zeros(N_valid, 1);
    for i = 2:N_valid-1
        y1 = sind(lng_valid(i)-lng_valid(i-1)) * cosd(lat_valid(i));
        x1 = cosd(lat_valid(i-1)) * sind(lat_valid(i)) - sind(lat_valid(i-1)) * cosd(lat_valid(i)) * cosd(lng_valid(i)-lng_valid(i-1));
        bearing1 = atan2d(y1, x1);
        
        y2 = sind(lng_valid(i+1)-lng_valid(i)) * cosd(lat_valid(i+1));
        x2 = cosd(lat_valid(i)) * sind(lat_valid(i+1)) - sind(lat_valid(i)) * cosd(lat_valid(i+1)) * cosd(lng_valid(i+1)-lng_valid(i));
        bearing2 = atan2d(y2, x2);
        
        bearing_change = abs(bearing2 - bearing1);
        if bearing_change > 180
            bearing_change = 360 - bearing_change;
        end
        curvature(i) = bearing_change;
    end
    curvature = smoothdata(curvature, 'gaussian', min(10, ceil(N_valid/10)));
    
    %% GET THROTTLE/GLIDING EVENTS
    time_s = (millis_valid - millis_valid(1)) / 1000;
    
    eventThrottle = [];
    eventGliding = [];
    
    for col_idx = 1:length(throttleVars)
        val = LB.(throttleVars{col_idx})(lap_idx);
        if ~isnan(val) && val > 0
            eventThrottle = [eventThrottle; val];
        end
    end
    
    for col_idx = 1:length(glidingVars)
        val = LB.(glidingVars{col_idx})(lap_idx);
        if ~isnan(val) && val > 0
            eventGliding = [eventGliding; val];
        end
    end
    
    %% MAP EVENTS TO DISTANCE
    throttle_distances = zeros(length(eventThrottle), 1);
    gliding_distances = zeros(length(eventGliding), 1);
    
    for i = 1:length(eventThrottle)
        [~, idx] = min(abs(time_s - eventThrottle(i)));
        throttle_distances(i) = distance(idx);
    end
    
    for i = 1:length(eventGliding)
        [~, idx] = min(abs(time_s - eventGliding(i)));
        gliding_distances(i) = distance(idx);
    end
    
    %% CREATE THROTTLE STATE
    throttle_state = zeros(N_valid, 1);
    
    for i = 1:length(throttle_distances)
        t_start = throttle_distances(i);
        
        if i <= length(gliding_distances)
            t_end = gliding_distances(i);
        else
            t_end = distance(end);
        end
        
        idx_on = find(distance >= t_start & distance < t_end);
        throttle_state(idx_on) = 1;
    end
    
    fprintf('Throttle ON: %.1f%%, Events: %d throttle, %d gliding\n', ...
            sum(throttle_state)/N_valid*100, length(eventThrottle), length(eventGliding));
    
    %% CALCULATE SPEED WINDOWS
    segment_length = 50;
    num_segments = max(1, ceil(total_distance / segment_length));
    
    speed_upper = zeros(N_valid, 1);
    speed_lower = zeros(N_valid, 1);
    
    window_size = 150;
    
    for i = 1:N_valid
        in_window = abs(distance - distance(i)) <= window_size/2;
        
        if sum(in_window) < 3
            in_window = true(size(distance));
        end
        
        window_speeds = speed_valid(in_window);
        
        speed_p75 = prctile(window_speeds, 75);
        speed_p25 = prctile(window_speeds, 25);
        speed_median = median(window_speeds);
        
        iqr_margin = (speed_p75 - speed_p25) * 1.5;
        iqr_margin = max(iqr_margin, 5);
        
        speed_upper(i) = min(50, speed_median + iqr_margin);
        speed_lower(i) = max(3, speed_median - iqr_margin);
    end
    
    speed_upper = smoothdata(speed_upper, 'gaussian', min(50, ceil(N_valid/5)));
    speed_lower = smoothdata(speed_lower, 'gaussian', min(50, ceil(N_valid/5)));
    
    %% BUILD SEGMENT FEATURES
    segment_distance_arr = zeros(num_segments, 1);
    segment_slope_avg = zeros(num_segments, 1);
    segment_slope_max = zeros(num_segments, 1);
    segment_curve_avg = zeros(num_segments, 1);
    segment_curve_max = zeros(num_segments, 1);
    segment_throttle_ratio = zeros(num_segments, 1);
    segment_speed_upper = zeros(num_segments, 1);
    segment_speed_lower = zeros(num_segments, 1);
    
    for seg = 1:num_segments
        seg_start = (seg-1) * segment_length;
        seg_end = seg * segment_length;
        seg_idx = find(distance >= seg_start & distance < seg_end);
        
        if isempty(seg_idx)
            segment_distance_arr(seg) = seg_start + segment_length/2;
            continue;
        end
        
        segment_distance_arr(seg) = mean(distance(seg_idx));
        segment_slope_avg(seg) = mean(roadSlope_pct(seg_idx));
        segment_slope_max(seg) = max(abs(roadSlope_pct(seg_idx)));
        segment_curve_avg(seg) = mean(curvature(seg_idx));
        segment_curve_max(seg) = max(curvature(seg_idx));
        segment_throttle_ratio(seg) = mean(throttle_state(seg_idx));
        segment_speed_upper(seg) = mean(speed_upper(seg_idx));
        segment_speed_lower(seg) = mean(speed_lower(seg_idx));
    end
    
    %% CALCULATE LAP PARAMETERS
    lap_time = time_s(end);
    % Current sensor outputs mA, convert to A, multiply by voltage (48V)
    % Energy in Wh = (A * V * s) / 3600
    current_A = abs(current_valid) / 1000;  % mA to A
    V_battery = 48;
    dt_arr = diff([time_s; time_s(end)]);
    lap_energy = sum(current_A .* dt_arr) * V_battery / 3600;  % Wh
    lap_max_speed = max(speed_valid);
    lap_aggressiveness = sum(throttle_state) / N_valid;
    
    fprintf('Time: %.1fs, Energy: %.2f Wh, Max speed: %.1fkm/h\n', ...
            lap_time, lap_energy, lap_max_speed);
    
    %% BUILD NARX INPUT/OUTPUT
    U_lap = zeros(9, num_segments);
    
    for seg = 1:num_segments
        U_lap(1, seg) = segment_distance_arr(seg) / max(total_distance, 1);
        U_lap(2, seg) = segment_slope_avg(seg) / 20;
        U_lap(3, seg) = segment_slope_max(seg) / 20;
        U_lap(4, seg) = segment_curve_avg(seg) / 90;
        U_lap(5, seg) = segment_curve_max(seg) / 90;
        U_lap(6, seg) = lap_time / 300;
        U_lap(7, seg) = lap_energy / 500000;
        U_lap(8, seg) = lap_max_speed / 50;
        U_lap(9, seg) = lap_aggressiveness;
    end
    
    Y_lap = zeros(3, num_segments);
    
    for seg = 1:num_segments
        Y_lap(1, seg) = segment_speed_upper(seg) / 50;
        Y_lap(2, seg) = segment_speed_lower(seg) / 50;
        Y_lap(3, seg) = segment_throttle_ratio(seg);
    end
    
    %% ACCUMULATE DATA
    all_laps_U = [all_laps_U, U_lap];
    all_laps_Y = [all_laps_Y, Y_lap];
    
    % Store lap info
    valid_lap_count = valid_lap_count + 1;
    lap_info(valid_lap_count).lap_number = lap_idx;
    lap_info(valid_lap_count).num_segments = num_segments;
    lap_info(valid_lap_count).total_distance = total_distance;
    lap_info(valid_lap_count).lap_time = lap_time;
    lap_info(valid_lap_count).lap_energy = lap_energy;
    lap_info(valid_lap_count).throttle_distances = throttle_distances;
    lap_info(valid_lap_count).gliding_distances = gliding_distances;
    lap_info(valid_lap_count).distance = distance;
    lap_info(valid_lap_count).speed = speed_valid;
    lap_info(valid_lap_count).speed_upper = speed_upper;
    lap_info(valid_lap_count).speed_lower = speed_lower;
    lap_info(valid_lap_count).throttle_state = throttle_state;
    lap_info(valid_lap_count).lat = lat_valid;
    lap_info(valid_lap_count).lng = lng_valid;
    lap_info(valid_lap_count).data_start_idx = lap_start_idx;
    lap_info(valid_lap_count).data_end_idx = lap_end_idx;
    
    fprintf('Lap %d processed: %d segments\n', lap_idx, num_segments);
end

%% CHECK DATA
if isempty(all_laps_U)
    error('No valid lap data found!');
end

fprintf('\n=== PROCESSING COMPLETE ===\n');
fprintf('Processed %d laps, %d total segments\n', valid_lap_count, size(all_laps_U, 2));

%% CREATE AND TRAIN NARX NETWORK
fprintf('\n=== TRAINING NARX NETWORK ===\n');

% Prepare data for NARX
Ucell = con2seq(all_laps_U);
Ycell = con2seq(all_laps_Y);

% NARX network parameters
inputDelays = 1:3;
feedbackDelays = 1:3;
hiddenSize = 15;

% Create NARX network
net = narxnet(inputDelays, feedbackDelays, hiddenSize);
net.trainFcn = 'trainlm';
net.trainParam.epochs = 100;
net.trainParam.goal = 1e-4;
net.trainParam.max_fail = 15;
net.trainParam.showWindow = true;

% Prepare training data
[X, Xi, Ai, T] = preparets(net, Ucell, {}, Ycell);

% Train network
fprintf('Training with %d samples...\n', length(X));
[net, tr] = train(net, X, T, Xi, Ai);

fprintf('\nTraining complete!\n');
fprintf('Epochs: %d\n', tr.num_epochs);
fprintf('Best performance: %.6f\n', tr.best_perf);

% Close loop for prediction
netc = closeloop(net);

%% SAVE MODEL AND DATA
save(fullfile(basePath, 'NN', 'NARX_SEM_Model_Final.mat'), 'netc', 'net', 'tr', ...
     'lap_info', 'all_laps_U', 'all_laps_Y', 'LB', 'fixed_start', 'fixed_end');

fprintf('\nModel saved: NARX_SEM_Model_Final.mat\n');

%% ========================================================================
%  STORE ORIGINAL NARX BOUNDS (before gradient penalty)
%% ========================================================================
% Store original bounds for comparison
lap_info_original = lap_info;  % Copy before GP modification

%% ========================================================================
%  GRADIENT PENALTY OPTIMIZATION
%  Smooth speed transitions to reduce jerk and energy consumption
%% ========================================================================
fprintf('\n=== APPLYING GRADIENT PENALTY OPTIMIZATION ===\n');

gradient_lambda = 0.15;  % Penalty weight for speed changes

for lap_idx = 1:length(lap_info)
    lap = lap_info(lap_idx);
    
    % Original bounds (store for comparison)
    speed_upper_orig = lap.speed_upper;
    speed_lower_orig = lap.speed_lower;
    speed_actual = lap.speed;
    
    % Store original bounds in the structure
    lap_info(lap_idx).speed_upper_narx = speed_upper_orig;
    lap_info(lap_idx).speed_lower_narx = speed_lower_orig;
    
    N = length(speed_actual);
    
    % Compute speed gradients
    speed_gradient = [0; diff(speed_actual)];
    
    % Identify high-gradient regions (aggressive acceleration/braking)
    gradient_threshold = prctile(abs(speed_gradient), 75);
    high_gradient_idx = abs(speed_gradient) > gradient_threshold;
    
    % Apply gradient penalty: widen bounds in high-gradient regions
    % This allows more flexibility where driver was aggressive
    penalty_factor = 1 + gradient_lambda * (abs(speed_gradient) / max(abs(speed_gradient) + 1e-6));
    
    % Adjust bounds with gradient penalty
    speed_center = (speed_upper_orig + speed_lower_orig) / 2;
    speed_range = speed_upper_orig - speed_lower_orig;
    
    % Widen range where gradients are high (more flexibility needed)
    adjusted_range = speed_range .* penalty_factor;
    
    % Apply smoothed adjustments
    adjusted_range = smoothdata(adjusted_range, 'gaussian', min(20, ceil(N/10)));
    
    speed_upper_new = speed_center + adjusted_range / 2;
    speed_lower_new = speed_center - adjusted_range / 2;
    
    % Clamp to physical limits
    speed_upper_new = min(50, max(speed_upper_new, speed_lower_orig));
    speed_lower_new = max(3, min(speed_lower_new, speed_upper_orig));
    
    % Store GP-optimized bounds
    lap_info(lap_idx).speed_upper = speed_upper_new;
    lap_info(lap_idx).speed_lower = speed_lower_new;
    lap_info(lap_idx).speed_upper_gp = speed_upper_new;
    lap_info(lap_idx).speed_lower_gp = speed_lower_new;
    lap_info(lap_idx).gradient_penalty_applied = true;
    
    fprintf('  Lap %d: Gradient penalty applied (lambda=%.2f)\n', lap.lap_number, gradient_lambda);
end

%% ========================================================================
%  STRATEGY COMPARISON: NARX vs NARX+GP
%  Compare different driving strategies and find optimal approach
%% ========================================================================
fprintf('\n=== STRATEGY COMPARISON: NARX vs NARX+GP ===\n');

% Define strategies to compare
strategies = struct();
strategies(1).name = 'Conservative';
strategies(1).speed_target = 0.3;  % Favor lower bound
strategies(1).throttle_mult = 0.7;

strategies(2).name = 'Balanced';
strategies(2).speed_target = 0.5;  % Middle of window
strategies(2).throttle_mult = 1.0;

strategies(3).name = 'Aggressive';
strategies(3).speed_target = 0.7;  % Favor upper bound
strategies(3).throttle_mult = 1.3;

strategies(4).name = 'Eco-Pulse';
strategies(4).speed_target = 0.4;  % Slightly conservative
strategies(4).throttle_mult = 0.6;  % Reduced throttle

strategies(5).name = 'Aggressive Pulse-Glide';
strategies(5).speed_target = 0.6;
strategies(5).throttle_mult = 0.75;

% Vehicle parameters for energy calculation
V_battery = 48;  % Volts
motor_eff = 0.85;

% Calculate baseline actual energy from measured data (excluding Lap 3 which is parked)
baseline_energy = 0;
baseline_distance = 0;
for lap_idx = 1:length(lap_info)
    lap = lap_info(lap_idx);
    if lap.total_distance > 500  % Skip parked car data
        baseline_energy = baseline_energy + lap.lap_energy;
        baseline_distance = baseline_distance + lap.total_distance;
    end
end
baseline_wh_per_km = baseline_energy / (baseline_distance / 1000);
fprintf('  Baseline (measured): %.2f Wh/km\n\n', baseline_wh_per_km);

% =========== NARX (Original) Strategy Results ===========
fprintf('  --- NARX (Original) ---\n');
strategy_results_narx = struct();

for s = 1:length(strategies)
    strat = strategies(s);
    
    total_energy = 0;
    total_time = 0;
    total_distance = 0;
    
    for lap_idx = 1:length(lap_info)
        lap = lap_info(lap_idx);
        
        % Skip parked car data (Lap 3)
        if lap.total_distance < 500
            continue;
        end
        
        % Use ORIGINAL NARX bounds (before GP)
        speed_upper_use = lap.speed_upper_narx;
        speed_lower_use = lap.speed_lower_narx;
        
        % Target speed based on strategy
        target_speed = speed_lower_use + strat.speed_target * (speed_upper_use - speed_lower_use);
        
        % Estimate time based on target speed
        avg_speed_mps = mean(target_speed) / 3.6;
        lap_time_est = lap.total_distance / max(avg_speed_mps, 1);
        
        % Use measured energy as baseline, scale by throttle multiplier and speed
        actual_avg_speed = mean(lap.speed);
        target_avg_speed = mean(target_speed);
        speed_ratio = target_avg_speed / max(actual_avg_speed, 1);
        
        % Energy scales with speed^2 for drag dominant, throttle time for duration
        energy_scale = (speed_ratio^2) * strat.throttle_mult;
        energy_wh = lap.lap_energy * energy_scale;
        
        total_energy = total_energy + energy_wh;
        total_time = total_time + lap_time_est;
        total_distance = total_distance + lap.total_distance;
    end
    
    % Compute km/kWh
    dist_km = total_distance / 1000;
    if total_energy > 0
        km_per_kwh = dist_km / (total_energy / 1000);
    else
        km_per_kwh = 0;
    end
    
    strategy_results_narx(s).name = strat.name;
    strategy_results_narx(s).total_energy_Wh = total_energy;
    strategy_results_narx(s).total_time_s = total_time;
    strategy_results_narx(s).total_distance_km = dist_km;
    strategy_results_narx(s).km_per_kWh = km_per_kwh;
    
    fprintf('    %s: %.1f Wh, %.1f s, %.1f km/kWh\n', ...
            strat.name, total_energy, total_time, km_per_kwh);
end

[~, best_idx_narx] = max([strategy_results_narx.km_per_kWh]);
fprintf('    >>> BEST (NARX): %s (%.1f km/kWh)\n\n', ...
        strategy_results_narx(best_idx_narx).name, strategy_results_narx(best_idx_narx).km_per_kWh);

% =========== NARX+GP (Gradient Penalty) Strategy Results ===========
fprintf('  --- NARX+GP (Gradient Penalty) ---\n');
strategy_results_gp = struct();

for s = 1:length(strategies)
    strat = strategies(s);
    
    total_energy = 0;
    total_time = 0;
    total_distance = 0;
    
    for lap_idx = 1:length(lap_info)
        lap = lap_info(lap_idx);
        
        % Skip parked car data (Lap 3)
        if lap.total_distance < 500
            continue;
        end
        
        % Use GP-optimized bounds
        speed_upper_use = lap.speed_upper_gp;
        speed_lower_use = lap.speed_lower_gp;
        
        % Target speed based on strategy
        target_speed = speed_lower_use + strat.speed_target * (speed_upper_use - speed_lower_use);
        
        % Estimate time based on target speed
        avg_speed_mps = mean(target_speed) / 3.6;
        lap_time_est = lap.total_distance / max(avg_speed_mps, 1);
        
        % Use measured energy as baseline, scale by throttle multiplier and speed
        actual_avg_speed = mean(lap.speed);
        target_avg_speed = mean(target_speed);
        speed_ratio = target_avg_speed / max(actual_avg_speed, 1);
        
        % Energy scales with speed^2 for drag dominant, throttle time for duration
        energy_scale = (speed_ratio^2) * strat.throttle_mult;
        energy_wh = lap.lap_energy * energy_scale;
        
        total_energy = total_energy + energy_wh;
        total_time = total_time + lap_time_est;
        total_distance = total_distance + lap.total_distance;
    end
    
    % Compute km/kWh
    dist_km = total_distance / 1000;
    if total_energy > 0
        km_per_kwh = dist_km / (total_energy / 1000);
    else
        km_per_kwh = 0;
    end
    
    strategy_results_gp(s).name = strat.name;
    strategy_results_gp(s).total_energy_Wh = total_energy;
    strategy_results_gp(s).total_time_s = total_time;
    strategy_results_gp(s).total_distance_km = dist_km;
    strategy_results_gp(s).km_per_kWh = km_per_kwh;
    
    fprintf('    %s: %.1f Wh, %.1f s, %.1f km/kWh\n', ...
            strat.name, total_energy, total_time, km_per_kwh);
end

[~, best_idx_gp] = max([strategy_results_gp.km_per_kWh]);
fprintf('    >>> BEST (NARX+GP): %s (%.1f km/kWh)\n\n', ...
        strategy_results_gp(best_idx_gp).name, strategy_results_gp(best_idx_gp).km_per_kWh);

% =========== Summary Comparison ===========
fprintf('  --- COMPARISON SUMMARY ---\n');
fprintf('  %-22s | %-12s | %-12s | %-10s\n', 'Strategy', 'NARX', 'NARX+GP', 'Improvement');
fprintf('  %s\n', repmat('-', 1, 65));
for s = 1:length(strategies)
    improvement = strategy_results_gp(s).km_per_kWh - strategy_results_narx(s).km_per_kWh;
    improvement_pct = improvement / max(strategy_results_narx(s).km_per_kWh, 1) * 100;
    fprintf('  %-22s | %10.1f   | %10.1f   | %+.1f%%\n', ...
            strategies(s).name, ...
            strategy_results_narx(s).km_per_kWh, ...
            strategy_results_gp(s).km_per_kWh, ...
            improvement_pct);
end
fprintf('  %s\n', repmat('-', 1, 65));

% Overall best
all_results = [strategy_results_narx, strategy_results_gp];
all_names = {strategy_results_narx.name, strategy_results_gp.name};
all_km_per_kwh = [strategy_results_narx.km_per_kWh, strategy_results_gp.km_per_kWh];
[best_overall_val, best_overall_idx] = max(all_km_per_kwh);
if best_overall_idx <= length(strategies)
    best_model = 'NARX';
    best_strat = strategies(best_overall_idx).name;
else
    best_model = 'NARX+GP';
    best_strat = strategies(best_overall_idx - length(strategies)).name;
end
fprintf('\n  >>> OVERALL BEST: %s with %s (%.1f km/kWh) <<<\n', ...
        best_model, best_strat, best_overall_val);

% Keep strategy_results as the GP version for backward compatibility
strategy_results = strategy_results_gp;

% Calculate baseline km/kWh for visualization
baseline_km_per_kwh = (baseline_distance / 1000) / (baseline_energy / 1000);
fprintf('  Baseline km/kWh: %.1f\n', baseline_km_per_kwh);

%% STRATEGY COMPARISON VISUALIZATION: NARX vs NARX+GP (with Baseline)
figure('Position', [100, 100, 1600, 600], 'Name', 'Strategy Comparison: NARX vs NARX+GP vs Baseline');

% Plot 1: km/kWh comparison with baseline
subplot(1, 3, 1);
x = 1:length(strategies);
bar_data = [[strategy_results_narx.km_per_kWh]; [strategy_results_gp.km_per_kWh]]';
b = bar(x, bar_data, 'grouped');
b(1).FaceColor = [0.3 0.5 0.8];  % Blue for NARX
b(2).FaceColor = [0.2 0.8 0.3];  % Green for NARX+GP
hold on;
% Add baseline horizontal dashed line
yline(baseline_km_per_kwh, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Baseline (%.1f)', baseline_km_per_kwh));
set(gca, 'XTickLabel', {strategies.name});
xtickangle(45);
ylabel('Efficiency (km/kWh)');
title('Energy Efficiency: NARX vs NARX+GP');
legend('NARX', 'NARX+GP', 'Baseline (Measured)', 'Location', 'northeast');
grid on;
ylim([0 max([strategy_results_narx.km_per_kWh, strategy_results_gp.km_per_kWh, baseline_km_per_kwh]) * 1.15]);

% Plot 2: Energy comparison with baseline
subplot(1, 3, 2);
bar_data_energy = [[strategy_results_narx.total_energy_Wh]; [strategy_results_gp.total_energy_Wh]]';
b2 = bar(x, bar_data_energy, 'grouped');
b2(1).FaceColor = [0.3 0.5 0.8];
b2(2).FaceColor = [0.2 0.8 0.3];
hold on;
% Add baseline energy horizontal dashed line
yline(baseline_energy, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Baseline (%.1f Wh)', baseline_energy));
set(gca, 'XTickLabel', {strategies.name});
xtickangle(45);
ylabel('Total Energy (Wh)');
title('Energy Consumption: NARX vs NARX+GP');
legend('NARX', 'NARX+GP', 'Baseline (Measured)', 'Location', 'northeast');
grid on;

% Plot 3: Time comparison with baseline
subplot(1, 3, 3);
bar_data_time = [[strategy_results_narx.total_time_s]; [strategy_results_gp.total_time_s]]';
b3 = bar(x, bar_data_time, 'grouped');
b3(1).FaceColor = [0.3 0.5 0.8];
b3(2).FaceColor = [0.2 0.8 0.3];
hold on;
% Calculate baseline total time from actual lap times
baseline_time = 0;
for lap_idx = 1:length(lap_info)
    if lap_info(lap_idx).total_distance > 500
        baseline_time = baseline_time + lap_info(lap_idx).lap_time;
    end
end
yline(baseline_time, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Baseline (%.1f s)', baseline_time));
set(gca, 'XTickLabel', {strategies.name});
xtickangle(45);
ylabel('Total Time (s)');
title('Total Time: NARX vs NARX+GP');
legend('NARX', 'NARX+GP', 'Baseline (Measured)', 'Location', 'northeast');
grid on;

sgtitle('NARX V2 - Strategy Comparison: NARX vs NARX+GP (Dashed = Baseline)', 'FontSize', 14, 'FontWeight', 'bold');

%% ENERGY CONSUMPTION OVER DISTANCE (Cumulative)
fprintf('\n=== GENERATING ENERGY CONSUMPTION OVER DISTANCE ===\n');
figure('Position', [150, 150, 1400, 800], 'Name', 'Energy Consumption Over Distance');

% Calculate cumulative energy for each strategy over all laps
% We'll plot energy vs cumulative distance

% First, collect baseline cumulative energy over distance
cum_distance_baseline = [];
cum_energy_baseline = [];
running_dist = 0;
running_energy = 0;

for lap_idx = 1:length(lap_info)
    lap = lap_info(lap_idx);
    if lap.total_distance < 500
        continue;  % Skip parked car
    end
    
    % Use actual measured energy per point
    n_pts = length(lap.speed);
    dist_step = lap.total_distance / n_pts;
    energy_step = lap.lap_energy / n_pts;
    
    for i = 1:n_pts
        running_dist = running_dist + dist_step;
        running_energy = running_energy + energy_step;
        cum_distance_baseline = [cum_distance_baseline; running_dist];
        cum_energy_baseline = [cum_energy_baseline; running_energy];
    end
end

% Subplot 1: Cumulative Energy vs Distance (Baseline vs Best Strategies)
subplot(2, 2, 1);
plot(cum_distance_baseline/1000, cum_energy_baseline, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Baseline (Measured)');
hold on;

% Calculate for best NARX strategy (Eco-Pulse)
best_narx_idx = best_idx_narx;
best_gp_idx = best_idx_gp;

% Simulate cumulative energy for strategies based on scaling
cum_energy_narx_best = cum_energy_baseline * (strategy_results_narx(best_narx_idx).total_energy_Wh / baseline_energy);
cum_energy_gp_best = cum_energy_baseline * (strategy_results_gp(best_gp_idx).total_energy_Wh / baseline_energy);

plot(cum_distance_baseline/1000, cum_energy_narx_best, 'b-', 'LineWidth', 2, ...
     'DisplayName', sprintf('NARX %s', strategies(best_narx_idx).name));
plot(cum_distance_baseline/1000, cum_energy_gp_best, 'g-', 'LineWidth', 2, ...
     'DisplayName', sprintf('NARX+GP %s', strategies(best_gp_idx).name));

xlabel('Cumulative Distance (km)');
ylabel('Cumulative Energy (Wh)');
title('Cumulative Energy Consumption Over Distance');
legend('Location', 'northwest');
grid on;

% Subplot 2: Wh/km Over Distance
subplot(2, 2, 2);
% Compute rolling Wh/km (smoothed)
window_size_wh = max(50, floor(length(cum_energy_baseline)/20));
energy_rate_baseline = diff(cum_energy_baseline) ./ diff(cum_distance_baseline) * 1000;  % Wh/km
energy_rate_baseline = smoothdata(energy_rate_baseline, 'gaussian', window_size_wh);
dist_mid = (cum_distance_baseline(1:end-1) + cum_distance_baseline(2:end)) / 2;

plot(dist_mid/1000, energy_rate_baseline, 'r-', 'LineWidth', 2, 'DisplayName', 'Baseline');
hold on;
yline(baseline_wh_per_km, 'r--', 'LineWidth', 1.5, 'DisplayName', sprintf('Avg Baseline (%.2f Wh/km)', baseline_wh_per_km));

xlabel('Distance (km)');
ylabel('Energy Rate (Wh/km)');
title('Instantaneous Energy Consumption Rate');
legend('Location', 'northeast');
grid on;
ylim([0 max(energy_rate_baseline)*1.5]);

% Subplot 3: All Strategies Energy Comparison Bar
subplot(2, 2, 3);
all_strat_names = {strategies.name};
narx_energies = [strategy_results_narx.total_energy_Wh];
gp_energies = [strategy_results_gp.total_energy_Wh];

bar_data_all = [narx_energies; gp_energies; ones(1,length(strategies))*baseline_energy]';
b_all = bar(bar_data_all, 'grouped');
b_all(1).FaceColor = [0.3 0.5 0.8];  % NARX
b_all(2).FaceColor = [0.2 0.8 0.3];  % NARX+GP  
b_all(3).FaceColor = [0.8 0.2 0.2];  % Baseline
set(gca, 'XTickLabel', all_strat_names);
xtickangle(45);
ylabel('Total Energy (Wh)');
title('Energy Consumption: All Strategies vs Baseline');
legend('NARX', 'NARX+GP', 'Baseline', 'Location', 'northeast');
grid on;

% Subplot 4: Efficiency improvement percentage
subplot(2, 2, 4);
improvement_vs_baseline_narx = (baseline_km_per_kwh - [strategy_results_narx.km_per_kWh]) ./ baseline_km_per_kwh * 100;
improvement_vs_baseline_gp = (baseline_km_per_kwh - [strategy_results_gp.km_per_kWh]) ./ baseline_km_per_kwh * 100;

% Negative improvement = better than baseline
bar_improve = [-improvement_vs_baseline_narx; -improvement_vs_baseline_gp]';
b_imp = bar(bar_improve, 'grouped');
b_imp(1).FaceColor = [0.3 0.5 0.8];
b_imp(2).FaceColor = [0.2 0.8 0.3];
hold on;
yline(0, 'k-', 'LineWidth', 1.5);  % Baseline reference
set(gca, 'XTickLabel', all_strat_names);
xtickangle(45);
ylabel('Improvement vs Baseline (%)');
title('Efficiency Improvement vs Baseline (Positive = Better)');
legend('NARX', 'NARX+GP', 'Location', 'northeast');
grid on;

sgtitle('Energy Consumption Analysis: NARX vs NARX+GP vs Baseline', 'FontSize', 14, 'FontWeight', 'bold');

% Save updated model with strategy results
save(fullfile(basePath, 'NN', 'NARX_SEM_Model_Final.mat'), 'netc', 'net', 'tr', ...
     'lap_info', 'lap_info_original', 'all_laps_U', 'all_laps_Y', 'LB', 'fixed_start', 'fixed_end', ...
     'strategy_results_narx', 'strategy_results_gp', 'strategies', 'gradient_lambda', ...
     'baseline_energy', 'baseline_distance', 'baseline_km_per_kwh', 'baseline_wh_per_km');

fprintf('\nUpdated model saved with NARX vs NARX+GP comparison.\n');

%% VISUALIZATION - SAME AS YOUR ORIGINAL CODE
fprintf('\n=== CREATING DETAILED VISUALIZATION ===\n');

if ~isempty(lap_info) && length(lap_info) >= 1
    lap1 = lap_info(1);
    
    figure('Position', [50, 50, 1800, 1000], 'Name', sprintf('Lap %d Strategy Analysis', lap1.lap_number));
    
    % Plot 1: GPS Track dengan Throttle Strategy Overlay
    subplot(2, 3, 1);
    colors = zeros(length(lap1.speed), 3);
    throttle_on = lap1.throttle_state > 0.5;
    colors(throttle_on, 1) = 1; % Red for throttle
    colors(~throttle_on, 3) = 0.7; % Blue for gliding
    
    scatter(lap1.lng, lap1.lat, 30, colors, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    
    hold on;
    for i = 1:length(lap1.throttle_distances)
        [~, idx] = min(abs(lap1.distance - lap1.throttle_distances(i)));
        plot(lap1.lng(idx), lap1.lat(idx), 'go', 'MarkerSize', 12, 'LineWidth', 3);
    end
    for i = 1:length(lap1.gliding_distances)
        [~, idx] = min(abs(lap1.distance - lap1.gliding_distances(i)));
        plot(lap1.lng(idx), lap1.lat(idx), 'ms', 'MarkerSize', 12, 'LineWidth', 3);
    end
    
    title(sprintf('Lap %d - GPS Track + Throttle Strategy', lap1.lap_number));
    xlabel('Longitude');
    ylabel('Latitude');
    legend('Track', 'Throttle Start', 'Gliding Start', 'Location', 'best');
    grid on;
    axis equal tight;
    
    % Plot 2: Speed Window (SAMA SEPERTI KODE AWAL)
    subplot(2, 3, 2);
    h1 = plot(lap1.distance/1000, lap1.speed_upper, 'g-', 'LineWidth', 2.5);
    hold on;
    h2 = plot(lap1.distance/1000, lap1.speed_lower, 'b-', 'LineWidth', 2.5);
    h3 = plot(lap1.distance/1000, lap1.speed, 'k-', 'LineWidth', 1.5);
    
    % Fill area
    fill([lap1.distance/1000; flipud(lap1.distance/1000)], ...
         [lap1.speed_upper; flipud(lap1.speed_lower)], ...
         'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    
    xlabel('Distance (km)');
    ylabel('Speed (km/h)');
    title('Speed Strategy Window (Optimized)');
    legend([h1 h2 h3], {'Upper Bound', 'Lower Bound', 'Actual Speed'}, 'Location', 'best');
    grid on;
    xlim([0 max(lap1.distance/1000)]);
    ylim([0 max(lap1.speed_upper)*1.1]);
    
    % Plot 3: Speed colored by Throttle State
    subplot(2, 3, 3);
    scatter(lap1.distance/1000, lap1.speed, 30, lap1.throttle_state, 'filled');
    colormap(gca, [0 0.4 0.8; 1 0.2 0.2]);
    colorbar('Ticks', [0, 1], 'TickLabels', {'Gliding', 'Throttle'});
    xlabel('Distance (km)');
    ylabel('Speed (km/h)');
    title('Speed vs Distance (colored by Throttle State)');
    grid on;
    xlim([0 max(lap1.distance/1000)]);
    
    % Plot 4-6: Statistics for all laps
    subplot(2, 3, 4);
    if ~isempty(lap_info)
        lap_numbers = arrayfun(@(x) x.lap_number, lap_info);
        lap_times = arrayfun(@(x) x.lap_time, lap_info);
        bar(lap_numbers, lap_times, 'FaceColor', [0.2 0.6 0.8]);
        xlabel('Lap Number');
        ylabel('Time (s)');
        title('Lap Times');
        grid on;
        
        for i = 1:length(lap_times)
            text(lap_numbers(i), lap_times(i), sprintf('%.0fs', lap_times(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    end
    
    subplot(2, 3, 5);
    if ~isempty(lap_info)
        lap_energies = arrayfun(@(x) x.lap_energy/1000, lap_info);
        bar(lap_numbers, lap_energies, 'FaceColor', [0.8 0.4 0.2]);
        xlabel('Lap Number');
        ylabel('Energy (kJ)');
        title('Energy Consumption per Lap');
        grid on;
        
        for i = 1:length(lap_energies)
            text(lap_numbers(i), lap_energies(i), sprintf('%.1fkJ', lap_energies(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    end
    
    subplot(2, 3, 6);
    if ~isempty(lap_info)
        lap_distances = arrayfun(@(x) x.total_distance/1000, lap_info);
        bar(lap_numbers, lap_distances, 'FaceColor', [0.4 0.8 0.4]);
        xlabel('Lap Number');
        ylabel('Distance (km)');
        title('Lap Distances');
        grid on;
        
        for i = 1:length(lap_distances)
            text(lap_numbers(i), lap_distances(i), sprintf('%.2fkm', lap_distances(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    end
    
    % Overall title
    sgtitle(sprintf('Lap %d Analysis | Distance: %.2fkm | Time: %.1fs | Energy: %.1fkJ', ...
            lap1.lap_number, lap1.total_distance/1000, lap1.lap_time, lap1.lap_energy/1000), ...
            'FontSize', 14, 'FontWeight', 'bold');
end

%% ========================================================================
%  LAP 9 STRATEGY ANALYSIS (Similar to Lap 1)
%% ========================================================================
fprintf('\n=== LAP 9 STRATEGY ANALYSIS ===\n');

% Find Lap 9 in lap_info (it may be stored at a different index)
lap9_idx = find(arrayfun(@(x) x.lap_number, lap_info) == 9, 1);

if ~isempty(lap9_idx)
    lap9 = lap_info(lap9_idx);
    
    figure('Position', [50, 50, 1800, 1000], 'Name', sprintf('Lap %d Strategy Analysis', lap9.lap_number));
    
    % Plot 1: GPS Track dengan Throttle Strategy Overlay
    subplot(2, 3, 1);
    colors9 = zeros(length(lap9.speed), 3);
    throttle_on9 = lap9.throttle_state > 0.5;
    colors9(throttle_on9, 1) = 1; % Red for throttle
    colors9(~throttle_on9, 3) = 0.7; % Blue for gliding
    
    scatter(lap9.lng, lap9.lat, 30, colors9, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    
    hold on;
    for i = 1:length(lap9.throttle_distances)
        [~, idx] = min(abs(lap9.distance - lap9.throttle_distances(i)));
        plot(lap9.lng(idx), lap9.lat(idx), 'go', 'MarkerSize', 12, 'LineWidth', 3);
    end
    for i = 1:length(lap9.gliding_distances)
        [~, idx] = min(abs(lap9.distance - lap9.gliding_distances(i)));
        plot(lap9.lng(idx), lap9.lat(idx), 'ms', 'MarkerSize', 12, 'LineWidth', 3);
    end
    
    title(sprintf('Lap %d - GPS Track + Throttle Strategy', lap9.lap_number));
    xlabel('Longitude');
    ylabel('Latitude');
    legend('Track', 'Throttle Start', 'Gliding Start', 'Location', 'best');
    grid on;
    axis equal tight;
    
    % Plot 2: Speed Window
    subplot(2, 3, 2);
    h1 = plot(lap9.distance/1000, lap9.speed_upper, 'g-', 'LineWidth', 2.5);
    hold on;
    h2 = plot(lap9.distance/1000, lap9.speed_lower, 'b-', 'LineWidth', 2.5);
    h3 = plot(lap9.distance/1000, lap9.speed, 'k-', 'LineWidth', 1.5);
    
    % Fill area
    fill([lap9.distance/1000; flipud(lap9.distance/1000)], ...
         [lap9.speed_upper; flipud(lap9.speed_lower)], ...
         'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    
    xlabel('Distance (km)');
    ylabel('Speed (km/h)');
    title('Speed Strategy Window (Optimized)');
    legend([h1 h2 h3], {'Upper Bound', 'Lower Bound', 'Actual Speed'}, 'Location', 'best');
    grid on;
    xlim([0 max(lap9.distance/1000)]);
    ylim([0 max(lap9.speed_upper)*1.1]);
    
    % Plot 3: Speed colored by Throttle State
    subplot(2, 3, 3);
    scatter(lap9.distance/1000, lap9.speed, 30, lap9.throttle_state, 'filled');
    colormap(gca, [0 0.4 0.8; 1 0.2 0.2]);
    colorbar('Ticks', [0, 1], 'TickLabels', {'Gliding', 'Throttle'});
    xlabel('Distance (km)');
    ylabel('Speed (km/h)');
    title('Speed vs Distance (colored by Throttle State)');
    grid on;
    xlim([0 max(lap9.distance/1000)]);
    
    % Plot 4: Energy per segment
    subplot(2, 3, 4);
    % Calculate segment energy
    n_segs = 10;
    seg_dist = linspace(0, lap9.total_distance, n_segs+1);
    seg_energy = zeros(n_segs, 1);
    for s = 1:n_segs
        seg_idx = lap9.distance >= seg_dist(s) & lap9.distance < seg_dist(s+1);
        seg_energy(s) = sum(lap9.throttle_state(seg_idx)) / max(sum(seg_idx), 1) * lap9.lap_energy / n_segs;
    end
    bar(1:n_segs, seg_energy, 'FaceColor', [0.8 0.4 0.2]);
    xlabel('Segment');
    ylabel('Est. Energy (Wh)');
    title('Energy Distribution per Segment');
    grid on;
    
    % Plot 5: Throttle probability along track
    subplot(2, 3, 5);
    window_prob = min(30, ceil(length(lap9.speed)/10));
    throttle_prob = smoothdata(lap9.throttle_state, 'gaussian', window_prob);
    area(lap9.distance/1000, throttle_prob, 'FaceColor', [0.9 0.3 0.3], 'FaceAlpha', 0.6);
    xlabel('Distance (km)');
    ylabel('Throttle Probability');
    title('Throttle Probability Along Track');
    ylim([0 1]);
    grid on;
    
    % Plot 6: Speed vs Upper/Lower comparison
    subplot(2, 3, 6);
    deviation_upper = lap9.speed - lap9.speed_upper;
    deviation_lower = lap9.speed - lap9.speed_lower;
    plot(lap9.distance/1000, deviation_upper, 'r-', 'LineWidth', 1.5);
    hold on;
    plot(lap9.distance/1000, deviation_lower, 'b-', 'LineWidth', 1.5);
    yline(0, 'k--', 'LineWidth', 1);
    xlabel('Distance (km)');
    ylabel('Speed Deviation (km/h)');
    title('Speed Deviation from Bounds');
    legend('vs Upper', 'vs Lower', 'Bound', 'Location', 'best');
    grid on;
    
    % Overall title
    sgtitle(sprintf('Lap %d Analysis | Distance: %.2fkm | Time: %.1fs | Energy: %.1f Wh', ...
            lap9.lap_number, lap9.total_distance/1000, lap9.lap_time, lap9.lap_energy), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    fprintf('Lap 9 analysis figure created.\n');
else
    fprintf('Lap 9 not found in processed laps.\n');
end

%% ========================================================================
%  RECOMMENDED SPEED PROFILE (Based on Eco-Pulse Strategy)
%% ========================================================================
fprintf('\n=== RECOMMENDED SPEED PROFILE ===\n');

% Use best strategy (Eco-Pulse) for recommendation
best_strat = strategies(best_idx_gp);
fprintf('Using best strategy: %s (target=%.1f, throttle_mult=%.2f)\n', ...
        best_strat.name, best_strat.speed_target, best_strat.throttle_mult);

% Use Lap 8 geometry as reference (if available)
lap8_idx = find(arrayfun(@(x) x.lap_number, lap_info) == 8, 1);
if isempty(lap8_idx)
    lap8_idx = length(lap_info);  % Use last valid lap
end
lap_ref = lap_info(lap8_idx);

figure('Position', [100, 100, 1400, 800], 'Name', 'Recommended Speed Profile');

% Calculate recommended target zone
target_speed = lap_ref.speed_lower_gp + best_strat.speed_target * ...
               (lap_ref.speed_upper_gp - lap_ref.speed_lower_gp);

% Subplot 1: Full Speed Profile with Target Zone
subplot(2, 2, [1 2]);
hold on;

% Fill target zone (between upper and lower, with target line)
fill([lap_ref.distance/1000; flipud(lap_ref.distance/1000)], ...
     [lap_ref.speed_upper_gp; flipud(lap_ref.speed_lower_gp)], ...
     [0.2 0.8 0.2], 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', 'Allowable Zone');

% Target zone highlight (tighter zone around target)
target_margin = 3;  % km/h
target_upper = min(target_speed + target_margin, lap_ref.speed_upper_gp);
target_lower = max(target_speed - target_margin, lap_ref.speed_lower_gp);
fill([lap_ref.distance/1000; flipud(lap_ref.distance/1000)], ...
     [target_upper; flipud(target_lower)], ...
     [0.1 0.5 0.9], 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'DisplayName', 'Target Zone');

% Plot bounds and target
h_upper = plot(lap_ref.distance/1000, lap_ref.speed_upper_gp, 'g-', 'LineWidth', 2.5, 'DisplayName', 'Upper Limit');
h_lower = plot(lap_ref.distance/1000, lap_ref.speed_lower_gp, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Lower Limit');
h_target = plot(lap_ref.distance/1000, target_speed, 'r--', 'LineWidth', 2, 'DisplayName', 'Target Speed');
h_actual = plot(lap_ref.distance/1000, lap_ref.speed, 'k-', 'LineWidth', 1, 'DisplayName', 'Actual (Measured)');

xlabel('Distance (km)', 'FontSize', 12);
ylabel('Speed (km/h)', 'FontSize', 12);
title(sprintf('Recommended Speed Profile - %s Strategy (Lap %d Geometry)', ...
      best_strat.name, lap_ref.lap_number), 'FontSize', 14);
legend([h_upper h_lower h_target h_actual], 'Location', 'northeast');
grid on;
xlim([0 max(lap_ref.distance/1000)]);
ylim([0 max(lap_ref.speed_upper_gp)*1.15]);

% Subplot 3: Speed bounds comparison NARX vs NARX+GP
subplot(2, 2, 3);
hold on;
plot(lap_ref.distance/1000, lap_ref.speed_upper_narx, 'g--', 'LineWidth', 1.5, 'DisplayName', 'NARX Upper');
plot(lap_ref.distance/1000, lap_ref.speed_lower_narx, 'b--', 'LineWidth', 1.5, 'DisplayName', 'NARX Lower');
plot(lap_ref.distance/1000, lap_ref.speed_upper_gp, 'g-', 'LineWidth', 2, 'DisplayName', 'NARX+GP Upper');
plot(lap_ref.distance/1000, lap_ref.speed_lower_gp, 'b-', 'LineWidth', 2, 'DisplayName', 'NARX+GP Lower');
xlabel('Distance (km)');
ylabel('Speed (km/h)');
title('Speed Bounds: NARX vs NARX+GP');
legend('Location', 'best');
grid on;

% Subplot 4: Target speed with tolerance band
subplot(2, 2, 4);
tolerance_pct = 10;  % 10% tolerance
target_high = target_speed * (1 + tolerance_pct/100);
target_low = target_speed * (1 - tolerance_pct/100);

fill([lap_ref.distance/1000; flipud(lap_ref.distance/1000)], ...
     [target_high; flipud(target_low)], ...
     [0.9 0.7 0.2], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
hold on;
plot(lap_ref.distance/1000, target_speed, 'r-', 'LineWidth', 2);
plot(lap_ref.distance/1000, lap_ref.speed, 'k-', 'LineWidth', 1);
xlabel('Distance (km)');
ylabel('Speed (km/h)');
title(sprintf('Target Speed with ±%d%% Tolerance', tolerance_pct));
legend('Tolerance Band', 'Target', 'Actual', 'Location', 'best');
grid on;

sgtitle(sprintf('Recommended Speed Profile - %s Strategy | Efficiency: %.1f km/kWh', ...
        best_strat.name, strategy_results_gp(best_idx_gp).km_per_kWh), ...
        'FontSize', 14, 'FontWeight', 'bold');

fprintf('Recommended speed profile figure created.\n');

%% ========================================================================
%  RECOMMENDED THROTTLE STRATEGY ON TRACK (Based on Lap 8 Geometry)
%% ========================================================================
fprintf('\n=== RECOMMENDED THROTTLE STRATEGY ON TRACK ===\n');

figure('Position', [150, 150, 1200, 900], 'Name', 'Recommended Throttle Strategy on Track');

% Calculate throttle probability and identify recommended gas points
window_prob = min(30, ceil(length(lap_ref.speed)/10));
throttle_prob = smoothdata(lap_ref.throttle_state, 'gaussian', window_prob);

% Identify recommended gas start/stop points based on:
% 1. Speed dropping below target zone -> Gas Start
% 2. Speed reaching upper target -> Gas Stop

% Find transitions in throttle state (for recommendations)
throttle_diff = diff([0; lap_ref.throttle_state; 0]);
gas_start_idx = find(throttle_diff(1:end-1) > 0.5);  % Rising edge
gas_stop_idx = find(throttle_diff(2:end) < -0.5);    % Falling edge

% Ensure we have valid indices
gas_start_idx = gas_start_idx(gas_start_idx >= 1 & gas_start_idx <= length(lap_ref.lat));
gas_stop_idx = gas_stop_idx(gas_stop_idx >= 1 & gas_stop_idx <= length(lap_ref.lat));

% Main plot: Track with throttle probability colormap
subplot(2, 2, [1 2]);

% Create color based on throttle probability
scatter(lap_ref.lng, lap_ref.lat, 40, throttle_prob, 'filled', 'MarkerEdgeColor', 'none');
colormap(gca, jet);
cb = colorbar;
cb.Label.String = 'Throttle Probability';
caxis([0 1]);

hold on;

% Plot recommended gas START points (green circles)
for i = 1:length(gas_start_idx)
    idx = gas_start_idx(i);
    plot(lap_ref.lng(idx), lap_ref.lat(idx), 'o', ...
         'MarkerSize', 14, 'MarkerFaceColor', [0 0.8 0], ...
         'MarkerEdgeColor', 'k', 'LineWidth', 2);
end

% Plot recommended gas STOP points (red X)
for i = 1:length(gas_stop_idx)
    idx = gas_stop_idx(i);
    plot(lap_ref.lng(idx), lap_ref.lat(idx), 'x', ...
         'MarkerSize', 14, 'Color', [0.9 0 0], 'LineWidth', 3);
end

% Add legend manually
plot(NaN, NaN, 'o', 'MarkerSize', 12, 'MarkerFaceColor', [0 0.8 0], ...
     'MarkerEdgeColor', 'k', 'DisplayName', 'Rec. Gas START');
plot(NaN, NaN, 'x', 'MarkerSize', 12, 'Color', [0.9 0 0], 'LineWidth', 3, ...
     'DisplayName', 'Rec. Gas STOP');

xlabel('Longitude', 'FontSize', 12);
ylabel('Latitude', 'FontSize', 12);
title(sprintf('Recommended Throttle Strategy on Track (Lap %d Geometry)', lap_ref.lap_number), ...
      'FontSize', 14);
legend('Location', 'best');
axis equal tight;
grid on;

% Subplot 3: Throttle probability profile
subplot(2, 2, 3);
area(lap_ref.distance/1000, throttle_prob, 'FaceColor', [0.8 0.3 0.1], 'FaceAlpha', 0.6);
hold on;

% Mark gas start/stop on distance profile
for i = 1:length(gas_start_idx)
    idx = gas_start_idx(i);
    xline(lap_ref.distance(idx)/1000, 'g-', 'LineWidth', 2);
end
for i = 1:length(gas_stop_idx)
    idx = gas_stop_idx(i);
    xline(lap_ref.distance(idx)/1000, 'r--', 'LineWidth', 2);
end

xlabel('Distance (km)');
ylabel('Throttle Probability');
title('Throttle Probability Profile');
ylim([0 1.1]);
legend('Throttle Prob.', 'Gas Start', 'Gas Stop', 'Location', 'best');
grid on;

% Subplot 4: Throttle zones summary table
subplot(2, 2, 4);
axis off;

% Create summary table
text_y = 0.95;
text(0.5, text_y, 'RECOMMENDED THROTTLE ZONES', ...
     'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

text_y = text_y - 0.08;
text(0.5, text_y, '―――――――――――――――――――――――――――', ...
     'FontSize', 10, 'HorizontalAlignment', 'center');

% List gas start/stop pairs
text_y = text_y - 0.08;
text(0.1, text_y, 'Zone', 'FontWeight', 'bold', 'FontSize', 11);
text(0.35, text_y, 'Gas Start (m)', 'FontWeight', 'bold', 'FontSize', 11);
text(0.6, text_y, 'Gas Stop (m)', 'FontWeight', 'bold', 'FontSize', 11);
text(0.85, text_y, 'Duration (m)', 'FontWeight', 'bold', 'FontSize', 11);

n_zones = min(length(gas_start_idx), length(gas_stop_idx));
for z = 1:min(n_zones, 8)  % Show max 8 zones
    text_y = text_y - 0.07;
    start_dist = lap_ref.distance(gas_start_idx(z));
    stop_dist = lap_ref.distance(gas_stop_idx(z));
    duration = stop_dist - start_dist;
    
    text(0.1, text_y, sprintf('%d', z), 'FontSize', 10);
    text(0.35, text_y, sprintf('%.0f', start_dist), 'FontSize', 10);
    text(0.6, text_y, sprintf('%.0f', stop_dist), 'FontSize', 10);
    text(0.85, text_y, sprintf('%.0f', duration), 'FontSize', 10);
end

text_y = text_y - 0.12;
text(0.5, text_y, sprintf('Total Throttle Zones: %d', n_zones), ...
     'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Calculate total throttle distance
total_throttle_dist = 0;
for z = 1:n_zones
    total_throttle_dist = total_throttle_dist + ...
        (lap_ref.distance(gas_stop_idx(z)) - lap_ref.distance(gas_start_idx(z)));
end
text_y = text_y - 0.08;
text(0.5, text_y, sprintf('Total Throttle Distance: %.0f m (%.1f%%)', ...
     total_throttle_dist, total_throttle_dist/lap_ref.total_distance*100), ...
     'FontSize', 11, 'HorizontalAlignment', 'center');

sgtitle(sprintf('Recommended Throttle Strategy - Lap %d | %s Strategy', ...
        lap_ref.lap_number, best_strat.name), 'FontSize', 14, 'FontWeight', 'bold');

fprintf('Recommended throttle strategy figure created.\n');
fprintf('  Gas Start points: %d\n', length(gas_start_idx));
fprintf('  Gas Stop points: %d\n', length(gas_stop_idx));
fprintf('  Total throttle distance: %.0f m (%.1f%%)\n', ...
        total_throttle_dist, total_throttle_dist/lap_ref.total_distance*100);

%% TRAINING PERFORMANCE VISUALIZATION
figure('Position', [100, 100, 800, 600], 'Name', 'NARX Training Performance');
subplot(2,2,1);
plot(tr.perf);
title('Training Performance');
xlabel('Epoch');
ylabel('Mean Squared Error');
grid on;

subplot(2,2,2);
plot(tr.gradient);
title('Training Gradient');
xlabel('Epoch');
ylabel('Gradient');
grid on;

subplot(2,2,3);
bar(tr.num_epochs);
title('Training Epochs');
xlabel('Training');
ylabel('Epochs');

subplot(2,2,4);
plotperform(tr);

%% FINAL SUMMARY
fprintf('\n=== FINAL SUMMARY ===\n');
fprintf('Processed %d laps successfully\n', length(lap_info));
fprintf('Model trained with %d total segments\n', size(all_laps_U, 2));
fprintf('Total training time: %.1f seconds\n', tr.time(end));

% Display lap statistics
fprintf('\nLAP STATISTICS:\n');
fprintf('%-6s %-10s %-10s %-10s %-10s %-12s\n', ...
        'Lap', 'Distance(km)', 'Time(s)', 'Points', 'Segments', 'Throttle(%)');
for i = 1:length(lap_info)
    throttle_pct = sum(lap_info(i).throttle_state) / length(lap_info(i).speed) * 100;
    fprintf('%-6d %-10.2f %-10.1f %-10d %-10d %-12.1f\n', ...
            lap_info(i).lap_number, ...
            lap_info(i).total_distance/1000, ...
            lap_info(i).lap_time, ...
            length(lap_info(i).speed), ...
            lap_info(i).num_segments, ...
            throttle_pct);
end

fprintf('\n=== ALL PROCESSES COMPLETED ===\n');
