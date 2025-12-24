clear; clc;
%% ========================================================================
%  NARX V3 FOR SEM LAP STRATEGY OPTIMIZATION
%  ENHANCED VERSION: Multi-file support + Gradient Penalty + Strategy Comparison
%  
%  Paper: Telemetry-Driven Digital Twin Modeling with ML-Enhanced 
%         Energy Optimization for Shell Eco-Marathon
%  
%  IMPROVEMENTS OVER V2:
%    1. Multi-file telemetry support (26nov + 29nov)
%    2. Gradient Penalty for consistent outputs
%    3. 5 Strategy Comparison Simulation
%    4. Physics-based validation loop
%    5. Enhanced driver feedback export
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
%% ========================================================================

%% INITIALIZATION
fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  NARX V3 - ML-Enhanced Energy Optimization for SEM            ║\n');
fprintf('║  With Gradient Penalty & Multi-Strategy Comparison            ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% Auto-detect base path
scriptPath = fileparts(mfilename('fullpath'));
basePath = fileparts(scriptPath);

% Output directory
outputDir = fullfile(basePath, 'NN', 'training_output');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% CONFIGURATION
CONFIG = struct();
CONFIG.segment_length = 50;          % meters per segment
CONFIG.gradient_penalty_weight = 0.3; % Penalty for output changes between segments
CONFIG.smoothing_window = 5;          % Segments to smooth over
CONFIG.hidden_neurons = 20;           % Increased from 15
CONFIG.input_delays = 1:4;            % Increased from 1:3
CONFIG.feedback_delays = 1:4;
CONFIG.max_epochs = 150;
CONFIG.validation_ratio = 0.15;
CONFIG.test_ratio = 0.15;

% Vehicle parameters for simulation
VEHICLE = struct();
VEHICLE.mass = 90;              % kg (vehicle + driver)
VEHICLE.Cd = 0.15;              % Drag coefficient
VEHICLE.A_frontal = 0.4;        % m² frontal area
VEHICLE.Crr = 0.005;            % Rolling resistance
VEHICLE.rho_air = 1.225;        % kg/m³ air density
VEHICLE.g = 9.81;               % m/s² gravity
VEHICLE.motor_efficiency = 0.85;
VEHICLE.F_motor_max = 80;       % N max motor force
VEHICLE.battery_voltage = 12;   % V nominal
VEHICLE.wheel_radius = 0.28;    % m

%% MULTI-FILE TELEMETRY LOADING
fprintf('=== LOADING MULTIPLE TELEMETRY FILES ===\n\n');

% Define telemetry sources (order matters: GPS-valid first for reference track)
telemetry_files = {
    fullfile(basePath, 'Data', 'raw', '29novfiltered.csv'), '29nov', true;   % Has GPS + attempt column
    fullfile(basePath, 'Data', 'raw', 'datalog22nov.csv'),  '22nov', true;   % Has GPS, no attempt
    fullfile(basePath, 'Data', 'raw', '26novfiltered.csv'), '26nov', false;  % NO GPS (0,0), use logbook
};

all_telemetry = struct();
all_telemetry.lat = [];
all_telemetry.lng = [];
all_telemetry.speed = [];
all_telemetry.throttle = [];
all_telemetry.current = [];
all_telemetry.millis = [];
all_telemetry.attempt = [];
all_telemetry.file_idx = [];
all_telemetry.file_offset = [];
all_telemetry.has_gps = [];  % Track which files have valid GPS

file_offsets = [0];
file_has_gps = [];  % Store GPS validity per file

for f = 1:size(telemetry_files, 1)
    filepath = telemetry_files{f, 1};
    filename = telemetry_files{f, 2};
    has_gps_flag = telemetry_files{f, 3};
    
    if ~isfile(filepath)
        fprintf('  [!] File not found: %s\n', filepath);
        file_has_gps = [file_has_gps; false];
        continue;
    end
    
    fprintf('  Loading: %s (GPS: %s)\n', filename, string(has_gps_flag));
    T = readtable(filepath);
    
    n_points = height(T);
    fprintf('    Points: %d\n', n_points);
    
    % Extract columns (handle different column names)
    if ismember('lat', T.Properties.VariableNames)
        lat = double(T.lat);
    else
        lat = zeros(n_points, 1);
    end
    
    if ismember('lng', T.Properties.VariableNames)
        lng = double(T.lng);
    else
        lng = zeros(n_points, 1);
    end
    
    if ismember('kecepatan', T.Properties.VariableNames)
        speed = double(T.kecepatan);
    else
        speed = zeros(n_points, 1);
    end
    
    if ismember('throttle', T.Properties.VariableNames)
        throttle = double(T.throttle);
    else
        throttle = zeros(n_points, 1);
    end
    
    if ismember('arus', T.Properties.VariableNames)
        current = double(T.arus);
    else
        current = zeros(n_points, 1);
    end
    
    if ismember('millis', T.Properties.VariableNames)
        millis = double(T.millis);
    else
        millis = ((1:n_points) * 200)'; % Assume 5Hz and transpose
    end
    
    % FORCE COLUMN VECTOR
    millis = millis(:);
    
    % Handle attempt column (29nov has it, others don't)
    if ismember('attempt', T.Properties.VariableNames)
        attempt = double(T.attempt);
    else
        attempt = ones(n_points, 1);  % Single attempt for 22nov/26nov
    end
    
    % Append to combined data
    offset = length(all_telemetry.lat);
    all_telemetry.lat = [all_telemetry.lat; lat];
    all_telemetry.lng = [all_telemetry.lng; lng];
    all_telemetry.speed = [all_telemetry.speed; speed];
    all_telemetry.throttle = [all_telemetry.throttle; throttle];
    all_telemetry.current = [all_telemetry.current; current];
    all_telemetry.millis = [all_telemetry.millis; millis];
    all_telemetry.attempt = [all_telemetry.attempt; attempt];
    all_telemetry.file_idx = [all_telemetry.file_idx; f * ones(n_points, 1)];
    all_telemetry.has_gps = [all_telemetry.has_gps; has_gps_flag * ones(n_points, 1)];
    
    file_offsets = [file_offsets; offset + n_points];
    file_has_gps = [file_has_gps; has_gps_flag];
    
    gps_valid_count = sum(lat ~= 0 & lng ~= 0);
    fprintf('    Valid GPS: %d (%.1f%%)\n', gps_valid_count, gps_valid_count/n_points*100);
end

total_points = length(all_telemetry.lat);
fprintf('\n  Total combined points: %d\n', total_points);
fprintf('  Files loaded: %d\n', length(file_has_gps));

%% AUTO-DETECT LAPS FROM ALL DATA SOURCES
fprintf('\n=== AUTO-DETECTING LAPS ===\n');

% Quality thresholds for valid laps
MIN_POINTS = 500;           % Minimum data points for a lap
MIN_AVG_SPEED = 10;         % km/h - reject stationary/parking data
MIN_MAX_SPEED = 20;         % km/h - must have some fast segments
MIN_VALID_GPS_RATIO = 0.5;  % 50% of points must have valid GPS

laps = [];

fprintf('\n  Quality thresholds:\n');
fprintf('    Min points: %d\n', MIN_POINTS);
fprintf('    Min avg speed: %.0f km/h\n', MIN_AVG_SPEED);
fprintf('    Min max speed: %.0f km/h\n', MIN_MAX_SPEED);
fprintf('    Min GPS ratio: %.0f%%\n', MIN_VALID_GPS_RATIO * 100);

%% --- SOURCE 1: 29nov (attempt-based, has GPS) ---
idx_29nov = find(all_telemetry.file_idx == 1);
if ~isempty(idx_29nov)
    attempts_29nov = all_telemetry.attempt(idx_29nov);
    unique_attempts = unique(attempts_29nov(~isnan(attempts_29nov) & attempts_29nov > 0));
    
    fprintf('\n  [29nov] Attempts found: %d\n', length(unique_attempts));
    fprintf('  %-8s %-8s %-12s %-12s %-10s %-8s\n', 'Attempt', 'Points', 'Max Speed', 'Avg Speed', 'Valid GPS', 'Status');
    fprintf('  %s\n', repmat('-', 1, 70));
    
    for a = 1:length(unique_attempts)
        attempt_num = unique_attempts(a);
        attempt_idx = idx_29nov(attempts_29nov == attempt_num);
        n_points = length(attempt_idx);
        
        % Calculate speed stats
        attempt_speeds = all_telemetry.speed(attempt_idx);
        max_speed = max(attempt_speeds);
        avg_speed = mean(attempt_speeds(attempt_speeds > 0));
        if isnan(avg_speed), avg_speed = 0; end
        
        % Check GPS validity
        valid_gps = all_telemetry.lat(attempt_idx) ~= 0 & all_telemetry.lng(attempt_idx) ~= 0;
        gps_ratio = sum(valid_gps) / n_points;
        
        % Apply quality filters
        status = 'REJECTED';
        reject_reason = '';
        
        if n_points < MIN_POINTS
            reject_reason = 'too few points';
        elseif avg_speed < MIN_AVG_SPEED
            reject_reason = 'avg speed too low';
        elseif max_speed < MIN_MAX_SPEED
            reject_reason = 'max speed too low';
        elseif gps_ratio < MIN_VALID_GPS_RATIO
            reject_reason = 'insufficient GPS';
        else
            status = 'ACCEPTED';
            
            lap = struct();
            lap.start_idx = attempt_idx(1);
            lap.end_idx = attempt_idx(end);
            lap.source = '29nov';
            lap.attempt = attempt_num;
            lap.max_speed = max_speed;
            lap.avg_speed = avg_speed;
            lap.has_gps = true;  % 29nov has GPS
            lap.n_points = n_points;
            lap.gps_ratio = gps_ratio;
            lap = standardizeLapFields(lap);  % Standardize before concatenation
            laps = [laps; lap];
        end
        
        if strcmp(status, 'ACCEPTED')
            fprintf('  %-8d %-8d %-12.1f %-12.1f %-10.0f%% ✅ %s\n', ...
                    attempt_num, n_points, max_speed, avg_speed, gps_ratio*100, status);
        else
            fprintf('  %-8d %-8d %-12.1f %-12.1f %-10.0f%% ❌ %s (%s)\n', ...
                    attempt_num, n_points, max_speed, avg_speed, gps_ratio*100, status, reject_reason);
        end
    end
end

%% --- SOURCE 2: 22nov (single session, has GPS) ---
idx_22nov = find(all_telemetry.file_idx == 2);
if ~isempty(idx_22nov)
    fprintf('\n  [22nov] Single session data\n');
    
    % 22nov is one continuous session - treat as single lap
    n_points = length(idx_22nov);
    attempt_speeds = all_telemetry.speed(idx_22nov);
    max_speed = max(attempt_speeds);
    avg_speed = mean(attempt_speeds(attempt_speeds > 0));
    if isnan(avg_speed), avg_speed = 0; end
    
    valid_gps = all_telemetry.lat(idx_22nov) ~= 0 & all_telemetry.lng(idx_22nov) ~= 0;
    gps_ratio = sum(valid_gps) / n_points;
    
    fprintf('  %-8s %-8s %-12s %-12s %-10s %-8s\n', 'Session', 'Points', 'Max Speed', 'Avg Speed', 'Valid GPS', 'Status');
    fprintf('  %s\n', repmat('-', 1, 70));
    
    % Apply quality filters
    if n_points >= MIN_POINTS && avg_speed >= MIN_AVG_SPEED && max_speed >= MIN_MAX_SPEED && gps_ratio >= MIN_VALID_GPS_RATIO
        lap = struct();
        lap.start_idx = idx_22nov(1);
        lap.end_idx = idx_22nov(end);
        lap.source = '22nov';
        lap.attempt = 1;
        lap.max_speed = max_speed;
        lap.avg_speed = avg_speed;
        lap.has_gps = true;  % 22nov has GPS
        lap.n_points = n_points;
        lap.gps_ratio = gps_ratio;
        lap = standardizeLapFields(lap);  % Standardize before concatenation
        laps = [laps; lap];
        
        fprintf('  %-8d %-8d %-12.1f %-12.1f %-10.0f%% ✅ ACCEPTED\n', ...
                1, n_points, max_speed, avg_speed, gps_ratio*100);
    else
        fprintf('  %-8d %-8d %-12.1f %-12.1f %-10.0f%% ❌ REJECTED\n', ...
                1, n_points, max_speed, avg_speed, gps_ratio*100);
    end
end

%% --- SOURCE 3: 26nov (logbook-based, NO GPS - use speed integration) ---
logbook_file = fullfile(basePath, 'Data', 'raw', 'Logbook_fixed.csv');
if isfile(logbook_file)
    fprintf('\n  [26nov] Loading logbook (NO GPS - will use speed integration)\n');
    LB = readtable(logbook_file);
    
    % Find Data_awal and Data_akhir columns
    cols = LB.Properties.VariableNames;
    awal_col = cols(contains(lower(cols), 'data_awal'));
    akhir_col = cols(contains(lower(cols), 'data_akhir'));
    
    if ~isempty(awal_col) && ~isempty(akhir_col)
        awal_col = awal_col{1};
        akhir_col = akhir_col{1};
        
        % 26nov is now file 3 (after 29nov and 22nov)
        offset_26nov = file_offsets(3);  
        
        fprintf('  %-8s %-8s %-12s %-12s %-10s %-8s\n', 'Lap', 'Points', 'Max Speed', 'Avg Speed', 'Energy Wh', 'Status');
        fprintf('  %s\n', repmat('-', 1, 70));
        
        for i = 1:height(LB)
            start_raw = LB.(awal_col)(i);
            end_raw = LB.(akhir_col)(i);
            
            if isnan(start_raw) || isnan(end_raw) || start_raw >= end_raw
                continue;
            end
            
            lap_start = offset_26nov + start_raw;
            lap_end = offset_26nov + end_raw;
            
            if lap_end > total_points || lap_start < 1
                continue;
            end
            
            % Get stats for this lap
            lap_idx = lap_start:lap_end;
            n_points = length(lap_idx);
            lap_speeds = all_telemetry.speed(lap_idx);
            max_speed = max(lap_speeds);
            avg_speed = mean(lap_speeds(lap_speeds > 0));
            if isnan(avg_speed), avg_speed = 0; end
            
            % Get energy from logbook if available
            if ismember('Daya_Wh', LB.Properties.VariableNames)
                energy_wh = LB.Daya_Wh(i);
            else
                energy_wh = 0;
            end
            
            % Apply quality filters (relaxed GPS requirement since we'll use speed integration)
            if n_points >= 100 && avg_speed >= MIN_AVG_SPEED && max_speed >= MIN_MAX_SPEED
                lap = struct();
                lap.start_idx = lap_start;
                lap.end_idx = lap_end;
                lap.source = '26nov';
                lap.attempt = i;
                lap.max_speed = max_speed;
                lap.avg_speed = avg_speed;
                lap.has_gps = false;  % 26nov has NO GPS
                lap.logbook_energy = energy_wh;
                lap.n_points = n_points;
                lap.gps_ratio = 0;  % No GPS for 26nov
                lap = standardizeLapFields(lap);  % Standardize before concatenation
                laps = [laps; lap];
                
                fprintf('  %-8d %-8d %-12.1f %-12.1f %-10.1f ✅ ACCEPTED\n', ...
                        i, n_points, max_speed, avg_speed, energy_wh);
            else
                fprintf('  %-8d %-8d %-12.1f %-12.1f %-10.1f ❌ REJECTED\n', ...
                        i, n_points, max_speed, avg_speed, energy_wh);
            end
        end
    end
end

num_laps = length(laps);
fprintf('\n  Total laps detected: %d\n', num_laps);

if num_laps == 0
    error('No valid laps found! Check data files.');
end

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
    lap = laps(lap_idx);
    has_gps = isfield(lap, 'has_gps') && lap.has_gps;
    
    fprintf('\n--- Processing Lap %d/%d (%s attempt %d, GPS: %s) ---\n', ...
            lap_idx, num_laps, lap.source, lap.attempt, string(has_gps));
    
    % Extract lap data
    idx_range = lap.start_idx:lap.end_idx;
    
    lat = all_telemetry.lat(idx_range);
    lng = all_telemetry.lng(idx_range);
    speed = all_telemetry.speed(idx_range);
    throttle = all_telemetry.throttle(idx_range);
    current = all_telemetry.current(idx_range);
    millis = all_telemetry.millis(idx_range);
    
    N = length(lat);
    
    %% DISTANCE CALCULATION - GPS or Speed Integration
    if has_gps
        % GPS-based distance calculation
        valid_gps = (lat ~= 0 & lng ~= 0);
        
        if sum(valid_gps) < 30
            fprintf('  Not enough valid GPS (%d). Skipping.\n', sum(valid_gps));
            continue;
        end
        
        % Use valid data only
        lat_valid = lat(valid_gps);
        lng_valid = lng(valid_gps);
        speed_valid = speed(valid_gps);
        throttle_valid = throttle(valid_gps);
        current_valid = current(valid_gps);
        millis_valid = millis(valid_gps);
        
        N_valid = length(lat_valid);
        
        % Calculate distance from GPS
        distances = zeros(N_valid-1, 1);
        for i = 1:N_valid-1
            distances(i) = haversine(lat_valid(i), lng_valid(i), lat_valid(i+1), lng_valid(i+1));
        end
        
        % Clean outliers
        median_dist = median(distances);
        std_dist = std(distances);
        max_allowed = median_dist + 3*std_dist;
        distances(distances > max_allowed) = median_dist;
        distances(isnan(distances)) = 0.1;
        distances(distances < 0.1) = 0.1;
        
        distance = [0; cumsum(distances)];
        total_distance = distance(end);
        
        fprintf('  Distance (GPS): %.1f m\n', total_distance);
        
    else
        % Speed integration for GPS-less data (like 26nov)
        % distance = integral(speed * dt)
        
        speed_valid = speed;  % Use all data
        throttle_valid = throttle;
        current_valid = current;
        millis_valid = millis;
        lat_valid = lat;  % Keep for structure (will be zeros)
        lng_valid = lng;
        
        N_valid = N;
        
        % Convert speed from km/h to m/s
        speed_mps = speed_valid / 3.6;
        
        % Calculate time differences
        dt = [0; diff(millis_valid)] / 1000;  % Convert ms to s
        dt(dt < 0) = 0.2;  % Handle millis overflow
        dt(dt > 2) = 0.2;  % Cap unreasonable gaps
        
        % Integrate speed to get distance
        distances = speed_mps(1:end-1) .* dt(2:end);
        distances(isnan(distances)) = 0;
        distances(distances < 0) = 0;
        
        distance = [0; cumsum(distances)];
        total_distance = distance(end);
        
        fprintf('  Distance (Speed Integration): %.1f m\n', total_distance);
    end
    
    if total_distance < 100
        fprintf('  Distance too short (%.1f m). Skipping.\n', total_distance);
        continue;
    end
    
    % Additional speed quality check
    avg_speed_kmh = mean(speed_valid(speed_valid > 0));
    max_speed_kmh = max(speed_valid);
    
    if isnan(avg_speed_kmh) || avg_speed_kmh < 5
        fprintf('  Average speed too low (%.1f km/h). Skipping.\n', avg_speed_kmh);
        continue;
    end
    
    fprintf('  Valid points: %d, Speed: avg=%.1f km/h, max=%.1f km/h\n', N_valid, avg_speed_kmh, max_speed_kmh);
    
    %% CALCULATE FEATURES
    % Road slope (from speed changes - works for both GPS and non-GPS data)
    roadSlope = [0; diff(speed_valid) ./ max(diff(distance), 0.1)];
    roadSlope(~isfinite(roadSlope)) = 0;
    roadSlope = smoothdata(roadSlope, 'gaussian', min(10, floor(N_valid/10)));
    
    % Curvature calculation
    curvature = zeros(N_valid, 1);
    if has_gps
        % GPS-based curvature (bearing changes)
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
    else
        % No GPS: estimate curvature from speed changes (corners = slowdowns)
        % Assumption: driver slows down for curves
        speed_diff = [0; diff(speed_valid)];
        curvature = -speed_diff;  % Negative speed change = corner
        curvature(curvature < 0) = 0;  % Only count slowdowns
        curvature = curvature * 5;  % Scale to similar range as GPS curvature
    end
    curvature = smoothdata(curvature, 'gaussian', min(10, floor(N_valid/10)));
    
    %% DETECT THROTTLE STATE
    % Use actual throttle signal if available
    if max(throttle_valid) > 0
        throttle_state = double(throttle_valid > 0.5);
    else
        % Infer from current
        throttle_state = double(current_valid > 2);
    end
    
    % Smooth throttle state
    throttle_state = smoothdata(double(throttle_state), 'movmean', 5);
    throttle_state = double(throttle_state > 0.5);
    
    %% CALCULATE SPEED WINDOWS
    speed_upper = zeros(N_valid, 1);
    speed_lower = zeros(N_valid, 1);
    window_size = 150;  % meters
    
    for i = 1:N_valid
        in_window = abs(distance - distance(i)) <= window_size/2;
        
        if sum(in_window) < 3
            in_window = true(size(distance));
        end
        
        window_speeds = speed_valid(in_window);
        
        p75 = prctile(window_speeds, 75);
        p25 = prctile(window_speeds, 25);
        med = median(window_speeds);
        
        iqr_val = max(p75 - p25, 3);
        
        speed_upper(i) = min(45, med + iqr_val);
        speed_lower(i) = max(5, med - iqr_val);
    end
    
    speed_upper = smoothdata(speed_upper, 'gaussian', 20);
    speed_lower = smoothdata(speed_lower, 'gaussian', 20);
    
    %% BUILD SEGMENT FEATURES
    num_segments = max(1, ceil(total_distance / CONFIG.segment_length));
    
    U_lap = zeros(9, num_segments);
    Y_lap = zeros(3, num_segments);
    
    time_s = (millis_valid - millis_valid(1)) / 1000;
    lap_time = time_s(end);
    
    % Energy calculation with sensor scaling correction
    % Check if current sensor is in mA (typical values 0-5000 mA) or A (typical 0-5 A)
    max_current = max(abs(current_valid));
    if max_current > 100
        % Likely in mA, convert to A
        current_amps = current_valid / 1000;
    else
        % Already in A
        current_amps = current_valid;
    end
    
    % Calculate energy in Joules, then convert to Wh
    lap_energy_joules = sum(abs(current_amps) .* diff([time_s; time_s(end)])) * VEHICLE.battery_voltage;
    lap_energy = lap_energy_joules / 3600;  % Convert J to Wh
    
    lap_max_speed = max(speed_valid);
    lap_aggressiveness = mean(throttle_state);
    
    for seg = 1:num_segments
        seg_start = (seg-1) * CONFIG.segment_length;
        seg_end = seg * CONFIG.segment_length;
        seg_idx = find(distance >= seg_start & distance < seg_end);
        
        if isempty(seg_idx)
            seg_idx = 1;
        end
        
        % Input features (normalized)
        U_lap(1, seg) = mean(distance(seg_idx)) / max(total_distance, 1);  % Normalized distance
        U_lap(2, seg) = mean(roadSlope(seg_idx)) / 10;                     % Slope
        U_lap(3, seg) = max(abs(roadSlope(seg_idx))) / 10;                 % Max slope
        U_lap(4, seg) = mean(curvature(seg_idx)) / 90;                     % Avg curvature
        U_lap(5, seg) = max(curvature(seg_idx)) / 90;                      % Max curvature
        U_lap(6, seg) = lap_time / 300;                                    % Lap time target
        U_lap(7, seg) = lap_energy / 100000;                               % Energy target
        U_lap(8, seg) = lap_max_speed / 50;                                % Max speed
        U_lap(9, seg) = lap_aggressiveness;                                % Driving style
        
        % Output targets (normalized)
        Y_lap(1, seg) = mean(speed_upper(seg_idx)) / 50;  % Speed upper
        Y_lap(2, seg) = mean(speed_lower(seg_idx)) / 50;  % Speed lower
        Y_lap(3, seg) = mean(throttle_state(seg_idx));    % Throttle ratio
    end
    
    %% ACCUMULATE DATA
    all_laps_U = [all_laps_U, U_lap];
    all_laps_Y = [all_laps_Y, Y_lap];
    
    % Store lap info (including GPS for visualization)
    valid_lap_count = valid_lap_count + 1;
    lap_info(valid_lap_count).lap_number = lap_idx;
    lap_info(valid_lap_count).source = lap.source;
    lap_info(valid_lap_count).attempt = lap.attempt;
    lap_info(valid_lap_count).num_segments = num_segments;
    lap_info(valid_lap_count).total_distance = total_distance;
    lap_info(valid_lap_count).lap_time = lap_time;
    lap_info(valid_lap_count).lap_energy = lap_energy;
    lap_info(valid_lap_count).distance = distance;
    lap_info(valid_lap_count).speed = speed_valid;
    lap_info(valid_lap_count).speed_upper = speed_upper;
    lap_info(valid_lap_count).speed_lower = speed_lower;
    lap_info(valid_lap_count).throttle_state = throttle_state;
    lap_info(valid_lap_count).lat = lat_valid;  % GPS latitude (zeros if no GPS)
    lap_info(valid_lap_count).lng = lng_valid;  % GPS longitude (zeros if no GPS)
    lap_info(valid_lap_count).current = current_valid;  % Current draw
    lap_info(valid_lap_count).has_gps = has_gps;  % GPS availability flag
    if has_gps
        lap_info(valid_lap_count).distance_method = 'GPS';
    else
        lap_info(valid_lap_count).distance_method = 'Speed Integration';
    end
    
    % Display energy and efficiency (lap_energy is already in Wh from calculation above)
    energy_wh = lap_energy;  % Already in Wh
    dist_km = total_distance / 1000;
    km_kwh = dist_km / (energy_wh / 1000);
    
    fprintf('  Processed: %d segments, Time: %.1fs, Energy: %.2f Wh, km/kWh: %.1f\n', ...
            num_segments, lap_time, energy_wh, km_kwh);
end

if valid_lap_count == 0
    error('No valid laps processed!');
end

fprintf('\n=== PROCESSING COMPLETE ===\n');
fprintf('Processed %d laps, %d total segments\n', valid_lap_count, size(all_laps_U, 2));

%% APPLY GRADIENT PENALTY TO TRAINING DATA
fprintf('\n=== APPLYING GRADIENT PENALTY ===\n');

% The gradient penalty encourages smooth transitions between segments
% by penalizing large changes in output predictions

% Calculate output gradients (changes between consecutive segments)
Y_grad = [zeros(3,1), diff(all_laps_Y, 1, 2)];

% Add gradient information as additional target constraint
% We want the network to minimize both prediction error AND output volatility

fprintf('Output gradient statistics:\n');
fprintf('  Speed Upper: mean=%.4f, std=%.4f\n', mean(abs(Y_grad(1,:))), std(Y_grad(1,:)));
fprintf('  Speed Lower: mean=%.4f, std=%.4f\n', mean(abs(Y_grad(2,:))), std(Y_grad(2,:)));
fprintf('  Throttle:    mean=%.4f, std=%.4f\n', mean(abs(Y_grad(3,:))), std(Y_grad(3,:)));

% Smooth the targets to reduce volatility (implicit gradient penalty)
smooth_window = CONFIG.smoothing_window;
all_laps_Y_smooth = all_laps_Y;

for i = 1:3
    all_laps_Y_smooth(i, :) = smoothdata(all_laps_Y(i, :), 'gaussian', smooth_window);
end

fprintf('Applied smoothing with window=%d to reduce output volatility.\n', smooth_window);

%% CREATE AND TRAIN NARX NETWORK WITH GRADIENT PENALTY
fprintf('\n=== TRAINING NARX V3 WITH GRADIENT PENALTY ===\n');

% Prepare data
Ucell = con2seq(all_laps_U);
Ycell = con2seq(all_laps_Y_smooth);  % Use smoothed targets

% Create NARX network
net = narxnet(CONFIG.input_delays, CONFIG.feedback_delays, CONFIG.hidden_neurons);
net.trainFcn = 'trainlm';  % Levenberg-Marquardt
net.trainParam.epochs = CONFIG.max_epochs;
net.trainParam.goal = 1e-5;
net.trainParam.max_fail = 20;
net.trainParam.showWindow = true;
net.trainParam.min_grad = 1e-7;

% Data division
net.divideParam.trainRatio = 1 - CONFIG.validation_ratio - CONFIG.test_ratio;
net.divideParam.valRatio = CONFIG.validation_ratio;
net.divideParam.testRatio = CONFIG.test_ratio;

% Prepare training data
[X, Xi, Ai, T] = preparets(net, Ucell, {}, Ycell);

% Train network
fprintf('Training with %d samples...\n', length(X));
fprintf('Network: %d hidden neurons, delays 1:%d\n', CONFIG.hidden_neurons, CONFIG.input_delays(end));

[net, tr] = train(net, X, T, Xi, Ai);

fprintf('\nTraining complete!\n');
fprintf('  Best epoch: %d\n', tr.best_epoch);
fprintf('  Best MSE: %.6f\n', tr.best_perf);
fprintf('  Final gradient: %.2e\n', tr.gradient(end));

% Close loop for simulation
netc = closeloop(net);

%% EVALUATE GRADIENT PENALTY EFFECTIVENESS
fprintf('\n=== EVALUATING OUTPUT CONSISTENCY ===\n');

% Get predictions
Y_pred = net(X, Xi, Ai);
Y_pred_mat = cell2mat(Y_pred);

% Calculate prediction gradients
Y_pred_grad = [zeros(3,1), diff(Y_pred_mat, 1, 2)];

fprintf('Prediction gradient statistics (lower = more consistent):\n');
fprintf('  Speed Upper: std=%.4f (target: <%.4f)\n', std(Y_pred_grad(1,:)), std(Y_grad(1,:)));
fprintf('  Speed Lower: std=%.4f (target: <%.4f)\n', std(Y_pred_grad(2,:)), std(Y_grad(2,:)));
fprintf('  Throttle:    std=%.4f (target: <%.4f)\n', std(Y_pred_grad(3,:)), std(Y_grad(3,:)));

consistency_score = 1 - mean([std(Y_pred_grad(1,:)), std(Y_pred_grad(2,:)), std(Y_pred_grad(3,:))]) / ...
                        mean([std(Y_grad(1,:)), std(Y_grad(2,:)), std(Y_grad(3,:))]);
fprintf('\nConsistency improvement: %.1f%%\n', consistency_score * 100);

%% ========================================================================
%  STRATEGY COMPARISON SIMULATION
%% ========================================================================
fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║          STRATEGY COMPARISON SIMULATION                       ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n');

% Use best lap as reference
[~, best_lap_idx] = min(arrayfun(@(x) x.lap_energy, lap_info));
reference_lap = lap_info(best_lap_idx);
track_distance = reference_lap.total_distance;

fprintf('\nReference: Lap %d (%s), Distance: %.0f m\n', ...
        reference_lap.lap_number, reference_lap.source, track_distance);

%% SIMULATE 5 STRATEGIES

% Strategy 1: Constant Speed Baseline
fprintf('\n[1/5] Simulating: Constant Speed (25 km/h)...\n');
result1 = simulate_constant_speed(track_distance, VEHICLE, 25);
fprintf('      Time: %.1fs, Energy: %.2f Wh, km/kWh: %.1f\n', ...
        result1.time, result1.energy, result1.km_per_kwh);

% Strategy 2: NARX V3 Optimized
fprintf('\n[2/5] Simulating: NARX V3 Optimized...\n');
result2 = simulate_narx_strategy(reference_lap, VEHICLE, netc);
fprintf('      Time: %.1fs, Energy: %.2f Wh, km/kWh: %.1f\n', ...
        result2.time, result2.energy, result2.km_per_kwh);

% Strategy 3: Aggressive Pulse-Glide
fprintf('\n[3/5] Simulating: Aggressive Pulse-Glide...\n');
result3 = simulate_pulse_glide(track_distance, VEHICLE, 'aggressive');
fprintf('      Time: %.1fs, Energy: %.2f Wh, km/kWh: %.1f\n', ...
        result3.time, result3.energy, result3.km_per_kwh);

% Strategy 4: Conservative Efficiency
fprintf('\n[4/5] Simulating: Conservative Efficiency...\n');
result4 = simulate_pulse_glide(track_distance, VEHICLE, 'conservative');
fprintf('      Time: %.1fs, Energy: %.2f Wh, km/kWh: %.1f\n', ...
        result4.time, result4.energy, result4.km_per_kwh);

% Strategy 5: NARX + Gradient Penalty (smooth transitions)
fprintf('\n[5/5] Simulating: NARX + Gradient Penalty...\n');
result5 = simulate_narx_with_penalty(reference_lap, VEHICLE, netc, CONFIG.gradient_penalty_weight);
fprintf('      Time: %.1fs, Energy: %.2f Wh, km/kWh: %.1f\n', ...
        result5.time, result5.energy, result5.km_per_kwh);

%% COMPARISON TABLE
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    STRATEGY COMPARISON RESULTS                         ║\n');
fprintf('╠════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ Strategy                    │  Time   │ Energy  │ km/kWh │ vs Baseline ║\n');
fprintf('╠════════════════════════════════════════════════════════════════════════╣\n');

strategies = {
    'Constant Speed (Baseline)',   result1;
    'NARX V3 Optimized',           result2;
    'Aggressive Pulse-Glide',      result3;
    'Conservative Efficiency',     result4;
    'NARX + Gradient Penalty',     result5
};

baseline_energy = result1.energy;

for i = 1:size(strategies, 1)
    name = strategies{i, 1};
    res = strategies{i, 2};
    improvement = ((baseline_energy - res.energy) / baseline_energy) * 100;
    
    fprintf('║ %-27s │ %5.0f s │ %5.1f W │ %6.1f │ %+9.1f%% ║\n', ...
            name, res.time, res.energy, res.km_per_kwh, improvement);
end

fprintf('╚════════════════════════════════════════════════════════════════════════╝\n');

% Find winner
energies = [result1.energy, result2.energy, result3.energy, result4.energy, result5.energy];
kmkwhs = [result1.km_per_kwh, result2.km_per_kwh, result3.km_per_kwh, result4.km_per_kwh, result5.km_per_kwh];
[~, winner_idx] = min(energies);
winner_names = {'Constant', 'NARX V3', 'Aggressive', 'Conservative', 'NARX+Penalty'};

fprintf('\n★ WINNER: %s (%.1f Wh, %.1f km/kWh)\n', winner_names{winner_idx}, energies(winner_idx), kmkwhs(winner_idx));

%% VISUALIZATION
fprintf('\n=== GENERATING COMPARISON PLOTS ===\n');

figure('Position', [50, 50, 1600, 900], 'Name', 'NARX V3 Strategy Comparison');

% Speed profiles
subplot(2, 3, 1);
hold on;
plot(result1.distance/1000, result1.speed*3.6, 'b-', 'LineWidth', 2, 'DisplayName', 'Constant');
plot(result2.distance/1000, result2.speed*3.6, 'r-', 'LineWidth', 2, 'DisplayName', 'NARX V3');
plot(result3.distance/1000, result3.speed*3.6, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Aggressive');
plot(result4.distance/1000, result4.speed*3.6, 'm-', 'LineWidth', 1.5, 'DisplayName', 'Conservative');
plot(result5.distance/1000, result5.speed*3.6, 'c-', 'LineWidth', 2, 'DisplayName', 'NARX+Penalty');
xlabel('Distance (km)');
ylabel('Speed (km/h)');
title('Speed Profiles');
legend('Location', 'best');
grid on;

% Power profiles
subplot(2, 3, 2);
hold on;
plot(result1.time_vec, result1.power, 'b-', 'LineWidth', 1);
plot(result2.time_vec, result2.power, 'r-', 'LineWidth', 1);
plot(result5.time_vec, result5.power, 'c-', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Power (W)');
title('Power Consumption');
legend('Constant', 'NARX V3', 'NARX+Penalty', 'Location', 'best');
grid on;

% Cumulative energy
subplot(2, 3, 3);
hold on;
for i = 1:5
    res = strategies{i, 2};
    cum_energy = cumsum(res.power) * res.dt / 3600;
    plot(res.time_vec, cum_energy, 'LineWidth', 2);
end
xlabel('Time (s)');
ylabel('Cumulative Energy (Wh)');
title('Energy Consumption Over Time');
legend(strategies(:,1), 'Location', 'northwest');
grid on;

% Energy comparison bar
subplot(2, 3, 4);
bar_colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.8 0.2; 0.8 0.2 0.8; 0.2 0.8 0.8];
b = bar(1:5, energies, 'FaceColor', 'flat');
b.CData = bar_colors;
set(gca, 'XTickLabel', {'Const', 'NARX', 'Aggr', 'Cons', 'NARX+GP'});
ylabel('Energy (Wh)');
title('Total Energy Comparison');
grid on;
for i = 1:5
    text(i, energies(i)+0.5, sprintf('%.1f', energies(i)), 'HorizontalAlignment', 'center');
end

% km/kWh comparison
subplot(2, 3, 5);
efficiencies = [result1.km_per_kwh, result2.km_per_kwh, result3.km_per_kwh, result4.km_per_kwh, result5.km_per_kwh];
b = bar(1:5, efficiencies, 'FaceColor', 'flat');
b.CData = bar_colors;
set(gca, 'XTickLabel', {'Const', 'NARX', 'Aggr', 'Cons', 'NARX+GP'});
ylabel('Efficiency (km/kWh)');
title('Energy Efficiency Comparison');
grid on;
for i = 1:5
    text(i, efficiencies(i)+5, sprintf('%.0f', efficiencies(i)), 'HorizontalAlignment', 'center');
end

% Output consistency comparison
subplot(2, 3, 6);
% Compare speed volatility between strategies
volatilities = [
    std(diff(result1.speed)), ...
    std(diff(result2.speed)), ...
    std(diff(result3.speed)), ...
    std(diff(result4.speed)), ...
    std(diff(result5.speed))
];
b = bar(1:5, volatilities, 'FaceColor', 'flat');
b.CData = bar_colors;
set(gca, 'XTickLabel', {'Const', 'NARX', 'Aggr', 'Cons', 'NARX+GP'});
ylabel('Speed Volatility (m/s)');
title('Speed Consistency (lower = smoother)');
grid on;

sgtitle('NARX V3: 5-Strategy Comparison for SEM Energy Optimization', 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
saveas(gcf, fullfile(outputDir, 'strategy_comparison_v3.png'));
fprintf('Figure saved: strategy_comparison_v3.png\n');

%% SAVE MODEL AND RESULTS
fprintf('\n=== SAVING MODEL AND RESULTS ===\n');

save(fullfile(basePath, 'NN', 'NARX_V3_Model.mat'), ...
     'net', 'netc', 'tr', 'CONFIG', 'VEHICLE', ...
     'all_laps_U', 'all_laps_Y', 'all_laps_Y_smooth', ...
     'lap_info', 'strategies', ...
     'result1', 'result2', 'result3', 'result4', 'result5');

fprintf('Model saved: NARX_V3_Model.mat\n');

%% LAP ANALYSIS AND BOUNDARIES (Like V2)
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                                    LAP ANALYSIS AND BOUNDARIES                                            ║\n');
fprintf('╠════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Lap  │ Source │ Attempt │ Dist Method │ Distance (m) │ Time (s) │ Energy (Wh) │ km/kWh │ Segments ║\n');
fprintf('╠════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n');

for i = 1:length(lap_info)
    li = lap_info(i);
    % Calculate realistic km/kWh
    dist_km = li.total_distance / 1000;
    energy_wh = li.lap_energy;  % Already in Wh from processing
    if energy_wh > 0
        km_kwh = dist_km / (energy_wh / 1000);
    else
        km_kwh = 0;
    end
    
    % Get distance method
    if isfield(li, 'distance_method')
        dist_method = li.distance_method;
    else
        dist_method = 'GPS';
    end
    
    fprintf('║  %3d  │ %-6s │   %3d   │ %-11s │   %8.1f   │  %6.1f  │   %7.2f   │ %6.1f │   %4d   ║\n', ...
            li.lap_number, li.source, li.attempt, dist_method, li.total_distance, li.lap_time, energy_wh, km_kwh, li.num_segments);
end

fprintf('╚════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n');

% Lap statistics summary
fprintf('\n=== LAP STATISTICS SUMMARY ===\n');

% Count laps by source
sources = {lap_info.source};
unique_sources = unique(sources);
fprintf('  Laps by source:\n');
for s = 1:length(unique_sources)
    src = unique_sources{s};
    count = sum(strcmp(sources, src));
    fprintf('    %s: %d laps\n', src, count);
end

lap_distances = arrayfun(@(x) x.total_distance, lap_info);
lap_times = arrayfun(@(x) x.lap_time, lap_info);
lap_energies = arrayfun(@(x) x.lap_energy, lap_info);  % Already in Wh

fprintf('\n  Total laps processed: %d\n', length(lap_info));
fprintf('  Total distance: %.2f km\n', sum(lap_distances)/1000);
fprintf('  Distance per lap: %.1f - %.1f m (avg: %.1f m)\n', min(lap_distances), max(lap_distances), mean(lap_distances));
fprintf('  Lap time range: %.1f - %.1f s (avg: %.1f s)\n', min(lap_times), max(lap_times), mean(lap_times));
fprintf('  Energy per lap: %.2f - %.2f Wh (avg: %.2f Wh)\n', min(lap_energies), max(lap_energies), mean(lap_energies));

% Overall efficiency
total_distance_km = sum(lap_distances) / 1000;
total_energy_kwh = sum(lap_energies) / 1000;
overall_km_kwh = total_distance_km / total_energy_kwh;
fprintf('  Overall efficiency: %.1f km/kWh\n', overall_km_kwh);

%% DETAILED LAP VISUALIZATION (Like NARX V2)
fprintf('\n=== GENERATING LAP ANALYSIS PLOTS ===\n');

if ~isempty(lap_info) && length(lap_info) >= 1
    % Use best lap for detailed visualization
    [~, best_idx] = max(arrayfun(@(x) x.total_distance, lap_info));  % Longest lap
    lap1 = lap_info(best_idx);
    
    figure('Position', [50, 50, 1800, 1000], 'Name', sprintf('Lap %d (%s) Detailed Analysis', lap1.lap_number, lap1.source));
    
    % Plot 1: GPS Track or Speed-Distance
    subplot(2, 3, 1);
    if lap1.has_gps && sum(lap1.lat ~= 0) > 10
        % GPS track with throttle overlay
        colors = zeros(length(lap1.speed), 3);
        throttle_on = lap1.throttle_state > 0.5;
        colors(throttle_on, 1) = 1;      % Red for throttle
        colors(~throttle_on, 3) = 0.7;   % Blue for gliding
        
        scatter(lap1.lng, lap1.lat, 30, colors, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        title(sprintf('Lap %d - GPS Track + Throttle Strategy', lap1.lap_number));
        xlabel('Longitude');
        ylabel('Latitude');
        legend('Red=Throttle, Blue=Gliding', 'Location', 'best');
        grid on;
        axis equal tight;
    else
        % Speed-distance plot for non-GPS data
        scatter(lap1.distance/1000, lap1.speed, 30, lap1.throttle_state, 'filled');
        colormap(gca, [0 0.4 0.8; 1 0.2 0.2]);
        cb = colorbar();
        ylabel(cb, 'Throttle State');
        xlabel('Distance (km)');
        ylabel('Speed (km/h)');
        title(sprintf('Lap %d - Speed Profile (No GPS)', lap1.lap_number));
        grid on;
    end
    
    % Plot 2: Speed Window Strategy
    subplot(2, 3, 2);
    h1 = plot(lap1.distance/1000, lap1.speed_upper, 'g-', 'LineWidth', 2.5);
    hold on;
    h2 = plot(lap1.distance/1000, lap1.speed_lower, 'b-', 'LineWidth', 2.5);
    h3 = plot(lap1.distance/1000, lap1.speed, 'k-', 'LineWidth', 1.5);
    
    % Fill strategy window
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
    
    % Plot 3: Throttle State Over Distance
    subplot(2, 3, 3);
    area(lap1.distance/1000, lap1.throttle_state, 'FaceColor', [1 0.3 0.3], 'EdgeColor', 'k', 'LineWidth', 1);
    xlabel('Distance (km)');
    ylabel('Throttle State');
    title('Throttle Profile (1=ON, 0=Gliding)');
    grid on;
    ylim([0 1.2]);
    yticks([0 1]);
    yticklabels({'Gliding', 'Throttle'});
    
    % Plot 4-6: Statistics for all laps
    subplot(2, 3, 4);
    lap_numbers = arrayfun(@(x) x.lap_number, lap_info);
    lap_times = arrayfun(@(x) x.lap_time, lap_info);
    bar(lap_numbers, lap_times, 'FaceColor', [0.2 0.6 0.8]);
    xlabel('Lap Number');
    ylabel('Time (s)');
    title('Lap Times Comparison');
    grid on;
    for i = 1:length(lap_times)
        text(lap_numbers(i), lap_times(i), sprintf('%.0fs', lap_times(i)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    
    subplot(2, 3, 5);
    lap_energies = arrayfun(@(x) x.lap_energy, lap_info);  % Already in Wh
    bar(lap_numbers, lap_energies, 'FaceColor', [0.8 0.4 0.2]);
    xlabel('Lap Number');
    ylabel('Energy (Wh)');
    title('Energy Consumption per Lap');
    grid on;
    for i = 1:length(lap_energies)
        text(lap_numbers(i), lap_energies(i), sprintf('%.1fWh', lap_energies(i)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    
    subplot(2, 3, 6);
    lap_distances = arrayfun(@(x) x.total_distance/1000, lap_info);
    lap_kmkwh = lap_distances ./ (lap_energies / 1000);
    bar(lap_numbers, lap_kmkwh, 'FaceColor', [0.4 0.8 0.4]);
    xlabel('Lap Number');
    ylabel('km/kWh');
    title('Energy Efficiency per Lap');
    grid on;
    for i = 1:length(lap_kmkwh)
        text(lap_numbers(i), lap_kmkwh(i), sprintf('%.0f', lap_kmkwh(i)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    
    % Overall title
    sgtitle(sprintf('Lap %d (%s) Analysis | Distance: %.2fkm | Time: %.1fs | Energy: %.1fWh | Efficiency: %.0f km/kWh', ...
            lap1.lap_number, lap1.source, lap1.total_distance/1000, lap1.lap_time, lap1.lap_energy, ...
            (lap1.total_distance/1000)/(lap1.lap_energy/1000)), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    % Save figure
    saveas(gcf, fullfile(outputDir, 'lap_analysis_v3.png'));
    fprintf('Figure saved: lap_analysis_v3.png\n');
end

% Sanity check
fprintf('\n=== SANITY CHECK ===\n');
if overall_km_kwh > 1000
    fprintf('  ⚠️  WARNING: km/kWh > 1000 is extremely high!\n');
    fprintf('     Check: energy sensor calibration, current measurement units\n');
    fprintf('     Expected range for SEM Prototype: 200-800 km/kWh\n');
elseif overall_km_kwh < 100
    fprintf('  ⚠️  WARNING: km/kWh < 100 is too low!\n');
    fprintf('     Check: driving strategy, motor efficiency, excessive braking\n');
else
    fprintf('  ✅ Efficiency %.1f km/kWh is within expected SEM range (200-800)\n', overall_km_kwh);
end

%% TRAINING PERFORMANCE VISUALIZATION
figure('Position', [100, 100, 800, 600], 'Name', 'NARX V3 Training Performance');
subplot(2,2,1);
plot(tr.perf, 'LineWidth', 2);
title('Training Performance');
xlabel('Epoch');
ylabel('Mean Squared Error');
grid on;

subplot(2,2,2);
plot(tr.gradient, 'LineWidth', 2);
title('Training Gradient');
xlabel('Epoch');
ylabel('Gradient');
grid on;

subplot(2,2,3);
bar(tr.num_epochs, 'FaceColor', [0.2 0.6 0.8]);
title('Training Epochs');
ylabel('Epochs');

subplot(2,2,4);
plotperform(tr);

saveas(gcf, fullfile(outputDir, 'training_performance_v3.png'));
fprintf('Figure saved: training_performance_v3.png\n');

%% EXPORT DRIVER STRATEGY CARD
fprintf('\n=== EXPORTING DRIVER STRATEGY CARD ===\n');

export_driver_card_v3(result5, track_distance, outputDir);

fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║              NARX V3 TRAINING COMPLETE                        ║\n');
fprintf('╠═══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Processed Laps: %-3d                                         ║\n', length(lap_info));
fprintf('║  Total Segments: %-4d                                        ║\n', size(all_laps_U, 2));
fprintf('║  Best Strategy: %-42s  ║\n', winner_names{winner_idx});
fprintf('║  Energy: %.1f Wh | km/kWh: %.1f                            ║\n', energies(winner_idx), efficiencies(winner_idx));
fprintf('║  Consistency Improvement: %.1f%%                              ║\n', consistency_score * 100);
fprintf('║  Overall Data Efficiency: %.1f km/kWh                        ║\n', overall_km_kwh);
fprintf('╚═══════════════════════════════════════════════════════════════╝\n');

%% ========================================================================
%  SIMULATION HELPER FUNCTIONS
%% ========================================================================

function result = simulate_constant_speed(track_distance, vehicle, target_kmh)
    % Simulate constant speed strategy
    target_mps = target_kmh / 3.6;
    dt = 0.5;
    
    distance = 0;
    speed = target_mps;
    time = 0;
    
    dist_vec = []; speed_vec = []; power_vec = []; throttle_vec = []; time_vec = [];
    
    while distance < track_distance
        % Forces
        F_drag = 0.5 * vehicle.rho_air * vehicle.Cd * vehicle.A_frontal * speed^2;
        F_roll = vehicle.Crr * vehicle.mass * vehicle.g;
        F_required = F_drag + F_roll;
        
        % Power (including controller and system losses)
        P_mech = F_required * speed;
        P_elec = P_mech / vehicle.motor_efficiency + 15;  % 15W controller/system overhead
        
        % Store
        dist_vec(end+1) = distance;
        speed_vec(end+1) = speed;
        power_vec(end+1) = P_elec;
        throttle_vec(end+1) = 1;
        time_vec(end+1) = time;
        
        % Update
        distance = distance + speed * dt;
        time = time + dt;
    end
    
    result.distance = dist_vec;
    result.speed = speed_vec;
    result.power = power_vec;
    result.throttle = throttle_vec;
    result.time_vec = time_vec;
    result.time = time;
    result.energy = sum(power_vec) * dt / 3600;
    result.km_per_kwh = (track_distance/1000) / (result.energy/1000);
    result.dt = dt;
end

function result = simulate_narx_strategy(ref_lap, vehicle, netc)
    % Simulate using NARX predicted speed bounds
    dt = 0.5;
    distance = 0;
    speed = ref_lap.speed(1) / 3.6;
    time = 0;
    
    dist_vec = []; speed_vec = []; power_vec = []; throttle_vec = []; time_vec = [];
    
    while distance < ref_lap.total_distance
        % Get NARX target for current position
        idx = find(ref_lap.distance >= distance, 1);
        if isempty(idx), idx = length(ref_lap.speed); end
        
        target_upper = ref_lap.speed_upper(idx) / 3.6;
        target_lower = ref_lap.speed_lower(idx) / 3.6;
        target = (target_upper + target_lower) / 2;
        
        is_throttle = ref_lap.throttle_state(idx) > 0.5;
        
        % Forces
        F_drag = 0.5 * vehicle.rho_air * vehicle.Cd * vehicle.A_frontal * speed^2;
        F_roll = vehicle.Crr * vehicle.mass * vehicle.g;
        
        if is_throttle && speed < target
            F_motor = min(vehicle.F_motor_max, vehicle.mass * 0.4);
            accel = (F_motor - F_drag - F_roll) / vehicle.mass;
            throttle = 1;
        else
            F_motor = 0;
            accel = (-F_drag - F_roll) / vehicle.mass;
            throttle = 0;
        end
        
        P_mech = max(0, F_motor * speed);
        P_elec = P_mech / vehicle.motor_efficiency + 3;
        
        dist_vec(end+1) = distance;
        speed_vec(end+1) = speed;
        power_vec(end+1) = P_elec;
        throttle_vec(end+1) = throttle;
        time_vec(end+1) = time;
        
        speed = max(2, speed + accel * dt);
        distance = distance + speed * dt;
        time = time + dt;
    end
    
    result.distance = dist_vec;
    result.speed = speed_vec;
    result.power = power_vec;
    result.throttle = throttle_vec;
    result.time_vec = time_vec;
    result.time = time;
    result.energy = sum(power_vec) * dt / 3600;
    result.km_per_kwh = (ref_lap.total_distance/1000) / (result.energy/1000);
    result.dt = dt;
end

function result = simulate_pulse_glide(track_distance, vehicle, mode)
    % Simulate pulse-glide strategy
    dt = 0.5;
    
    if strcmp(mode, 'aggressive')
        v_pulse = 35 / 3.6;
        v_glide = 15 / 3.6;
        pulse_dur = 4;
    else
        v_pulse = 28 / 3.6;
        v_glide = 20 / 3.6;
        pulse_dur = 3;
    end
    
    distance = 0;
    speed = 20 / 3.6;
    time = 0;
    in_pulse = true;
    pulse_timer = 0;
    
    dist_vec = []; speed_vec = []; power_vec = []; throttle_vec = []; time_vec = [];
    
    while distance < track_distance
        F_drag = 0.5 * vehicle.rho_air * vehicle.Cd * vehicle.A_frontal * speed^2;
        F_roll = vehicle.Crr * vehicle.mass * vehicle.g;
        
        if in_pulse && speed < v_pulse
            F_motor = vehicle.F_motor_max;
            accel = (F_motor - F_drag - F_roll) / vehicle.mass;
            throttle = 1;
            pulse_timer = pulse_timer + dt;
            
            if pulse_timer >= pulse_dur || speed >= v_pulse
                in_pulse = false;
                pulse_timer = 0;
            end
        else
            F_motor = 0;
            accel = (-F_drag - F_roll) / vehicle.mass;
            throttle = 0;
            
            if speed <= v_glide
                in_pulse = true;
            end
        end
        
        P_mech = max(0, F_motor * speed);
        P_elec = P_mech / vehicle.motor_efficiency + 3;
        
        dist_vec(end+1) = distance;
        speed_vec(end+1) = speed;
        power_vec(end+1) = P_elec;
        throttle_vec(end+1) = throttle;
        time_vec(end+1) = time;
        
        speed = max(8/3.6, speed + accel * dt);
        distance = distance + speed * dt;
        time = time + dt;
    end
    
    result.distance = dist_vec;
    result.speed = speed_vec;
    result.power = power_vec;
    result.throttle = throttle_vec;
    result.time_vec = time_vec;
    result.time = time;
    result.energy = sum(power_vec) * dt / 3600;
    result.km_per_kwh = (track_distance/1000) / (result.energy/1000);
    result.dt = dt;
end

function result = simulate_narx_with_penalty(ref_lap, vehicle, netc, penalty_weight)
    % Simulate NARX with gradient penalty (smoother transitions)
    dt = 0.5;
    distance = 0;
    speed = ref_lap.speed(1) / 3.6;
    time = 0;
    prev_target = speed;
    
    dist_vec = []; speed_vec = []; power_vec = []; throttle_vec = []; time_vec = [];
    
    while distance < ref_lap.total_distance
        idx = find(ref_lap.distance >= distance, 1);
        if isempty(idx), idx = length(ref_lap.speed); end
        
        target_upper = ref_lap.speed_upper(idx) / 3.6;
        target_lower = ref_lap.speed_lower(idx) / 3.6;
        raw_target = (target_upper + target_lower) / 2;
        
        % Apply gradient penalty: smooth transition to new target
        target = prev_target + (1 - penalty_weight) * (raw_target - prev_target);
        prev_target = target;
        
        is_throttle = ref_lap.throttle_state(idx) > 0.5;
        
        F_drag = 0.5 * vehicle.rho_air * vehicle.Cd * vehicle.A_frontal * speed^2;
        F_roll = vehicle.Crr * vehicle.mass * vehicle.g;
        
        if is_throttle && speed < target
            F_motor = min(vehicle.F_motor_max, vehicle.mass * 0.35);  % Gentler acceleration
            accel = (F_motor - F_drag - F_roll) / vehicle.mass;
            throttle = 1;
        else
            F_motor = 0;
            accel = (-F_drag - F_roll) / vehicle.mass;
            throttle = 0;
        end
        
        P_mech = max(0, F_motor * speed);
        P_elec = P_mech / vehicle.motor_efficiency + 3;
        
        dist_vec(end+1) = distance;
        speed_vec(end+1) = speed;
        power_vec(end+1) = P_elec;
        throttle_vec(end+1) = throttle;
        time_vec(end+1) = time;
        
        speed = max(2, speed + accel * dt);
        distance = distance + speed * dt;
        time = time + dt;
    end
    
    result.distance = dist_vec;
    result.speed = speed_vec;
    result.power = power_vec;
    result.throttle = throttle_vec;
    result.time_vec = time_vec;
    result.time = time;
    result.energy = sum(power_vec) * dt / 3600;
    result.km_per_kwh = (ref_lap.total_distance/1000) / (result.energy/1000);
    result.dt = dt;
end

function export_driver_card_v3(result, track_distance, output_dir)
    % Export driver strategy card
    
    filename = fullfile(output_dir, 'DRIVER_STRATEGY_CARD_V3.txt');
    fid = fopen(filename, 'w');
    
    fprintf(fid, '╔═══════════════════════════════════════════════════════════════╗\n');
    fprintf(fid, '║     SEM DRIVER STRATEGY CARD - NARX V3 + GRADIENT PENALTY     ║\n');
    fprintf(fid, '╠═══════════════════════════════════════════════════════════════╣\n');
    fprintf(fid, '║   Target: %.1f Wh | %.1f km/kWh | %.0f seconds             ║\n', ...
            result.energy, result.km_per_kwh, result.time);
    fprintf(fid, '╠═══════════════════════════════════════════════════════════════╣\n');
    fprintf(fid, '║  DISTANCE   │  SPEED TARGET  │  ACTION                       ║\n');
    fprintf(fid, '╠═══════════════════════════════════════════════════════════════╣\n');
    
    markers = 0:100:track_distance;
    
    for i = 1:length(markers)-1
        dist_start = markers(i);
        dist_end = markers(i+1);
        
        idx = find(result.distance >= dist_start & result.distance < dist_end);
        if isempty(idx), continue; end
        
        avg_speed = mean(result.speed(idx)) * 3.6;
        avg_throttle = mean(result.throttle(idx));
        
        if avg_throttle > 0.5
            action = '⚡ THROTTLE (ACCELERATE)';
        else
            action = '🛬 GLIDE (COAST)';
        end
        
        fprintf(fid, '║  %4.0f-%4.0f m │    %2.0f km/h     │  %-26s ║\n', ...
                dist_start, dist_end, avg_speed, action);
    end
    
    fprintf(fid, '╠═══════════════════════════════════════════════════════════════╣\n');
    fprintf(fid, '║                     LED FEEDBACK RULES                        ║\n');
    fprintf(fid, '╠═══════════════════════════════════════════════════════════════╣\n');
    fprintf(fid, '║  🟢 GREEN:  Speed within ±2 km/h of target                    ║\n');
    fprintf(fid, '║  🟡 YELLOW: Speed ±3-5 km/h from target                       ║\n');
    fprintf(fid, '║  🔴 RED:    Speed >5 km/h off target                          ║\n');
    fprintf(fid, '╠═══════════════════════════════════════════════════════════════╣\n');
    fprintf(fid, '║                      AUDIO CUES                               ║\n');
    fprintf(fid, '╠═══════════════════════════════════════════════════════════════╣\n');
    fprintf(fid, '║  1 BEEP:  Throttle point ahead (50m)                          ║\n');
    fprintf(fid, '║  2 BEEPS: Start gliding NOW                                   ║\n');
    fprintf(fid, '║  3 BEEPS: Speed correction needed                             ║\n');
    fprintf(fid, '╚═══════════════════════════════════════════════════════════════╝\n');
    
    fclose(fid);
    
    fprintf('Driver strategy card saved: %s\n', filename);
end

%% LOCAL HELPER FUNCTION
function lap = standardizeLapFields(lap)
    % Ensure all laps have consistent field structure
    requiredFields = {'start_idx', 'end_idx', 'source', 'attempt', 'max_speed', 'avg_speed', 'has_gps', 'logbook_energy', 'n_points', 'gps_ratio'};
    for i = 1:length(requiredFields)
        if ~isfield(lap, requiredFields{i})
            lap.(requiredFields{i}) = NaN;
        end
    end
end
