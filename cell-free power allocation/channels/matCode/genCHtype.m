function CH = genCHtype(NT, K, L, ty)
    
    %% Settings
    
    scenario_set = ["3GPP_38.901_UMi", "3GPP_38.901_UMa", "3GPP_38.901_RMa", "3GPP_38.901_Indoor_Mixed_Office", "3GPP_38.901_Indoor_Open_Office"];
    scenario_type = scenario_set(ty); %choose one from 1~5
    
    if ty==1 || ty==2
        D1 = 500;
        D2 = 500;
    elseif ty == 4 || ty==5
        D1 = 120;
        D2 = 50;
    else
        return
    end
    
    width_area = D1; % Width of the area in[m]
    length_area = D2; % Length of the are in[m]
    
    no_AP = L; % Number of APs randomly located in the area, denoted as L
    no_rx = K; % Number of MTs (i.e., Users / UEs)  randomly located in the area, denoted as K
    
    fc = 3.5e9; % carrier frequency in [Hz]
    BW = 20e6; % Bandwidth in [Hz]
    SCS = 15e3; % Subcarrier spacing in [Hz]
    
    no_RB = 106; %Maxium number of RBs, determined by SCS and BW as described in TS 38.101-1
    update_rate = 71.36e-6; %channel update_rate in [s], also a snap within user's track. It's set as once per OFDM symbol, approximately equals to 1/SCS*107% considering CP duration
    
    simulated_slot = 1; % number of time-slots simulated
    simulated_RB = 1; % number of RBs simulated
    
    is_static_channel = true; % if true, the simulated_slot, user speed, and all time-varying characteristics will be invalid
    
    % intermediate variables: 
    Nsubcarrier = 12; %an RB includes 12 subcarriers
    Nslot = 14; % a slot includes 14 OFDM symbols
    Nfft = 2^ceil(log2(BW/SCS)); % FFT points
    simulated_time = simulated_slot * Nslot * update_rate; %simulation duration in [s]
    
    if simulated_RB>no_RB || no_RB*SCS*Nsubcarrier>BW
        disp('RB error')
    end
     
    %% Other Settings
    % set heights, user speed, user indoor-ratio, etc
    
    switch scenario_type
        case "3GPP_38.901_UMi"
            % isd = 200; % ISD in [m]
            % no_go_dist = 10; % Min. UE-BS 2D distance in [m]
            BS_height = 10; %BS height in[m]
            UE_height = 1.5; %ground UE height in[m]
            ratio = 0; %indoor ratio
            speed = 3/3.6; %UE speed in [m/s]
    
        case "3GPP_38.901_UMa"
            % isd = 500;
            % no_go_dist = 35;
            BS_height = 25;
            UE_height = 1.5;
            ratio = 0;
            speed = 3/3.6;
    
        case "3GPP_38.901_RMa" 
            % isd = 1732; % or 5000
            % no_go_dist = 35;
            BS_height = 35; 
            UE_height = 1.5;
            ratio = 0.5; % 50% indoor and 50% in car
            speed = NaN; % undefined in specification
            if fc > 7e9
                disp('invalid fc for RMa') % Up to 7Ghz
            end
    
        case "3GPP_38.901_Indoor_Mixed_Office"
            % isd = 20;
            % no_go_dist = [];
            BS_height = 3;
            UE_height = 1;
            ratio = 1;
            speed = 3/3.6;
            % no_sector = 1;
    
        case "3GPP_38.901_Indoor_Open_Office" % the difference to Mixed_Office is in LOS prob.
            % isd = 20;
            % no_go_dist = [];
            BS_height = 3;
            UE_height = 1;
            ratio = 1;
            speed = 3/3.6;
            % no_sector = 1;
    end
    
    if is_static_channel
        simulated_time = 0;
        speed = 0;
        simulated_slot = 0;
        update_rate = 0;
    end
    
    %% Antenna Setting
    
    % %AP antenna configuration, to be copied and randomly rotated for each AP
    aBS  = qd_arrayant( '3gpp-mmw', 1, NT, fc, 1, [], 0.5, 1, 1, 2.5, 2.5);
    % aBS  = qd_arrayant( '3gpp-mmw', 2, 2, fc, 3, [], 0.5, 1, 2, 2.5, 2.5); %type, M, N, fc, Polarization indicator, downtilt in [deg] (only for indicator>=4), element spacing in [lambda], Mg, Ng, panel V-spacing in [lambda], panel H-spacing in [lambda]
    
    
    NT = aBS.no_elements;
    % %MT antenna configuration, to be copied and randomly rotated for each user
    
    aMT = qd_arrayant('omni'); %(1) single antenna
    
    % aMT = qd_arrayant('omni'); %(2) two antennas of different polarizations
    % aMT.copy_element(1,2);
    % aMT.Fa(:,:,2) = 0;
    % aMT.Fb(:,:,2) = 1;
    
    % aMT  = qd_arrayant( '3gpp-mmw', 1, 2, fc, 3, [], 0.5, 1, 1, [], []); %(3) multiple antennas
    
    NR = aMT.no_elements;
    
    %% Determine the locations of APs  and UEs
    posi_AP = width_area * rand(no_AP,1) + length_area * 1i*rand(no_AP,1);
    posi_UE = width_area * rand(no_rx,1) + length_area * 1i*rand(no_rx,1);
    
    wrapWidth = repmat([-width_area 0 width_area],[3 1]);
    wrapLength = repmat([-length_area 0 length_area],[3 1])';
    wrapLocations = wrapWidth(:)' + 1i*wrapLength(:)';
    posi_AP_warp = repmat(posi_AP,[1 length(wrapLocations)]) + repmat(wrapLocations,[no_AP 1]);
    %3|6|9
    %2|5|8
    %1|4|7
    
    isAPsimulated = zeros(size(posi_AP_warp)); % Determines whether a wrapped AP participates in simulation
    UE_whichReplica = zeros(no_AP, no_rx); % for a user, which are the wrapped APs he connected
    
    for k = 1:no_rx % find the nearest AP among the 9 wrapped replicas
        [~, whichReplica] = min(abs(posi_AP_warp - repmat(posi_UE(k), size(posi_AP_warp))),[],2);
        isAPsimulated(sub2ind([no_AP,9], (1:no_AP)', whichReplica)) = 1;
        UE_whichReplica(:, k) = whichReplica;
    end
    
    %% Generate Layout 
    qd_s = qd_simulation_parameters;% general simulation parameters
    qd_s.center_frequency = fc; 
    qd_s.use_3GPP_baseline = 1; %Use 3GPP model, but large-scale parameters are not updated for terminal mobility, so time-slot and speed cannot be too large
    qd_s.show_progress_bars = 0; % Disable progress bar 
    
    qd_l = qd_layout;
    qd_l.no_tx = nnz(isAPsimulated); % The number of wrapped APs that need to be simulated
    posi_AP_warp_simulated = posi_AP_warp(isAPsimulated == 1);  % and their locations (in column order)
    qd_l.tx_position(1,:) = real(posi_AP_warp_simulated);
    qd_l.tx_position(2,:) = imag(posi_AP_warp_simulated);
    
    for l = 1:qd_l.no_tx
        qd_l.tx_array(l) = copy(aBS);  %APs' antennas
        qd_l.tx_array(l).rotate_pattern(unifrnd(-180, 180),'z'); %rotate the antenna in [deg]
    end
    
    if is_static_channel
        qd_s.sample_density=1.2; %single-mobility
    else
        qd_s.set_speed(speed*3.6 , update_rate); %auto set sample_density
        qd_l.update_rate = update_rate;
    end
    
    qd_l.tx_position(3,:) = BS_height;
    qd_l.simpar = qd_s;
    
    %% Generate Users
    qd_l.no_rx = no_rx;
    
    switch scenario_type
        case {"3GPP_38.901_UMi", "3GPP_38.901_UMa", "3GPP_38.901_RMa"} 
                % Generate users randomly in the area and on different floors
                % Users move in random directions
                qd_l.randomize_rx_positions( 1, UE_height, UE_height, simulated_time*speed, [], []); %generate users: max_dist, min_height, max_height, track_length, rx ind, min_dist, orientation
    
                qd_l.rx_position(1,:) = real(posi_UE);
                qd_l.rx_position(2,:) = imag(posi_UE);
    
                ue_floor = randi(5,1,no_rx) + 3;                   % random in 4~8, Number of floors in a building in TR 36.873
                for n = 1 : no_rx
                    ue_floor( n ) =  randi(  ue_floor( n ) );                 % Floor level of the UE
    
                    qd_l.rx_array(n) = copy(aMT); % the antenna for a user
                    qd_l.rx_array(n).rotate_pattern(unifrnd(-180, 180, 3, 1),'xyz'); %rotate the antenna in [deg]
    
                end
                qd_l.rx_position(3,:) = 3*(ue_floor-1) + UE_height;           % Height in meters
    
    
                indoor_rx = qd_l.set_scenario(char(scenario_type),[],[],ratio); % set scenario
                qd_l.rx_position(3,~indoor_rx) = UE_height;            % Set outdoor-users' height
    
        case {"3GPP_38.901_Indoor_Mixed_Office", "3GPP_38.901_Indoor_Open_Office"}
            % Generate users randomly in the area (office), all in the same floor
            % Users move in random directions
                qd_l.randomize_rx_positions( 1, UE_height, UE_height, simulated_time*speed, [], []); %generate users: max_dist, min_height, max_height, track_length, rx ind, min_dist, orientation
                qd_l.rx_position(1,:) = real(posi_UE);
                qd_l.rx_position(2,:) = imag(posi_UE);
                for n = 1 : no_rx
                    qd_l.rx_array(n) = copy(aMT); % the antenna for a user
                    qd_l.rx_array(n).rotate_pattern(unifrnd(-180, 180, 3, 1),'xyz'); %rotate the antenna in [deg]
                end
                indoor_rx = qd_l.set_scenario(char(scenario_type),[],[],ratio); % set scenario
    end
    
    %% Generate channels
    qd_b = qd_l.init_builder;   % Creates 'qd_builder' objects based on layout specification. LOS users, NLOS users, O2I users, etc, will be separated
    gen_parameters( qd_b ); % Generate LSF and SSF parameters
    qd_c = get_channels( qd_b ); % Generate channels
    
    %% Find the index (in qc_d) of each Tx-Rx pair, by the 'name' in qd_c()
    names = {qd_c.name};
    
    tx_idx_cell = regexp(names,'Tx0*(\d+)_','tokens');
    rx_idx_cell = regexp(names,'Rx0*(\d+)','tokens');
    
    tx_idx = cellfun(@(c) str2double(c{1}{1}), tx_idx_cell);
    rx_idx = cellfun(@(c) str2double(c{1}{1}), rx_idx_cell);
    
    isAPsimulated_col = isAPsimulated(:);
    
    %% Generate frequency-domain channels 
    % averaged over each RB and each time slot
    
    all_subcarrier = no_RB * Nsubcarrier;
    % simulated_subcarrier = simulated_RB*Nsubcarrier;
    simulated_subcarrier = 1;
    Nsubcarrier = 1;
    simulated_subcarrier_place = (-all_subcarrier/2:-all_subcarrier/2+simulated_subcarrier-1)/all_subcarrier;
    
    if is_static_channel
        CH = zeros(NR, NT, simulated_RB, no_rx, no_AP);% static_channel
    
        for l = 1:no_AP
            for k = 1:no_rx
                idx_liner = (UE_whichReplica(l,k)-1)*no_AP+l; 
                which_tx = sum(isAPsimulated_col(1:idx_liner));
                which_rx = k;
                txrx_in_qdc = find(tx_idx == which_tx & rx_idx == which_rx, 1, 'first');
    
                CHfreq = qd_c(txrx_in_qdc).fr(all_subcarrier*SCS, simulated_subcarrier_place); %get channels on the subcarriers needed. To ensure a correct SCS, the inputted BW only includes data subcarriers
                CHRB = mean(reshape(CHfreq, NR, NT, Nsubcarrier, simulated_RB), 3);%averaged over each RB
                CH(:,:,:,k,l) = CHRB;
    
            end
        end
    
    
    
    else % non-static channel
        CH = zeros(NR, NT, simulated_RB, simulated_slot, no_rx, no_AP);% nonstatic_channel
        simulated_symbols = simulated_slot*Nslot; %number of snaps may be a little larger than this
    
        for l = 1:no_AP
            for k = 1:no_rx
                idx_liner = (UE_whichReplica(l,k)-1)*no_AP+l; 
                which_tx = sum(isAPsimulated_col(1:idx_liner));
                which_rx = k;
                txrx_in_qdc = find(tx_idx == which_tx & rx_idx == which_rx, 1, 'first'); 
    
                CHfreq = qd_c(txrx_in_qdc).fr(all_subcarrier*SCS, simulated_subcarrier_place, 1:simulated_symbols); %get channels on the subcarriers and symbols needed
                CHfreq = mean(reshape(CHfreq, NR, NT, simulated_subcarrier, Nslot,simulated_slot), 4);%averaged over each slot
                CHRB = mean(reshape(CHfreq, NR, NT, Nsubcarrier, simulated_RB, simulated_slot), 3);%averaged over each RB
                CH(:,:,:,:,k,l) = CHRB;
            end
        end
    
    end
    
    % size(CH)
    % sum(abs(CH).^2,"all")

    %% noise
    noiseFigure = 9;
    noiseVariancedBW = -174 + 10*log10(BW) + noiseFigure-30;
    CH = squeeze(CH) * 10^(-noiseVariancedBW/20);

