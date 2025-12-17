function CH = genCHtype(K,NT,type)
% Scenario changes

%% Settings

    scenario_set = ["3GPP_38.901_UMi", "3GPP_38.901_UMa", "3GPP_38.901_RMa", "3GPP_38.901_Indoor_Mixed_Office", "3GPP_38.901_Indoor_Open_Office"];
    scenario_type = scenario_set(type); %choose one from 1~5
    
    no_sector = 1/3; % number of sectors in the BS:  1 or 3 or 6, or 1/3 (simulte one of three sectors, users are also set in the one sector)
    % if no_sector=3 or 6, the number of antennas at the BS (NT) will be multiplied
    % Indoor scenarios consider one sector only
    
    fc = 3.5e9; % carrier frequency in [Hz]
    BW = 20e6; % Bandwidth in [Hz] (doesn't matter beacause narrow-band channel considered)
    SCS = 15e3; % Subcarrier spacing in [Hz] (doesn't matter beacause narrow-band channel considered)
    
    no_RB = 106; %Maxium number of RBs, determined by SCS and BW as described in TS 38.101-2 (doesn't matter beacause narrow-band channel considered)
    update_rate = 0; %channel update_rate in [s], also a snap within user's track. It's set as once per OFDM symbol, approximately equals to 1/SCS*107% considering CP duration
    
    no_rx = K; % Number of MTs (i.e., Users / UEs), denoted as K
    simulated_slot = 0; % number of time-slots simulated
    simulated_RB = 1; % number of RBs simulated
    
    is_static_channel = true; % if true, the simulated_slot, user speed, and all time-varying characteristics will be invalid
    
    % intermediate variables: 
    Nsubcarrier = 12; %an RB includes 12 subcarriers (use 1 beacause narrow-band channel)
    Nslot = 14; % a slot includes 14 OFDM symbols
    % Nfft = 2^ceil(log2(BW/SCS)); % FFT points
    simulated_time = simulated_slot * Nslot * update_rate; %simulation duration in [s]
    
    if simulated_RB>no_RB || no_RB*SCS*Nsubcarrier>BW
        disp('RB error')
    end
     
    %% Other Settings
    % set user speed, cell ISD, user indoor-ratio, etc
    
    switch scenario_type
        case "3GPP_38.901_UMi"
            isd = 200; % ISD in [m]
            no_go_dist = 10; % Min. UE-BS 2D distance in [m]
            BS_height = 10; %BS height in[m]
            UE_height = 1.5; %ground UE height in[m]
            ratio = 0.8; %indoor ratio
            speed = 3/3.6; %UE speed in [m/s]
            
        case "3GPP_38.901_UMa"
            isd = 500;
            no_go_dist = 35;
            BS_height = 25;
            UE_height = 1.5;
            ratio = 0.8;
            speed = 3/3.6;
            
        case "3GPP_38.901_RMa" 
            isd = 1732; % or 5000
            no_go_dist = 35;
            BS_height = 35; 
            UE_height = 1.5;
            ratio = 0.5; % 50% indoor and 50% in car
            speed = NaN; % undefined in specification
            if fc > 7e9
                disp('invalid fc for RMa') % Up to 7Ghz
            end
    
        case "3GPP_38.901_Indoor_Mixed_Office"
            isd = 20;
            no_go_dist = [];
            BS_height = 3;
            UE_height = 1;
            ratio = 1;
            speed = 3/3.6;
            no_sector = 1;
    
        case "3GPP_38.901_Indoor_Open_Office" % the difference to Mixed_Office is in LOS prob.
            isd = 20;
            no_go_dist = [];
            BS_height = 3;
            UE_height = 1;
            ratio = 1;
            speed = 3/3.6;
            no_sector = 1;
    end
    
    if is_static_channel
        simulated_time = 0;
        speed = 0;
        simulated_slot = 0;
        update_rate = 0;
    end
    
    %% Antenna Setting
    
    % BS antenna configuration for a sector
    
    % aBS  = qd_arrayant( '3gpp-mmw', 1, NT, fc, 1, [], d_lam, 1, 1, 2.5, 2.5); %type, M, N, fc, Polarization indicator, downtilt in [deg] (only for indicator>=4), element spacing in [lambda], Mg, Ng, panel V-spacing in [lambda], panel H-spacing in [lambda]
    aBS  = qd_arrayant( '3gpp-mmw', 2, 2, fc, 3, [], 0.5, 1, 2, 2.5, 2.5); %type, M, N, fc, Polarization indicator, downtilt in [deg] (only for indicator>=4), element spacing in [lambda], Mg, Ng, panel V-spacing in [lambda], panel H-spacing in [lambda]

    if scenario_type=="3GPP_38.901_UMi" || scenario_type=="3GPP_38.901_UMa" || scenario_type=="3GPP_38.901_RMa"
        aBS.element_position(1,:) = 0.5;  % Distance from pole in [m]
    end
    
    % MT antenna configuration, to be copied and randomly rotated for each user
    
    aMT = qd_arrayant('omni'); %(1) single antenna
    
    NR = aMT.no_elements;
    
    %% Generate Layout 
    qd_s = qd_simulation_parameters;% general simulation parameters
    qd_s.center_frequency = fc; 
    qd_s.use_3GPP_baseline = 1; %Use 3GPP model, but large-scale parameters are not updated for terminal mobility, so time-slot and speed cannot be too large
    qd_s.show_progress_bars = 0; % Disable progress bar 
    
    switch no_sector
        case 1/3
            qd_l = qd_layout; % only simulate one sector (at 0 deg) in the BS with 3 sectors
            qd_l.tx_array = aBS; 
        case 1
            qd_l = qd_layout.generate( 'hexagonal', 1, isd, aBS); % pre-defined type: hexagonal (BS has 1 sector), number of sites (1 for single cell), isd, arrayant for the sector
        case 3
            qd_l = qd_layout.generate( 'regular', 1, isd, aBS); % pre-defined type: regular (BS has 3 sectors at 30, 150, and -90 deg), no_sites, isd, arrayant for each sector
        case 6
            qd_l = qd_layout.generate( 'regular6', 1, isd, aBS); % pre-defined type: regular6 (BS has 6 sectors at 0, 60, 120 ... deg), no_sites, isd, arrayant for each sector
    end
    
    if is_static_channel
        qd_s.sample_density=1.2; %single-mobility
    else
        qd_s.set_speed(speed*3.6 , update_rate); %auto set sample_density
        qd_l.update_rate = update_rate;
    end
    
    qd_l.tx_position(3,:) = BS_height;
    qd_l.simpar = qd_s;
    
    NT = qd_l.tx_array.no_elements;
    
    %% Generate Users
    
    qd_l.no_rx = no_rx;
    
    switch scenario_type
        case {"3GPP_38.901_UMi", "3GPP_38.901_UMa", "3GPP_38.901_RMa"} 
            switch no_sector
                case 1/3
                    % Only generate users in a sector at -60 deg~60 deg
                    % Users move in random directions
                    qd_l.randomize_rx_positions( isd/2, UE_height, UE_height, simulated_time*speed, [], no_go_dist); %generate users: max_dist, min_height, max_height, track_length, rx ind, min_dist, orientation (default:random)
                    
                    x = qd_l.rx_position(1,:); % make users into the sector
                    y = qd_l.rx_position(2,:);
                    rho = sqrt(x.^2 + y.^2);          
                    theta = atan2(y, x);         
                    theta_compressed = mod(theta, 2 * pi / 3) - pi / 3;
                    qd_l.rx_position(1,:) = rho .* cos(theta_compressed);
                    qd_l.rx_position(2,:) = rho .* sin(theta_compressed); % make users into the sector
    
                    ue_floor = randi(5,1,no_rx) + 3;                   % random in 4~8, Number of floors in a building defined in TR 36.973 % the building for RMa is undefined
                    for n = 1 : no_rx
                        ue_floor( n ) =  randi(  ue_floor( n ) );                 % Floor level of the UE
    
                        qd_l.rx_array(n) = copy(aMT); % the antenna for a user
                        qd_l.rx_array(n).rotate_pattern(unifrnd(-180, 180, 3, 1),'xyz'); %rotate the antenna in [deg]
                        
                    end
                    qd_l.rx_position(3,:) = 3*(ue_floor-1) + UE_height;           % Height in meters
    
                    indoor_rx = qd_l.set_scenario(char(scenario_type),[],[],ratio); % set scenario, assign users with LOS/NLOS, O2I or not, based on specific probability
                    qd_l.rx_position(3,~indoor_rx) = UE_height;            % Set outdoor-users' height
    
    
                case {1, 3, 6}
                    % Generate users randomly in the cell
                    % Users move in random directions
    
                    qd_l.randomize_rx_positions( isd/2, UE_height, UE_height, simulated_time*speed, [], no_go_dist); %generate users: max_dist, min_height, max_height, track_length, rx ind, min_dist, orientation
    
                    ue_floor = randi(5,1,no_rx) + 3;                   % random in 4~8, Number of floors in a building in TR 36.873
                    for n = 1 : no_rx
                        ue_floor( n ) =  randi(  ue_floor( n ) );                 % Floor level of the UE
    
                        qd_l.rx_array(n) = copy(aMT); % the antenna for a user
                        qd_l.rx_array(n).rotate_pattern(unifrnd(-180, 180, 3, 1),'xyz'); %rotate the antenna in [deg]
    
                    end
                    qd_l.rx_position(3,:) = 3*(ue_floor-1) + UE_height;           % Height in meters
    
                    indoor_rx = qd_l.set_scenario(char(scenario_type),[],[],ratio); % set scenario
                    qd_l.rx_position(3,~indoor_rx) = UE_height;            % Set outdoor-users' height
            end
        
        case {"3GPP_38.901_Indoor_Mixed_Office", "3GPP_38.901_Indoor_Open_Office"}
            % Generate users randomly in the cell/office, all in the same floor
            % Users move in random directions
                    qd_l.randomize_rx_positions( isd/2, UE_height, UE_height, simulated_time*speed, [], no_go_dist); %generate users: max_dist, min_height, max_height, track_length, rx ind, min_dist, orientation
                    for n = 1 : no_rx
                        qd_l.rx_array(n) = copy(aMT); % the antenna for a user
                        qd_l.rx_array(n).rotate_pattern(unifrnd(-180, 180, 3, 1),'xyz'); %rotate the antenna in [deg]
                    end
                    indoor_rx = qd_l.set_scenario(char(scenario_type),[],[],ratio); % set scenario
    end
    
    %If one want all users to be LOS/NLOS or in-/out-door (NOT following probability), they can enable 
    % qd_l.set_scenario('3GPP_38.901_UMi_LOS'); %should also set ratio = 0 at the begining
    % qd_l.set_scenario('3GPP_38.901_UMi_NLOS'); %should also set ratio = 0 at the begining
    % qd_l.set_scenario('3GPP_38.901_UMi_LOS_O2I'); %should also set ratio = 1 at the begining
    % qd_l.set_scenario('3GPP_38.901_UMi_NLOS_O2I'); %should also set ratio = 1 at the begining
    % 
    % qd_l.set_scenario('3GPP_38.901_UMa_LOS');  %should also set ratio = 0 at the begining
    % qd_l.set_scenario('3GPP_38.901_UMa_NLOS');  %should also set ratio = 0 at the begining
    % qd_l.set_scenario('3GPP_38.901_UMa_LOS_O2I');  %should also set ratio = 1 at the begining
    % qd_l.set_scenario('3GPP_38.901_UMa_NLOS_O2I');   %should also set ratio = 1 at the begining
    % 
    % qd_l.set_scenario('3GPP_38.901_RMa_LOS');  %should also set ratio = 0 at the begining
    % qd_l.set_scenario('3GPP_38.901_RMa_NLOS');  %should also set ratio = 0 at the begining
    % qd_l.set_scenario('3GPP_38.901_RMa_LOS_O2I');  %should also set ratio = 1 at the begining
    % qd_l.set_scenario('3GPP_38.901_RMa_NLOS_O2I');   %should also set ratio = 1 at the begining
    % 
    % qd_l.set_scenario('3GPP_38.901_Indoor_LOS'); % ratio is always 1
    % qd_l.set_scenario('3GPP_38.901_Indoor_NLOS'); % ratio is always 1

    %% Generate channels
    qd_b = qd_l.init_builder;   % Creates 'qd_builder' objects based on layout specification. LOS users, NLOS users, O2I users, etc, will be separated
    gen_parameters( qd_b ); % Generate LSF and SSF parameters
    qd_c = get_channels( qd_b ); % Generate channels
    
    %% Generate frequency-domain channels 
    % averaged over each RB and each time slot
    
    all_subcarrier = no_RB * Nsubcarrier;
    
    simulated_subcarrier = 1; %
    Nsubcarrier = 1; %

    simulated_subcarrier_place = (-all_subcarrier/2:-all_subcarrier/2+simulated_subcarrier-1)/all_subcarrier;
    
    if is_static_channel
        CH = zeros(NR, NT, simulated_RB, no_rx);% static_channel
    
        for k = 1:no_rx
            CHfreq = qd_c(k).fr(all_subcarrier*SCS, simulated_subcarrier_place); %get channels on the subcarriers needed. To ensure a correct SCS, the inputted BW only includes data subcarriers
            CHRB = mean(reshape(CHfreq, NR, NT, Nsubcarrier, simulated_RB), 3);%averaged over each RB
            CH(:,:,:,k) = CHRB;
        end
    
    else
        CH = zeros(NR, NT, simulated_RB, simulated_slot, no_rx);% varying_channel
    
        simulated_symbols = simulated_slot*Nslot; %number of snaps may be a little larger than this
        
        for k = 1:no_rx
            CHfreq = qd_c(k).fr(all_subcarrier*SCS, simulated_subcarrier_place, 1:simulated_symbols); %get channels on the subcarriers and symbols needed
            CHfreq = mean(reshape(CHfreq, NR, NT, simulated_subcarrier, Nslot,simulated_slot), 4);%averaged over each slot
            CHRB = mean(reshape(CHfreq, NR, NT, Nsubcarrier, simulated_RB, simulated_slot), 3);%averaged over each RB
            CH(:,:,:,:,k) = CHRB;
        end
    end
    
    % size(CH)
    % sum(abs(CH).^2,"all")


    %% noise
    noiseFigure = 9;
    noiseVariancedBm = -174 + 10*log10(BW) + noiseFigure;

    CH = squeeze(CH) * 10^(-noiseVariancedBm/20);
end