%{

    This function implements a generic cnn algorithm

%}
function [output_struct] = implement_cnn_algorithm(input_struct)

% Initialize the system timer to count the total elapsed time in this func. 
% tic

%% Assign the input parameters to function's local variables
% Convolutional Layers' Settings
conv_input_layer = input_struct.conv_input_channel;
conv_input_size = input_struct.conv_input_size;
conv_layer_content = input_struct.conv_layer_content;
conv_learning_rate = input_struct.conv_learning_rate;
conv_zero_padding_size = input_struct.conv_zero_padding_size;
conv_kernel_size = input_struct.conv_kernel_size;
conv_kernel_init = input_struct.conv_kernel_init;
conv_pooling_type = input_struct.conv_pooling_type;
conv_pooling_size = input_struct.conv_pooling_size;
conv_bias_type = input_struct.conv_bias_type;
conv_activation = input_struct.conv_activation;
                            
% Fully Connected Layers' Settings
fulconn_output_neuron_count = input_struct.fulconn_output_neuron_count; % Fixed
fulconn_hidden_layer_neuron_content = input_struct.fulconn_hidden_layer_neuron_content; 
fulconn_learning_rate = input_struct.fulconn_learning_rate;
fulconn_weight_init = input_struct.fulconn_weight_init;
fulconn_activation = input_struct.fulconn_activation;
fulconn_momentum_coef = input_struct.fulconn_momentum_coef;
fulconn_weight_decay = input_struct.fulconn_weight_decay;

% Network's General Settings
epoch_num = input_struct.epoch_num;
minibatch_size = input_struct.minibatch_size;
train_data = input_struct.train_data;
train_label = input_struct.train_label;
validation_data = input_struct.validation_data;
validation_label = input_struct.validation_label;
test_data = input_struct.test_data;
test_label = input_struct.test_label;
validation_section = input_struct.validation_section;


%% Prepare the network structure

% Initially calculate the conv layer output sizes.
% ((( OK )))
[conv_convolution_out_size_cell, conv_pooling_out_size_cell] ...
                            = calc_conv_out_size( conv_input_size , ...
                                                  conv_layer_content , ...
                                                  conv_zero_padding_size , ...
                                                  conv_kernel_size , ...
                                                  conv_pooling_size );

% 1st) Define conv and fully connected weights and initialize them
% ---------------------------------------------------------------------
    
% Prepare conv layer weight and bias parameters
% ((( OK ))) 
conv_weight_cell = prepare_conv_weight_init( conv_input_layer , ...
                                             conv_layer_content , ...
                                             conv_kernel_size , ...
                                             conv_kernel_init );
% ((( OK ))) 
conv_bias_cell = prepare_conv_bias_init( conv_layer_content , ...
                                         conv_convolution_out_size_cell , ...
                                         conv_bias_type , ... 
                                         conv_kernel_init);

% Prepare fully connected weight initialization (bias is included)
last_conv_layer_size = conv_pooling_out_size_cell{end};
fulconn_input_neuron_count = last_conv_layer_size(1) ...
                            * last_conv_layer_size(2) ...
                            * conv_layer_content(end);
% ((( OK ))) 
fulconn_weight_cell = prepare_fulconn_weight_init( fulconn_input_neuron_count , ...
                                                   fulconn_output_neuron_count, ...
                                                   fulconn_hidden_layer_neuron_content,...
                                                   fulconn_weight_init);


% 2nd) Define conv and fully connected delta weight and init them to 0
% ---------------------------------------------------------------------

% Prepare conv layer delta weight and delta bias parameters
% ((( OK ))) 
conv_delta_weight_cell = prepare_conv_weight_init( conv_input_layer , ...
                                                   conv_layer_content , ...
                                                   conv_kernel_size , ...
                                                   'zeros' );
% ((( OK ))) 
conv_delta_bias_cell = prepare_conv_bias_init( conv_layer_content , ...
                                               conv_convolution_out_size_cell , ...
                                               conv_bias_type , ... 
                                               'zeros');
                                     
% Prepare fully connected layer delta weight initialization (bias is included)
% ((( OK ))) 
fulconn_delta_weight_cell = prepare_fulconn_weight_init( fulconn_input_neuron_count , ...
                                                         fulconn_output_neuron_count, ...
                                                         fulconn_hidden_layer_neuron_content,...
                                                         'zeros');

% 3rd) Define neuron output values
% ---------------------------------------------------------------------

% Prepare conv convolution layer output parameters
conv_convolution_output_cell = prepare_conv_output_cell( conv_layer_content , ...
                                                         conv_convolution_out_size_cell);

% Prepare conv pooling layer output parameters
conv_pooling_output_cell = prepare_conv_output_cell( conv_layer_content , ...
                                                     conv_pooling_out_size_cell);

% Prepare fully connected layer neuron output parameters
fulconn_output_cell = prepare_fulconn_output_cell( fulconn_output_neuron_count, ...
                                                   fulconn_hidden_layer_neuron_content);


%{
    IMPORTANT NOTE: (about fully connected layer neuron output parameters)

    In the fully connected layers, bias coefficient -1 will be padded to 
    the respective hidden layer outputs as an input to the next layer. Thus
    this operation is not included in the definition of neurons.

    The respective weights of the bias coefficient are already defined in
    weight parameters of the fully connected part.

%}

% 4th) Define neuron gradient values
% ---------------------------------------------------------------------

% Prepare conv layer gradient parameters
conv_gradient_cell = prepare_conv_output_cell(conv_layer_content , ...
                                            conv_convolution_out_size_cell);

% Prepare fully connected layer gradient parameters
fulconn_gradient_cell = prepare_fulconn_output_cell( fulconn_output_neuron_count, ...
                                                     fulconn_hidden_layer_neuron_content);
                                                 


% 5th) Define other parameters for memory allocation.
% ---------------------------------------------------------------------
                                                 
conv_pooling_map_cell = prepare_conv_output_cell(conv_layer_content , ...
                                            conv_convolution_out_size_cell);
 
% defining_parameters_time = toc

%% Prepare output parameters

% Error metrics (mean cross entropy error and mean classification error)
train_mcee_per_epoch = zeros(1,epoch_num);  
train_mce_per_epoch = zeros(1,epoch_num);       
validation_mcee_per_epoch = zeros(1,epoch_num);  
validation_mce_per_epoch = zeros(1,epoch_num);       
test_mcee_per_epoch = zeros(1,epoch_num);  
test_mce_per_epoch = zeros(1,epoch_num);      
% Resultant weight per epoch
conv_weight_per_epoch = cell(epoch_num,1); 
conv_bias_per_epoch = cell(epoch_num,1); 
fulconn_weight_per_epoch = cell(epoch_num,1); 
% Total elapsed time
total_elapsed_time = 0;

%% Perform requested task

for i_epoch = 1:epoch_num
%     disp(['Validation Sec #:' num2str(validation_section) ...
%         ' - Epoch #: ' num2str(i_epoch)]);

%% Train the network for the current epoch   

%     begining_epoch_time = toc

    % Initially, determine samples to be processed in each minibatch
    train_data_count = length(train_label);
    minibatch_order = randperm(train_data_count);
    minibatch_count = ceil(train_data_count/minibatch_size);

    % FOR each minibatch, find the average delta_weight and update weights
    for i_mb = 1: minibatch_count
    disp(['Validation Sec #:' num2str(validation_section) ...
        ' - Epoch #: ' num2str(i_epoch) ...
        ' - MiniBatch #: ' num2str(i_mb) ' / ' num2str(minibatch_count) ]);
    
%         begining_minibatch_time = toc

        % Get the minibatch data
        if (i_mb*minibatch_size > train_data_count) 
            % If last batch, size may differ according to the selected minibatch size
            minibatch_ind = minibatch_order( (i_mb-1)*minibatch_size+1 : end );
        else
            minibatch_ind = minibatch_order( (i_mb-1)*minibatch_size+1 : i_mb*minibatch_size );
        end

        % Initially store the previous fulconn_weight_cell
        %     (for momentum implementation)
        prev_fulconn_delta_weight_cell = fulconn_delta_weight_cell;

        % Then, initialize a dummy delta_weight.
        fulconn_delta_weight_cell = prepare_fulconn_weight_init( fulconn_input_neuron_count , ...
                                                                 fulconn_output_neuron_count, ...
                                                                 fulconn_hidden_layer_neuron_content,...
                                                                 'zeros');

        % FOR each sample in the minibatch
        for i_samp = 1:length(minibatch_ind)
            
%             minibatch_processing_initiated_time = toc

            % Prepare input vector
            input_data = train_data(minibatch_ind(i_samp),:);

            % Extract the individual red, green, and blue color channels.
            %   IMPORTANT NOTE: This input configuration is just for this
            %                   project.
            red_ch = rot90(reshape(im2double(input_data(1:1024)),32,32),3);
            green_ch = rot90(reshape(im2double(input_data(1024+1:1024*2)),32,32),3);
            blue_ch = rot90(reshape(im2double(input_data(1024*2+1:1024*3)),32,32),3);
            
            % Store all of the data in a single vector
            conv_input_layer = zeros(3,32,32);
            conv_input_layer(1,:,:) = red_ch;
            conv_input_layer(2,:,:) = green_ch;
            conv_input_layer(3,:,:) = blue_ch;

            % Forward propagate conv layer input through the network.(func)
            [conv_convolution_output_cell , ...
             conv_pooling_output_cell , ...
             conv_pooling_map_cell , ...
             fulconn_input_layer , ...
             fulconn_output_cell] = forward_propagate( conv_input_layer , ...
                                                       conv_weight_cell , ...
                                                       conv_zero_padding_size , ...
                                                       conv_pooling_type , ...
                                                       conv_pooling_size , ...
                                                       conv_activation , ...
                                                       conv_bias_type , ...
                                                       conv_bias_cell , ...
                                                       fulconn_weight_cell , ...
                                                       fulconn_activation , ...
                                                       conv_convolution_output_cell , ...
                                                       conv_pooling_output_cell , ...
                                                       conv_pooling_map_cell , ...
                                                       fulconn_output_cell);

%             forward_propagated_data_time = toc
            
            % calculate the error
            desired_value = zeros(1,fulconn_output_neuron_count);
            desired_value(train_label(minibatch_ind(i_samp))) = 1;
            err = (fulconn_output_cell{end} - desired_value);

            
            % calculate neuron_gradient_cell (func)
            [conv_gradient_cell , ...
             fulconn_gradient_cell ] = backprop_gradient( err , ...
                                                          fulconn_weight_cell , ...
                                                          fulconn_activation , ...
                                                          fulconn_output_cell , ...
                                                          conv_zero_padding_size , ...
                                                          conv_pooling_out_size_cell , ...
                                                          conv_layer_content , ...
                                                          conv_pooling_map_cell , ...
                                                          conv_pooling_size , ...
                                                          conv_activation , ...
                                                          conv_convolution_output_cell , ...
                                                          conv_weight_cell , ...
                                                          conv_gradient_cell , ...
                                                          fulconn_gradient_cell);

%             backpropagated_gradient_time = toc
            
            % compute delta_weight  
            [dummy_conv_delta_weight_cell , ...
             dummy_conv_delta_bias_cell , ...
             dummy_fulconn_delta_weight_cell] = calc_delta_weight( conv_learning_rate , ...
                                                                   conv_zero_padding_size , ...
                                                                   conv_input_layer , ...
                                                                   conv_pooling_output_cell , ...
                                                                   conv_gradient_cell , ...
                                                                   fulconn_learning_rate , ...
                                                                   fulconn_input_layer , ...
                                                                   fulconn_output_cell , ...
                                                                   fulconn_gradient_cell , ...
                                                                   conv_bias_type , ...
                                                                   conv_delta_weight_cell , ...
                                                                   conv_delta_bias_cell , ...
                                                                   fulconn_delta_weight_cell);

            
            % accumulate delta_weight for conv and fulconn parts (accumulate)
            
            % First, conv part
            for i_cell = 1:length(conv_delta_weight_cell)
                % Update conv delta weight
                conv_delta_weight_cell{i_cell} = conv_delta_weight_cell{i_cell} ...
                    + dummy_conv_delta_weight_cell{i_cell};
                
                % Update conv delta bias
                conv_delta_bias_cell{i_cell} = conv_delta_bias_cell{i_cell} ...
                    + dummy_conv_delta_bias_cell{i_cell};
            end
            
            % Second, fulconn part
            for i_cell = 1:length(fulconn_delta_weight_cell)
                % Update fulconn delta weight
                fulconn_delta_weight_cell{i_cell} = fulconn_delta_weight_cell{i_cell} ...
                    + dummy_fulconn_delta_weight_cell{i_cell};
            end
            
%             accumulated_error_time = toc

        end

        %% Average delta_weight values for conv and fulconn parts
        %   !!! Note that the last batch's size may differ. !!!
        
        % For conv layers
        for i_cell = 1:length(conv_delta_weight_cell)
            conv_delta_weight_cell{i_cell} = conv_delta_weight_cell{i_cell}/length(minibatch_ind);
            conv_delta_bias_cell{i_cell} = conv_delta_bias_cell{i_cell}/length(minibatch_ind);
        end
        
        % For fulconn layers
        for i_cell = 1:length(fulconn_delta_weight_cell)
            fulconn_delta_weight_cell{i_cell} = fulconn_delta_weight_cell{i_cell}/length(minibatch_ind);
        end
            

        %% Update weight values 
        
        % For conv layers
        for i_cell = 1:length(conv_weight_cell)
            conv_weight_cell{i_cell} = conv_weight_cell{i_cell} ...
                + conv_delta_weight_cell{i_cell};
            
            conv_bias_cell{i_cell} = conv_bias_cell{i_cell} ...
                + conv_delta_bias_cell{i_cell};
        end
        
        % For fulconn layer 
        %      using averaged delta_weight and prev_delta_weight (momentum)
        for i_cell = 1:length(fulconn_weight_cell)
            % Initially get the selected layer's learning rate value.
            layer_learning_rate = fulconn_learning_rate(i_cell);
            
            % Perform fulconn weight update
            fulconn_weight_cell{i_cell} = fulconn_weight_cell{i_cell} ...
                + fulconn_delta_weight_cell{i_cell} ...
                + fulconn_momentum_coef*prev_fulconn_delta_weight_cell{i_cell}...
                - layer_learning_rate*fulconn_weight_decay*fulconn_weight_cell{i_cell};
        end
        
%         minibatch_finished_time = toc

    xxx = 0;    
    end

%% Using trained network weights, calculate MCCE and MCE
%   MCEE : mean cross entropy error 
%   MCE : mean classification error 



    % Calculate MCCE and MCE of train data for each epoch

    train_mcee_acc = 0;
    train_mce_acc = 0;

    for i_mcce_mce = 1:length(train_label)

        % Prepare input vector
        input_data = train_data(i_mcce_mce,:);

        % Extract the individual red, green, and blue color channels.
        %   IMPORTANT NOTE: This input configuration is just for this
        %                   project.
        red_ch = rot90(reshape(im2double(input_data(1:1024)),32,32),3);
        green_ch = rot90(reshape(im2double(input_data(1024+1:1024*2)),32,32),3);
        blue_ch = rot90(reshape(im2double(input_data(1024*2+1:1024*3)),32,32),3);

        % Store all of the data in a single vector
        conv_input_layer = zeros(3,32,32);
        conv_input_layer(1,:,:) = red_ch;
        conv_input_layer(2,:,:) = green_ch;
        conv_input_layer(3,:,:) = blue_ch;

        % Forward propagate conv layer input through the network.(func)
        [conv_convolution_output_cell , ...
         conv_pooling_output_cell , ...
         conv_pooling_map_cell , ...
         fulconn_input_layer , ...
         fulconn_output_cell] = forward_propagate( conv_input_layer , ...
                                                   conv_weight_cell , ...
                                                   conv_zero_padding_size , ...
                                                   conv_pooling_type , ...
                                                   conv_pooling_size , ...
                                                   conv_activation , ...
                                                   conv_bias_type , ...
                                                   conv_bias_cell , ...
                                                   fulconn_weight_cell , ...
                                                   fulconn_activation , ...
                                                   conv_convolution_output_cell , ...
                                                   conv_pooling_output_cell , ...
                                                   conv_pooling_map_cell , ...
                                                   fulconn_output_cell);

        % calculate MCEE
        desired_value = zeros(1,fulconn_output_neuron_count);
        desired_value(train_label(i_mcce_mce)) = 1;
        err = -sum(desired_value(:) .* log(fulconn_output_cell{end}(:)));  
        train_mcee_acc = train_mcee_acc + err;
        
        % calculate MCE
        desired_value = train_label(i_mcce_mce); 
        estimated_value = find(fulconn_output_cell{end} == max(fulconn_output_cell{end}),1);
        err = (desired_value ~= estimated_value ); 
        train_mce_acc = train_mce_acc + err;
        
    end

    train_mcee = train_mcee_acc/length(train_label)
    train_mce = train_mce_acc/length(train_label)
    
    train_mcee_per_epoch(i_epoch) = train_mcee;
    train_mce_per_epoch(i_epoch) = train_mce;

    
%     train_error_calculated_time = toc


%     % Calculate MCEE and MCE of validation data for each epoch
% 
%     validation_mcee_acc = 0;
%     validation_mce_acc = 0;
% 
%     for i_mcce_mce = 1:length(validation_label)
%         
%         % Prepare input vector
%         input_data = validation_data(i_mcce_mce,:);
% 
%         % Extract the individual red, green, and blue color channels.
%         %   IMPORTANT NOTE: This input configuration is just for this
%         %                   project.
%         red_ch = rot90(reshape(im2double(input_data(1:1024)),32,32),3);
%         green_ch = rot90(reshape(im2double(input_data(1024+1:1024*2)),32,32),3);
%         blue_ch = rot90(reshape(im2double(input_data(1024*2+1:1024*3)),32,32),3);
% 
%         % Store all of the data in a single vector
%         conv_input_layer = zeros(3,32,32);
%         conv_input_layer(1,:,:) = red_ch;
%         conv_input_layer(2,:,:) = green_ch;
%         conv_input_layer(3,:,:) = blue_ch;
% 
%         % Forward propagate conv layer input through the network.(func)
%         [conv_convolution_output_cell , ...
%          conv_pooling_output_cell , ...
%          conv_pooling_map_cell , ...
%          fulconn_input_layer , ...
%          fulconn_output_cell] = forward_propagate( conv_input_layer , ...
%                                                    conv_weight_cell , ...
%                                                    conv_zero_padding_size , ...
%                                                    conv_pooling_type , ...
%                                                    conv_pooling_size , ...
%                                                    conv_activation , ...
%                                                    conv_bias_type , ...
%                                                    conv_bias_cell , ...
%                                                    fulconn_weight_cell , ...
%                                                    fulconn_activation , ...
%                                                    conv_convolution_output_cell , ...
%                                                    conv_pooling_output_cell , ...
%                                                    conv_pooling_map_cell , ...
%                                                    fulconn_output_cell);
% 
%         % calculate MCEE
%         desired_value = zeros(1,fulconn_output_neuron_count);
%         desired_value(validation_label(i_mcce_mce)) = 1;
%         err = -sum(desired_value(:) .* log(fulconn_output_cell{end}(:))); 
%         validation_mcee_acc = validation_mcee_acc + err;
%         
%         % calculate MCE
%         desired_value = validation_label(i_mcce_mce); 
%         estimated_value = find(fulconn_output_cell{end} == max(fulconn_output_cell{end}),1);
%         err = (desired_value ~= estimated_value ); 
%         validation_mce_acc = validation_mce_acc + err;
% 
%     end
% 
%     validation_mcee_per_epoch(i_epoch) = validation_mcee_acc/length(validation_label);
%     validation_mce_per_epoch(i_epoch) = validation_mce_acc/length(validation_label);



    % Calculate MCCE and MCE of test data for each epoch

    test_mcee_acc = 0;
    test_mce_acc = 0;

    for i_mcce_mce = 1:length(test_label)
        
        
        
        % Prepare input vector
        input_data = test_data(i_mcce_mce,:);

        % Extract the individual red, green, and blue color channels.
        %   IMPORTANT NOTE: This input configuration is just for this
        %                   project.
        red_ch = rot90(reshape(im2double(input_data(1:1024)),32,32),3);
        green_ch = rot90(reshape(im2double(input_data(1024+1:1024*2)),32,32),3);
        blue_ch = rot90(reshape(im2double(input_data(1024*2+1:1024*3)),32,32),3);

        % Store all of the data in a single vector
        conv_input_layer = zeros(3,32,32);
        conv_input_layer(1,:,:) = red_ch;
        conv_input_layer(2,:,:) = green_ch;
        conv_input_layer(3,:,:) = blue_ch;

        % Forward propagate conv layer input through the network.(func)
        [conv_convolution_output_cell , ...
         conv_pooling_output_cell , ...
         conv_pooling_map_cell , ...
         fulconn_input_layer , ...
         fulconn_output_cell] = forward_propagate( conv_input_layer , ...
                                                   conv_weight_cell , ...
                                                   conv_zero_padding_size , ...
                                                   conv_pooling_type , ...
                                                   conv_pooling_size , ...
                                                   conv_activation , ...
                                                   conv_bias_type , ...
                                                   conv_bias_cell , ...
                                                   fulconn_weight_cell , ...
                                                   fulconn_activation , ...
                                                   conv_convolution_output_cell , ...
                                                   conv_pooling_output_cell , ...
                                                   conv_pooling_map_cell , ...
                                                   fulconn_output_cell);

        % calculate MCEE
        desired_value = zeros(1,fulconn_output_neuron_count);
        desired_value(test_label(i_mcce_mce)) = 1;
        err = -sum(desired_value(:) .* log(fulconn_output_cell{end}(:))); 
        test_mcee_acc = test_mcee_acc + err;
        
        % calculate MCE
        desired_value = test_label(i_mcce_mce); 
        estimated_value = find(fulconn_output_cell{end} == max(fulconn_output_cell{end}),1);
        err = (desired_value ~= estimated_value ); 
        test_mce_acc = test_mce_acc + err;
        
    end
    
    test_mcee = test_mcee_acc/length(test_label)
    test_mce = test_mce_acc/length(test_label)

    test_mcee_per_epoch(i_epoch) = test_mcee;
    test_mce_per_epoch(i_epoch) = test_mce;
    
    
%     test_error_calculated_time = toc
    
%     save('backup.mat','train_mcee_per_epoch','train_mce_per_epoch',...
%                       'validation_mcee_per_epoch','validation_mce_per_epoch',...
%                       'test_mcee_per_epoch','test_mce_per_epoch');

%% Store weight matrix computed for this epoch
conv_weight_per_epoch{i_epoch} = conv_weight_cell;
conv_bias_per_epoch{i_epoch} = conv_bias_cell;
fulconn_weight_per_epoch{i_epoch} = fulconn_weight_cell;

end



% Get the total elapsed time.
total_elapsed_time = toc;



%% Assign the output values

% Error metrics (mean cross entropy error and mean classification error)
output_struct.train_mcee_per_epoch = train_mcee_per_epoch;  
output_struct.train_mce_per_epoch = train_mce_per_epoch;       
output_struct.validation_mcee_per_epoch = validation_mcee_per_epoch;  
output_struct.validation_mce_per_epoch = validation_mce_per_epoch;       
output_struct.test_mcee_per_epoch = test_mcee_per_epoch;  
output_struct.test_mce_per_epoch = test_mce_per_epoch;      
% Resultant weight per epoch
output_struct.conv_weight_per_epoch = conv_weight_per_epoch; 
output_struct.conv_bias_per_epoch = conv_bias_per_epoch; 
output_struct.fulconn_weight_per_epoch = fulconn_weight_per_epoch; 
% Total elapsed time
output_struct.total_elapsed_time = total_elapsed_time;


end

%% Functions to define system parameters

%{

    This function calculates the output sizes of convolutioal layers
    according to the given hyperparameters.

    Output types : cell array

%}
function [conv_convolution_out_size_cell, conv_pooling_out_size_cell] ...
                  = calc_conv_out_size( conv_input_size , ...
                                        conv_layer_content , ...
                                        conv_zero_padding_size , ...
                                        conv_kernel_size , ...
                                        conv_pooling_size )
                                    
% define output
conv_convolution_out_size_cell = cell(size(conv_layer_content));
conv_pooling_out_size_cell = cell(size(conv_layer_content));

% Initialize loop parameters
layer_input_size = conv_input_size;

% Update output content iteratively
for i = 1: length(conv_layer_content)

    % Zero padding
    layer_input_size = layer_input_size + 2*conv_zero_padding_size(i);

    % Reduce dimension due to convolution.
    curr_kernel_size = conv_kernel_size{i};
    layer_kerneled_size = layer_input_size - curr_kernel_size + 1;
    conv_convolution_out_size_cell{i} = layer_kerneled_size;

    % Reduce dimension due to pooling.
    curr_pooling_size = conv_pooling_size{i};
    layer_pooled_size = ceil(layer_kerneled_size ./ curr_pooling_size);

    % Assign it as the output size of this conv layer.
    conv_pooling_out_size_cell{i} = layer_pooled_size;

    % Store the output size of previos one as the input size of next
    layer_input_size = layer_pooled_size;
end                                  
end

%{

    Define the convolutional layer weights (kernels) and initialize them
    according to "conv_kernel_init" input parameters content.

    == 'zeros' => Initialize all parameters to 0.
    o.w.       => Follow the directives of the input.

    Output type : cell array {1 , n} (m , k , x , y)

%}
function conv_weight_cell = prepare_conv_weight_init( conv_input_channel , ...
                                                      conv_layer_content , ...
                                                      conv_kernel_size , ...
                                                      conv_kernel_init )
    
%% Define outer cell array ( for each conv layer)
conv_weight_cell = cell(1,length(conv_layer_content));

%% Define inner arrays ( for kernel matrices in each layer)
in_ch_num = conv_input_channel;

for i = 1:length(conv_weight_cell)

    % Get the current layer's output channel count.
    out_ch_num = conv_layer_content(i);

    % Get the current layer's kernel size
    kernel_x_size = conv_kernel_size{i}(1);
    kernel_y_size = conv_kernel_size{i}(2);

    % Define the respective cell array
    conv_weight_cell{i} = zeros( in_ch_num , ...
                                 out_ch_num , ... 
                                 conv_kernel_size{i}(1) , ...
                                 conv_kernel_size{i}(2));

    % The input channel count of next layer will be output channel of
    % current layer.
    in_ch_num = conv_layer_content(i);
end

%% Initialize them according to the given settings (if not 'zeros').
for i = 1:length(conv_weight_cell)
    if ~(strcmp(conv_kernel_init,'zeros'))
        if strcmp(conv_kernel_init{i}{1},'normal')

            % Get random initialization parameters
            normal_mean = conv_kernel_init{i}{2};
            normal_std = conv_kernel_init{i}{3};

            % Get layer channel content
            in_ch_count = size(conv_weight_cell{i},1);
            out_ch_count = size(conv_weight_cell{i},2);

            % Get kernel content
            kernel_x_size = size(conv_weight_cell{i},3);
            kernel_y_size = size(conv_weight_cell{i},4);


            for j = 1:in_ch_count
                for k = 1:out_ch_count
                    conv_weight_cell{i}(j,k,:,:) = normrnd(normal_mean , ...
                                                           normal_std , ...
                                                           kernel_x_size , ...
                                                           kernel_y_size );
                end
            end

        elseif strcmp(conv_kernel_init{i}{1},'uniform')

            % Get random initialization parameters
            uniform_lower_lim = min([conv_kernel_init{i}{2} conv_kernel_init{i}{3}]);
            uniform_upper_lim = max([conv_kernel_init{i}{2} conv_kernel_init{i}{3}]);

            % Get layer channel content
            in_ch_count = size(conv_weight_cell{i},1);
            out_ch_count = size(conv_weight_cell{i},2);

            % Get kernel content
            kernel_x_size = size(conv_kernel_size{i},3);
            kernel_y_size = size(conv_kernel_size{i},4);


            for j = 1:in_ch_count
                for k = 1:out_ch_count
                    conv_weight_cell{i}(j,k,:,:) = unifrnd(uniform_lower_lim , ...
                                                           uniform_upper_lim , ...
                                                           kernel_x_size , ...
                                                           kernel_y_size );
                end
            end
        end
    end
end                                    
end


%{

    Define the convolutional layer bias and initialize them according to 
    "conv_kernel_init" input parameters content.

    == 'zeros' => Initialize all parameters to 0.
    o.w.       => Follow the directives of the input.

    Output type : cell array {1 , n} (k , x , y)

%}
function conv_bias_cell = prepare_conv_bias_init( conv_layer_content , ...
                                                  conv_out_size_cell , ...
                                                  conv_bias_type , ... 
                                                  conv_kernel_init )
                                              
%% Define outer cell array ( for each conv layer)
conv_bias_cell = cell(1,length(conv_layer_content));   

%% Define inner arrays ( for bias matrices in each layer)
for i = 1:length(conv_bias_cell)

    % Get the current layer's output channel count.
    out_ch_num = conv_layer_content(i);

    % Get the current layer's kernel size
    if strcmp(conv_bias_type{i},'untied')
        out_ch_x_size = conv_out_size_cell{i}(1);
        out_ch_y_size = conv_out_size_cell{i}(2);
    elseif strcmp(conv_bias_type{i},'tied')
        out_ch_x_size = 1;
        out_ch_y_size = 1;
    end

    % Define the respective cell array
    conv_bias_cell{i} = zeros( out_ch_num , ... 
                               out_ch_x_size , ...
                               out_ch_y_size);
end

%% Initialize them according to the given settings (if not 'zeros').
for i = 1:length(conv_bias_cell)
    if ~(strcmp(conv_kernel_init,'zeros'))
        if strcmp(conv_kernel_init{i}{1},'normal')

            % Get random initialization parameters
            normal_mean = conv_kernel_init{i}{2};
            normal_std = conv_kernel_init{i}{3};

            % Get layer channel content
            out_ch_count = size(conv_bias_cell{i},1);

            % Get kernel content
            bias_x_size = size(conv_bias_cell{i},2);
            bias_y_size = size(conv_bias_cell{i},3);


            for j = 1:out_ch_count
                conv_bias_cell{i}(j,:,:) = normrnd(normal_mean , ...
                                                   normal_std , ...
                                                   bias_x_size , ...
                                                   bias_y_size );
            end

        elseif strcmp(conv_kernel_init{i}{1},'uniform')

            % Get random initialization parameters
            uniform_lower_lim = min([conv_kernel_init{i}{2} conv_kernel_init{i}{3}]);
            uniform_upper_lim = max([conv_kernel_init{i}{2} conv_kernel_init{i}{3}]);

            % Get layer channel content
            out_ch_count = size(conv_bias_cell{i},1);

            % Get kernel content
            bias_x_size = size(conv_bias_cell{i},2);
            bias_y_size = size(conv_bias_cell{i},3);

            for j = 1:out_ch_count
                conv_bias_cell{i}(j,:,:) = unifrnd(uniform_lower_lim , ...
                                                   uniform_upper_lim , ...
                                                   bias_x_size , ...
                                                   bias_y_size );
            end
        end
    end
end                                    
end


%{

    Define the fully conencted layer weights and initialize them according  
    to "fulconn_weight_init" input parameters content.

    == 'zeros' => Initialize all parameters to 0.
    o.w.       => Follow the directives of the input.

    Output type : cell array {1 , n} (x , y)

---------------------------------------------------------------------------

    IMPORTANT NOTE : The bias is included.

%}
function fulconn_weight_cell = prepare_fulconn_weight_init( fulconn_input_neuron_count , ...
                                                            fulconn_output_neuron_count, ...
                                                            fulconn_hidden_layer_neuron_content,...
                                                            fulconn_weight_init)
% Number of Neurons per layer.
neuron_per_layer = [fulconn_input_neuron_count , ...
                    fulconn_hidden_layer_neuron_content(:)' , ...
                    fulconn_output_neuron_count];

% Number of connection levels (between layer)
conn_lvl_count = length(neuron_per_layer) - 1;

% Define the outer cell array
fulconn_weight_cell = cell(1,conn_lvl_count);

for i = 1 : conn_lvl_count
    
    % Get the surronding neuron counts
    source_neuron_count = neuron_per_layer(i);
    destination_neuron_count = neuron_per_layer(i+1);
    
    % Perform initialization (+1 in source+1 is to represent bias weights)
    if strcmp(fulconn_weight_init,'zeros')
        fulconn_weight_cell{i} = zeros(source_neuron_count+1,destination_neuron_count);
    else
        if strcmp(fulconn_weight_init{i}{1},'normal')
            
            % Get random initialization parameters
            normal_mean = fulconn_weight_init{i}{2};
            normal_std = fulconn_weight_init{i}{3};
            
            % Perform initialization
            fulconn_weight_cell{i} = normrnd(normal_mean , ...
                                             normal_std , ...
                                             source_neuron_count+1 , ...
                                             destination_neuron_count );
                                           
        elseif strcmp(fulconn_weight_init{i}{1},'uniform')
            
            % Get random initialization parameters
            uniform_lower_lim = min([fulconn_weight_init{i}{2} fulconn_weight_init{i}{3}]);
            uniform_upper_lim = max([fulconn_weight_init{i}{2} fulconn_weight_init{i}{3}]);
            
            % Perform the initialization
            fulconn_weight_cell{i} = unifrnd(uniform_lower_lim , ...
                                             uniform_upper_lim , ...
                                             source_neuron_count+1 , ...
                                             destination_neuron_count );
        end
    end
end                                              
end

%{

    Define the convolutional layer outputs and initialize them to 0.

    Output type : cell array {1 , n} (k , x , y)

%}
function conv_output_cell = prepare_conv_output_cell(conv_layer_content , ...
                                                     conv_out_size_cell)
% Define the cell array
conv_output_cell = cell(1,length(conv_layer_content));

% Initialize output neuron places for each convolutional layer
for i = 1:length(conv_layer_content)

    % Get the current conv layer kernel size
    out_ch_num = conv_layer_content(i);

    % Get the size of each kernel's output for current conv layer
    out_ch_x_size = conv_out_size_cell{i}(1);
    out_ch_y_size = conv_out_size_cell{i}(2);

    % Initialize the content of this layer as zeros
    conv_output_cell{i} = zeros(out_ch_num , out_ch_x_size , out_ch_y_size);
end 
end

%{

    Define the fully connected layer outputs and initialize them to 0.

    Output type : cell array {1 , n} (x , 1)
     
%}
function fulconn_output_cell = prepare_fulconn_output_cell( fulconn_output_neuron_count, ...
                                                            fulconn_hidden_layer_neuron_content)
% Concat total neuron information
fulconn_layer_content = [fulconn_hidden_layer_neuron_content(:)' , ...
                            fulconn_output_neuron_count];

% Define the cell array 
fulconn_output_cell = cell(1,length(fulconn_layer_content));

% Initialize output neuron places for each fully connected layer
for i = 1:length(fulconn_layer_content)

    % Get the current fulconn layer neuron size.
    neuron_num = fulconn_layer_content(i);

    % Initialize the content of this layer as zeros
    fulconn_output_cell{i} = zeros(neuron_num,1);

end                                             
end

%% Functions to implement Forward Propagation

%{

    Implement forward propagation through the network
     
%}
function [conv_convolution_output_cell , ...
          conv_pooling_output_cell , ...
          conv_pooling_map_cell , ...
          fulconn_input_layer , ...
          fulconn_output_cell] = forward_propagate( conv_input_layer , ...
                                                    conv_weight_cell , ...
                                                    conv_zero_padding_size , ...
                                                    conv_pooling_type , ...
                                                    conv_pooling_size , ...
                                                    conv_activation , ...
                                                    conv_bias_type , ...
                                                    conv_bias_cell , ...
                                                    fulconn_weight_cell , ...
                                                    fulconn_activation , ...
                                                    conv_convolution_output_cell , ...
                                                    conv_pooling_output_cell , ...
                                                    conv_pooling_map_cell , ...
                                                    fulconn_output_cell)

% Implement conv forward propagation
for i = 1 : length(conv_weight_cell)
    % conv layer one step forward propagate
    [conv_convolution_output , ...
     conv_pooling_output , ...
     conv_pooling_map ] = conv_one_step_forward_propagate( conv_input_layer , ...
                                                           conv_weight_cell{i} , ...
                                                           conv_zero_padding_size(i) , ...
                                                           conv_pooling_type{i} , ...
                                                           conv_pooling_size{i} , ...
                                                           conv_activation{i} , ...
                                                           conv_bias_type{i} , ...
                                                           conv_bias_cell{i} , ...
                                                           conv_convolution_output_cell{i} , ...
                                                           conv_pooling_output_cell{i} , ...
                                                           conv_pooling_map_cell{i});
 
    % Update output parameters
    conv_convolution_output_cell{i} = conv_convolution_output;
    conv_pooling_output_cell{i} = conv_pooling_output;
    conv_pooling_map_cell{i} = conv_pooling_map;
    
    % Update input layer
    conv_input_layer = conv_pooling_output;
end

% Transfer the conv part output to fulconn part as input layer
fulconn_input_layer = conv2fulconn_connect( conv_pooling_output );

% Prepare input for forward propagation
fulconn_input = fulconn_input_layer;

% Implement fulconn forward propagation
for i = 1 : length(fulconn_weight_cell)
    % fulconn layer cone step forward propagate
    [fulconn_output] = fulconn_one_step_forward_propagate( fulconn_input , ...
                                                           fulconn_weight_cell{i} , ...
                                                           fulconn_activation{i});
    
    % Update output parameters
    fulconn_output_cell{i} = fulconn_output;
    
    % Update input layer
    fulconn_input = fulconn_output;
end                                                
end

%{

    Implement one step forward propagation through conv part.

    NOTE : conv_bias_type seems to be unused.
     
%}
function [conv_convolution_output , ...
          conv_pooling_output , ...
          conv_pooling_map ] = conv_one_step_forward_propagate( conv_input_layer , ...
                                                                conv_weight , ...
                                                                conv_zero_padding_size , ...
                                                                conv_pooling_type , ...
                                                                conv_pooling_size , ...
                                                                conv_activation , ...
                                                                conv_bias_type , ...
                                                                conv_bias , ...
                                                                conv_convolution_output , ...
                                                                conv_pooling_output , ...
                                                                conv_pooling_map )
                                                            
% For each output channel, perform convolution and pooling
for i = 1 : size(conv_weight,2)

    %% Convolution part

    % Initialize conv output accumulator
    acc = zeros(size(conv_convolution_output,2),size(conv_convolution_output,3));

    % Accumulate each convolution operation.
    for j = 1: size(conv_weight,1)
        % Get parameters
        input_ch = reshape(conv_input_layer(j , : , :),size(conv_input_layer,2),size(conv_input_layer,3));
        kernel = reshape(conv_weight( j , i , : , : ),size(conv_weight,3),size(conv_weight,4));
        % Perform zero padding
        zp_input_ch = padarray(input_ch,[conv_zero_padding_size conv_zero_padding_size]);
        % Perform convolution and accumulation
        acc = acc + conv2(zp_input_ch , rot90(kernel,2) , 'valid' );
    end

    % Add bias
    if strcmp(conv_bias_type,'tied')
        biased_acc = acc + conv_bias(i);
    else
        biased_acc = acc + reshape(conv_bias(i,:,:),size(conv_bias,2),size(conv_bias,3));
    end

    % Apply transfer function
    activated_acc = apply_activation( biased_acc , conv_activation );

    % Assign to the output
    conv_convolution_output(i,:,:) = activated_acc;

    %% Pooling part

    % Check the given pooling type
    if strcmp(conv_pooling_type,'max')
        
        % Pre pooling size
        activated_acc_x_len = size(activated_acc,1);
        activated_acc_y_len = size(activated_acc,2);
        
        % Post pooling size
        pooled_activated_acc_x_len = ceil(activated_acc_x_len/conv_pooling_size(1));
        pooled_activated_acc_y_len = ceil(activated_acc_y_len/conv_pooling_size(2));
        
        % Pooling template
        pooling_template = ones(conv_pooling_size(1) , conv_pooling_size(2));

        % Index respective pooing regions
        idx_sequence = 1:(pooled_activated_acc_x_len*pooled_activated_acc_y_len);
        idx = kron(reshape(idx_sequence,pooled_activated_acc_y_len,[]).',pooling_template);
        idx = idx(1:activated_acc_x_len , 1:activated_acc_y_len);

        % Perform max pooling
        maxpool = reshape(accumarray(idx(:),activated_acc(:),[],@(x) max(x)),pooled_activated_acc_y_len,[]).';

        % Calculate pooling map
        kroned_max_pool = kron(maxpool,pooling_template);
        kroned_max_pool = kroned_max_pool(1:activated_acc_x_len , 1:activated_acc_y_len);
        maxpool_map = (activated_acc - kroned_max_pool) == 0;
        
        % Assign to the output
        conv_pooling_output(i,:,:) = maxpool;
        conv_pooling_map(i,:,:) = maxpool_map;
        
        

%         for p_i = 1 : size(conv_pooling_output,2)
%             for p_j = 1 : size(conv_pooling_output,3)
% 
%                 % Calculate the respective indices
%                 i_indices = ((p_i-1)*conv_pooling_size(1) + 1) : min((p_i*conv_pooling_size(1)) , size(activated_acc,1));
%                 j_indices = ((p_j-1)*conv_pooling_size(2) + 1) : min((p_j*conv_pooling_size(2)) , size(activated_acc,2));
% 
%                 % Get the to be pooled data
%                 pool_data = activated_acc(i_indices,j_indices);
% 
%                 % Find the max of the pool
%                 pool_output = max(max(pool_data));
% 
%                 % Prepare map of pooling operation
%                 [pooling_map_i , pooling_map_j] = find(pool_data == pool_output);
%                 dummy_pool_map = zeros( length(i_indices) , length(j_indices) );
%                 dummy_pool_map(pooling_map_i,pooling_map_j) = 1;
% 
%                 % Assign to the output
%                 conv_pooling_output(i,p_i,p_j) = pool_output;
%                 conv_pooling_map(i,i_indices,j_indices) = dummy_pool_map;
%             end
%         end

    elseif strcmp(conv_pooling_type,'mean')

        for p_i = 1 : size(conv_pooling_output,2)
            for p_j = 1 : size(conv_pooling_output,3)

                % Calculate the respective indices
                i_indices = ((p_i-1)*conv_pooling_size(1) + 1) : min((p_i*conv_pooling_size(1)) , size(activated_acc,1));
                j_indices = ((p_j-1)*conv_pooling_size(2) + 1) : min((p_j*conv_pooling_size(2)) , size(activated_acc,2));

                % Get the to be pooled data
                pool_data = activated_acc(i_indices,j_indices);

                % Find the mean of the pool
                pool_output = mean(pool_data(:));

                % Prepare map of pooling operation
                dummy_pool_map = (1/(length(i_indices) * length(j_indices))) ...
                    .* ones( length(i_indices) , length(j_indices) );

                % Assign to the output
                conv_pooling_output(i,p_i,p_j) = pool_output;
                conv_pooling_map(i,i_indices,j_indices) = dummy_pool_map;
            end
        end

    else
        error('Invalid pooling type name.');
    end

end                                                  
end

%{

    Convert the conv output to fulconn input format.
     
%}
function fulconn_input_layer = conv2fulconn_connect( conv_pooling_output )

% Allocate memory space for the output
fulconn_input_layer = zeros(length(conv_pooling_output(:)),1);

% Fro each channel
for i = 1 : size(conv_pooling_output,1)

    % Get channel data and respective information
    ch_data = conv_pooling_output(i,:,:);
    ch_data = ch_data(:);
    ch_data_length = length(ch_data);

    % Calculate respective address region 
    idx = ((i-1)*ch_data_length) + 1;

    % Store the data.
    fulconn_input_layer( idx : (idx + ch_data_length - 1) ) = ch_data;
end
end

%{

    Implement one step forward propagation through fulconn part.
     
%}
function [fulconn_output] = fulconn_one_step_forward_propagate( fulconn_input_layer , ...
                                                                fulconn_weight , ...
                                                                fulconn_activation)
% Perform matrix multiplication. (pads the bias 1)
v = [fulconn_input_layer(:)' 1] * fulconn_weight;   

% Apply activation function
fulconn_output = apply_activation(v,fulconn_activation);
end


%% Functions to implement Backpropagate Neuron Gradient

%{

    Backpropagate the gradient
     
%}
function [conv_gradient_cell , ...
          fulconn_gradient_cell ] = backprop_gradient( fulconn_err , ...
                                                       fulconn_weight_cell , ...
                                                       fulconn_activation , ...
                                                       fulconn_output_cell , ...
                                                       conv_zero_padding_size , ...
                                                       conv_pooling_out_size_cell , ...
                                                       conv_content , ...
                                                       conv_pooling_map_cell , ...
                                                       conv_pooling_size , ...
                                                       conv_activation , ...
                                                       conv_convolution_output_cell , ...
                                                       conv_weight_cell , ...
                                                       conv_gradient_cell , ...
                                                       fulconn_gradient_cell)
    
% Backpropagate error through fulconn part
[fulconn_gradient_cell , ...
 fulconn_input_gradient ] = fulconn_backprop_gradient ( fulconn_err , ...
                                                        fulconn_weight_cell , ...
                                                        fulconn_activation , ...
                                                        fulconn_output_cell , ...
                                                        fulconn_gradient_cell);

% Connect the fulconn gradient information to conv gradient layers
[conv_err] = fulconn2conv_connect ( conv_pooling_out_size_cell{end} , ...
                                    conv_content(end) , ... 
                                    fulconn_input_gradient , ...
                                    conv_activation{end} , ...
                                    conv_convolution_output_cell{end} , ...
                                    conv_pooling_size{end} , ...
                                    conv_pooling_map_cell{end});


% Backpropagate error through conv part
[conv_gradient_cell] = conv_backprop_gradient ( conv_err , ...
                                                conv_zero_padding_size , ...
                                                conv_pooling_out_size_cell , ...
                                                conv_content , ...
                                                conv_weight_cell , ...
                                                conv_pooling_map_cell , ...
                                                conv_pooling_size , ...
                                                conv_convolution_output_cell , ...
                                                conv_activation , ...
                                                conv_gradient_cell);

end

%{

    Fulconn layer one step back propagating gradient.
     
%}
function [fulconn_gradient_cell , ...
          fulconn_input_gradient ] = fulconn_backprop_gradient ( fulconn_err , ...
                                                                 fulconn_weight_cell , ...
                                                                 fulconn_activation , ...
                                                                 fulconn_output_cell , ...
                                                                 fulconn_gradient_cell)   
% Store last fulconn layer gradient                                                      
fulconn_gradient_cell{end} = fulconn_err(:);

% Backpropagate the error through the rest of fulconn layers.
if length(fulconn_gradient_cell) > 1
    for i = (length(fulconn_gradient_cell) - 1) : -1 : 1

        % Get currently processed layer's output.
        layer_output = fulconn_output_cell{i}(:); 
        
        % Get current layer's activation type
        act_type = fulconn_activation{i};
        
        % Calculate derivative of act for this layer
        act_derivative = derivative_of_activation(layer_output,act_type);
        
        % Calculate the next gradient. (Check the order of matrices)
        fulconn_gradient_cell{i} = ((fulconn_weight_cell{i+1}(1:end-1,:)) ...
            *(fulconn_gradient_cell{i+1})).*act_derivative;
    end
end 

% Finally, calculate gradient at the input neurons of the fulconn part.
%   Note that, the activation on the fulconn input layers is linear. Thus,
%   its derivative is just 1. Thus, we just need to backpropagate the
%   gradient through the weights.
fulconn_input_gradient = ((fulconn_weight_cell{1}(1:end-1,:)) ...
                         *(fulconn_gradient_cell{1}));
                                            
end

%{

    Fulconn layer to conv layer gradient connection.
     
%}
function [conv_err] = fulconn2conv_connect ( conv_last_layer_pooling_out_size , ...
                                             conv_last_layer_content , ... 
                                             fulconn_input_gradient , ...
                                             conv_last_layer_activation , ...
                                             conv_last_layer_convolution_output , ...
                                             conv_last_layer_pooling_size , ...
                                             conv_last_layer_pooling_map )
% Allocate memory for the output
conv_err = conv_last_layer_convolution_output;

% Prepare a container for the conv pooling last layer
conv_last_layer_pooling_output = zeros( conv_last_layer_content , ...
                                        conv_last_layer_pooling_out_size(1) , ...
                                        conv_last_layer_pooling_out_size(2));

single_pooling_element_num = conv_last_layer_pooling_out_size(1) ...
                             * conv_last_layer_pooling_out_size(2);
                                    
for i = 1 : conv_last_layer_content
    
    % At the first step, reshape the fulconn input neurons in the conv output
    %   format.
    fulconn_input_partial = fulconn_input_gradient ( ...
                            ( (i-1)*single_pooling_element_num + 1 ) : ...
                            ( i*single_pooling_element_num ) ) ;
                        
    fulconn_input_partial_reshaped = reshape ( fulconn_input_partial , ...
                                               conv_last_layer_pooling_out_size(1) , ...
                                               conv_last_layer_pooling_out_size(2));
    
    conv_last_layer_pooling_output(i,:,:) = fulconn_input_partial_reshaped;
    
    % Secondly, backpropagate the gradient through pooling layer.
    pooling_region = ones ( conv_last_layer_pooling_size(1) , ...
                            conv_last_layer_pooling_size(2));
                         
    upsampled_pool = kron(fulconn_input_partial_reshaped , pooling_region);
    
    upsampled_pool = upsampled_pool(1:size(conv_last_layer_pooling_map,2) , ...
                                    1:size(conv_last_layer_pooling_map,3));
                                
    pooling_map_2D = reshape(conv_last_layer_pooling_map(i,:,:) , ...
                             size(conv_last_layer_pooling_map,2) , ...
                             size(conv_last_layer_pooling_map,3));
    
    unpooled_data = upsampled_pool.*pooling_map_2D;
    
    % Finally, backpropagate the error through convolution activation.
    convolution_data_2D = reshape(conv_last_layer_convolution_output(i,:,:) , ...
                                  size(conv_last_layer_pooling_map,2) , ...
                                  size(conv_last_layer_pooling_map,3));
                              
    conv_err(i,:,:) = unpooled_data .* derivative_of_activation( convolution_data_2D , ...
                                                                 conv_last_layer_activation );
end                                         
end

%{

    Conv layer one step back propagating gradient.
     
%}
function [conv_gradient_cell] = conv_backprop_gradient ( conv_err , ...
                                                         conv_zero_padding_size , ...
                                                         conv_pooling_out_size_cell , ...
                                                         conv_content , ...
                                                         conv_weight_cell , ...
                                                         conv_pooling_map_cell , ...
                                                         conv_pooling_size , ...
                                                         conv_convolution_output_cell , ...
                                                         conv_activation , ...
                                                         conv_gradient_cell)
% Store last conv layer gradient
conv_gradient_cell{end} = conv_err;

% Backpropagate the error through the rest of conv layers.
if length(conv_gradient_cell) > 1
    
    for i_layer = (length(conv_gradient_cell)-1) : -1 : 1
        
        % Zero padding size 
        zero_padding_size = conv_zero_padding_size(i_layer);
    
        % Prepare container for prev layer pooling output
        conv_gardient_at_pooling_output = zeros( conv_content(i_layer) , ...
                                                 conv_pooling_out_size_cell{i_layer}(1) , ...
                                                 conv_pooling_out_size_cell{i_layer}(2));

        % Firstly, backpropagate error through kernel weights (WORST PART)
        
        for i = 1 : size(conv_weight_cell{i_layer+1},1)
            
            gradient_per_input = zeros(conv_pooling_out_size_cell{i_layer}(1) + zero_padding_size*2 , ...
                                       conv_pooling_out_size_cell{i_layer}(2) + zero_padding_size*2);
                
            for q = 1 : size(conv_weight_cell{i_layer+1},2);      
                
                K_qi = reshape(conv_weight_cell{i_layer+1}(i,q,:,:) , ...
                               size(conv_weight_cell{i_layer+1},3) , ... 
                               size(conv_weight_cell{i_layer+1},4));
                           
                G_q = reshape(conv_gradient_cell{i_layer+1}(q,:,:) , ...
                              size(conv_gradient_cell{i_layer+1},2) , ...
                              size(conv_gradient_cell{i_layer+1},3));
                          
                G_q_padded = padarray(G_q , [(size(K_qi,1)-1) (size(K_qi,2)-1)]);
                     
                % Perform update
                gradient_per_input = gradient_per_input + conv2(G_q_padded , K_qi , 'valid');
                
%                 for l = 1:size(G_q,1)
%                     for n = 1:size(G_q,2)
%                         
%                         % Prepare target regions
%                         target_x_region = l:(l+size(K_qi,1)-1);
%                         target_y_region = n:(n+size(K_qi,2)-1);
%                         
%                         % Perform update
%                         gradient_per_input( target_x_region , target_y_region ) = gradient_per_input( target_x_region , target_y_region ) ...
%                         + (K_qi * G_q(l,n));
%                 
%                     end
%                 end
                
            end
                                   
            
            % Assign valid region to the respective gradient place
            conv_gardient_at_pooling_output(i,:,:) = ...
                    gradient_per_input((zero_padding_size+1):(end-zero_padding_size) , ...
                                       (zero_padding_size+1):(end-zero_padding_size));
            
        end
        
        
        
        
        % Secondly, backpropagate error through pooling the pooling layer

        for i = 1:conv_content(i_layer)

            pooling_region = ones ( conv_pooling_size{i_layer}(1) , ...
                                    conv_pooling_size{i_layer}(2));

            pooled_data_2D = reshape(conv_gardient_at_pooling_output(i,:,:) , ...
                                     conv_pooling_out_size_cell{i_layer}(1) , ...
                                     conv_pooling_out_size_cell{i_layer}(2));

            upsampled_pool = kron(pooled_data_2D , pooling_region);

            upsampled_pool = upsampled_pool(1:size(conv_pooling_map_cell{i_layer},2) , ...
                                            1:size(conv_pooling_map_cell{i_layer},3));
                                        
            pooling_map_2D = reshape(conv_pooling_map_cell{i_layer}(i,:,:) , ...
                                     size(conv_pooling_map_cell{i_layer} , 2) , ...
                                     size(conv_pooling_map_cell{i_layer} , 3));

            unpooled_data = upsampled_pool.*pooling_map_2D;

            % Thirdly, backpropagate error through convolutional activation.
            convolution_data_2D = reshape(conv_convolution_output_cell{i_layer}(i,:,:) , ...
                                          size(conv_convolution_output_cell{i_layer}(i,:,:),2) , ...
                                          size(conv_convolution_output_cell{i_layer}(i,:,:),3));
                              
            conv_err = unpooled_data .* derivative_of_activation( convolution_data_2D , ...
                                                                  conv_activation{i_layer} );
                                                          
            % Update output
            conv_gradient_cell{i_layer}(i,:,:) = conv_err;

        end
    
    end
    
end                                                            
end


%% Functions to calculate delta weight values

%{

    Implement the activation function application.
     
%}
function [conv_delta_weight_cell , ...
          conv_delta_bias_cell , ...
          fulconn_delta_weight_cell] = calc_delta_weight( conv_learning_rate , ...
                                                          conv_zero_padding_size , ...
                                                          conv_input_layer , ...
                                                          conv_pooling_output_cell , ...
                                                          conv_gradient_cell , ...
                                                          fulconn_learning_rate , ...
                                                          fulconn_input_layer , ...
                                                          fulconn_output_cell , ...
                                                          fulconn_gradient_cell , ...
                                                          conv_bias_type , ...
                                                          conv_delta_weight_cell , ...
                                                          conv_delta_bias_cell , ...
                                                          fulconn_delta_weight_cell )
% For conv part                                                      
for i = 1 : length ( conv_learning_rate)

    % Calculate a conv layer delta weights
    [conv_delta_weight , ...
     conv_delta_bias] = conv_one_step_calc_delta_weight ( conv_learning_rate(i) , ...
                                                          conv_zero_padding_size(i) , ...
                                                          conv_input_layer , ...
                                                          conv_gradient_cell{i} , ...
                                                          conv_bias_type{i} , ...
                                                          conv_delta_weight_cell{i} , ...
                                                          conv_delta_bias_cell{i} );
    % Update outputs
    conv_delta_weight_cell{i} = conv_delta_weight;
    conv_delta_bias_cell{i} = conv_delta_bias;

    % Prepare the next conv layer input
    conv_input_layer = conv_pooling_output_cell{i};

end

% For fulconn part
for i = 1 : length ( fulconn_learning_rate )
    
    % Calculate a fulconn layer delta weights
    [fulconn_delta_weight] = fulconn_one_step_calc_delta_weight ( fulconn_learning_rate(i) , ...
                                                                  fulconn_input_layer , ...
                                                                  fulconn_gradient_cell{i});  
                                                                   
    % Update outputs
    fulconn_delta_weight_cell{i} = fulconn_delta_weight;
    
    % Prepare the next fulconn layer input
    fulconn_input_layer = fulconn_output_cell{i};
                                                            
end
end

%{

    Implement conv part one step delta weight calculation.
     
%}
function [conv_delta_weight , ...
          conv_delta_bias] = conv_one_step_calc_delta_weight ( conv_learning_rate , ...
                                                               conv_zero_padding_size , ...
                                                               conv_input_layer , ...
                                                               conv_gradient , ...
                                                               conv_bias_type , ...
                                                               conv_delta_weight , ...
                                                               conv_delta_bias )
% Calculate conv delta_weight                                                           
                                                           
for i = 1 : size(conv_delta_weight,2) 
    for l = 1 : size(conv_delta_weight,1)
        
        % Prepare input and gradient
        V_l = reshape( conv_input_layer(l,:,:) , ...
                       size(conv_input_layer,2) , ...
                       size(conv_input_layer,3));
        % Perform zero padding to V_l
        V_l = padarray(V_l,[conv_zero_padding_size conv_zero_padding_size]);
        
        G_i = reshape( conv_gradient(i,:,:) , ...
                       size(conv_gradient,2) , ...
                       size(conv_gradient,3));
       
        % Update weight
        K_il = conv2( V_l , rot90(G_i,2) , 'valid' );
         
        % Update delta weight
        conv_delta_weight(l,i,:,:) = - conv_learning_rate * K_il;
    end
    
    % Update bias
    gradient = reshape(conv_gradient(i,:,:) , ...
                       size(conv_gradient,2) , ...
                       size(conv_gradient,3));
    if strcmp(conv_bias_type , 'tied')
        conv_delta_bias(i,:,:) = - conv_learning_rate * sum(gradient(:));
    elseif strcmp(conv_bias_type , 'untied')
        conv_delta_bias(i,:,:) = - conv_learning_rate * gradient;
    end
        
end

end

%{

    Implement fulconn part one step delta weight calculation.
     
%}
function [fulconn_delta_weight] = fulconn_one_step_calc_delta_weight ( fulconn_learning_rate , ...
                                                                       fulconn_input_layer , ...
                                                                       fulconn_gradient)
% Calculate fulconn delta_weight
curr_layer_output = [fulconn_input_layer(:) ; 1]; % Add bias
next_layer_gradient = fulconn_gradient(:)';
fulconn_delta_weight = -fulconn_learning_rate*(curr_layer_output*next_layer_gradient);                                                             
                                                              
end



%% General purpose functions

%{

    Implement the activation function application.
     
%}
function activation_output = apply_activation(v,activation_type)

    % Check the given activation type parameter.
    if strcmp( activation_type , 'ReLU' )
        % Appy ReLU activation function.
        activation_output = v.*(v>0);
    elseif strcmp( activation_type , 'tanh' )
        % Apply tanh activation function.
        activation_output = tanh(v); 
    elseif strcmp( activation_type , 'logistic' )
        % Apply sigmoid activation function.
        activation_output = sigmoid(v);  
    elseif strcmp( activation_type , 'softmax' )
        % Apply softmax function
        activation_output = exp(v) / sum(exp(v));  
    else
        % Issue an error message.
        error('Invalid activation function name.');
    end
end

%{

    Implement the derivative of activation function.
     
%}
function derivative_output = derivative_of_activation(o,activation_type)

    % Check the given activation type parameter.
    if strcmp( activation_type , 'ReLU' )
        % Appy ReLU activation function.
        derivative_output = ( o > 0 );
    elseif strcmp( activation_type , 'tanh' )
        % Apply tanh activation function.
        derivative_output = 1-o.^2; 
    elseif strcmp( activation_type , 'logistic' )
        % Apply sigmoid activation function.
        derivative_output = o-o.^2;  
%     elseif strcmp( activation_type , 'softmax' )
%         % Apply softmax function
%         derivative_output = 0;  
    else
        % Issue an error message.
        error('Invalid activation function name.');
    end
end

%{

FUNCTION : sigmoid

Implements the sigmoid activaion function.

%}
function [o] = sigmoid(v)

o = 1./(1+exp(-v));

end












                                    
                                    
                                    
