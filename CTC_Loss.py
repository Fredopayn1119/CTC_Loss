import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function



#function for applying the forward backward algorithm
def forward_backward_algorithm(probs, target, target_lengths, blank_index):

  num_batches = probs.size(0)
  num_time_frames = probs.size(1)
  num_labels = probs.size(2)


  #variable to store loss for each sequence/batch
  losses = torch.zeros(num_batches)


  #variables that will receive the forward and backward variables of each batch
  combined_forward = []
  combined_backward = []
  modified_targets = []

  #variable to store rescaling factors of each timestep for each batch/sequence
  C_t = torch.zeros(num_batches,num_time_frames)


  
  #get the maximum modified target length (for padding forward and backward variables later on)
  temp_target_lengths = target_lengths.clone()
  max_modified_target_len = torch.max(temp_target_lengths.apply_(lambda x: 2*x+1))

  

 #go through each batch (each sequence)

 #get the appropriate sequence (target) length for the current batch using target_lengths
  sum_prev_batches_sizes = 0
  for batch in range(num_batches):
      curr_target = []
      if batch == 0:
        curr_target = target[:target_lengths[batch]]
      else:
        
        curr_target = target[sum_prev_batches_sizes:(sum_prev_batches_sizes+target_lengths[batch])]
      
      sum_prev_batches_sizes += target_lengths[batch]
      curr_target_length = target_lengths[batch].item()

      #creating modified target sequence l' with blanks added (resulting in a sequence of length (2*len(target))+1)
      modified_target = [blank_index]
      for i in curr_target:
        modified_target += [i.item(), blank_index]
        
      modified_target = torch.tensor(modified_target, dtype=torch.long)


      #initial all-zeros tensors for forward and backward variables
      a = torch.zeros(num_time_frames, len(modified_target), dtype=torch.float)
      b = torch.zeros(num_time_frames, len(modified_target), dtype=torch.float)


      #pad forward and backward variables based on largest 
      a = F.pad(a, pad=(0, max_modified_target_len-len(modified_target), 0, 0))
      b = F.pad(b, pad=(0, max_modified_target_len-len(modified_target), 0, 0))


      #initializing the forward variables
      a[0,0] = probs[batch,0,blank_index]
      a[0,1] = probs[batch,0,curr_target[0]]


      log_probs = 0


      #recursion loop to get values for forward variable
      for t in range(1,num_time_frames):
        for s in range(len(modified_target)):
          if modified_target[s] == blank_index or modified_target[s-2] == modified_target[s]:
            if (s-1 <= len(modified_target)-1 and s-1 >= 0):
              a[t,s] = (a[t-1,s] + a[t-1,s-1]) * probs[batch,t,modified_target[s]]
            else:
              a[t,s] = (a[t-1,s] + 0) * probs[batch,t,modified_target[s]]
          else:
            if (s-2 <= len(modified_target)-1 and s-2 >= 0):
              a[t,s] = (a[t-1,s] + a[t-1,s-1] + a[t-1,s-2]) * probs[batch,t,modified_target[s]]
            else:
              a[t,s] = (a[t-1,s] + a[t-1,s-1] + 0) * probs[batch,t,modified_target[s]]          

        #rescaling the values of a[t,s] for the given t to prevent underflow as mentioned in the paper
        C_t[batch,t] = torch.sum(a[t])
        a[t] = a[t] / torch.sum(a[t])



      #initializing the backward variables
      b[num_time_frames-1,len(modified_target)-1] = probs[batch,num_time_frames-1,blank_index]
      b[num_time_frames-1,len(modified_target)-2] = probs[batch,num_time_frames-1,curr_target[len(curr_target)-1]]


      for t in range(num_time_frames-2,0,-1):
        for s in range(len(modified_target)-1,0,-1):
          if modified_target[s] == blank_index or modified_target[s-2] == modified_target[s]:
            if (s+1 <= len(modified_target)-1 and s+1 >= 0):
              b[t,s] = (b[t+1,s] + b[t+1,s+1]) * probs[batch,t,modified_target[s]]
            else:
              b[t,s] = (b[t+1,s] + 0) * probs[batch,t,modified_target[s]]
          else:
            if (s+2 <= len(modified_target)-1 and s+2 >= 0):
              b[t,s] = (b[t+1,s] + b[t+1,s+1] + b[t+1,s+2]) * probs[batch,t,modified_target[s]]
            else:
              b[t,s] = (b[t+1,s] + b[t+1,s+1] + 0) * probs[batch,t,modified_target[s]]

        #rescaling the values of b[t][s] for the given t to prevent underflow as mentioned in the paper
        b[t] = b[t] / torch.sum(b[t])

      


      #calculating the loss using rescaling variable for current batch/sequence
      losses[batch] = torch.logsumexp(C_t[batch],dim=0)

      #pad the targets to have the length of max(target_lengths)
      padding_zeros = torch.zeros([max_modified_target_len-len(modified_target)],dtype=torch.long)
      modified_target = torch.cat([modified_target, padding_zeros],dim=0)

      combined_forward.append(a)
      combined_backward.append(b)
      modified_targets.append(modified_target)

  tensor_combined_forward = torch.stack((combined_forward), dim=0)
  tensor_combined_backward = torch.stack((combined_backward), dim=0)
  tensor_modified_targets = torch.stack((modified_targets), dim=0)    

  return losses,tensor_combined_forward,tensor_combined_backward,tensor_modified_targets





#helper function to compute ctc gradient in backward function
def ctc_gradient(probs,a_vars,b_vars,modified_targets,target_lengths):

  num_batches = probs.size(0)
  num_time_frames = probs.size(1)
  num_labels = probs.size(2)


  #initialize tensor for gradient
  gradient = torch.zeros(num_batches,num_time_frames,num_labels,dtype=torch.float)

  #compute the gradient of the objective function as paper describes
  for batch in range(num_batches):
    for t in range(num_time_frames):
      for k in range(num_labels):
        a_b_sum = 0
        Z_t = 0
        for s in range(len(modified_targets[batch])):
          Z_t += (a_vars[batch,t,s] * b_vars[batch,t,s])/probs[batch,t,modified_targets[batch,s]]
          if modified_targets[batch,s] == k:
            a_b_sum += a_vars[batch,t,s] * b_vars[batch,t,s]
        gradient[batch,t,k] = probs[batch,t,k] - ((a_b_sum) / probs[batch,t,k]*Z_t)

  return gradient
        
        




class RNNModel(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
    super(RNNModel, self).__init__()

    self.hidden_dim = hidden_dim
    self.layer_dim = layer_dim

    self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
    self.fc = nn.Linear(hidden_dim, output_dim)
    self.softmax = nn.Softmax(dim=2)

  def forward(self, x):
    h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
    out, hn = self.rnn(x, h0)
    out = self.fc(out)
    out = self.softmax(out)
    return out





class CTCImplementation(Function):
  @staticmethod
  def forward(ctx, rnn_output, targets, target_lengths, blank_index):
    log_probs = F.log_softmax(rnn_output,dim=2)
    losses, a_vars, b_vars, modified_targets = forward_backward_algorithm(log_probs, targets, target_lengths, blank_index)
    ctx.save_for_backward(log_probs, rnn_output, targets, target_lengths, a_vars, b_vars, modified_targets)

    return losses.sum()


  @staticmethod
  def backward(ctx, grad_output):
    log_probs, rnn_output, targets, target_lengths, a_vars, b_vars, modified_targets = ctx.saved_tensors
    num_batches = rnn_output.size(0)
    num_time_frames = rnn_output.size(1)
    num_labels = rnn_output.size(2)

    gradient = ctc_gradient(rnn_output, a_vars, b_vars, modified_targets, target_lengths)


    return gradient*grad_output, None, None, None, None




#Example dimensions for RNN

input_dim = 15
hidden_dim = 16
num_layers = 3

#the output dimensions assumes that there are a total of 27 possible characters, where one of them respresents a blank character
output_dim = 27 

#defining the blank index as 26
blank_index = 26


num_time_frames = 15
num_batches = 2

example_input = torch.randn(num_batches, num_time_frames, input_dim)

#targets are concatenated (this tensor contains 2 targets corresponding to a batch size of 2)
#this example target translates to [m, y, c, a, t], where the 2 targets are m,y and c,a,t
example_target = torch.tensor([12, 24, 2, 0, 19], dtype=torch.long)

#length/num labels in each target/batch
example_target_lengths = torch.tensor([2, 3], dtype=torch.long)

#the implementation assumes that all targets have an equal input length (equal number of time frames)
example_input_lengths = torch.tensor([num_time_frames, num_time_frames], dtype=torch.long)


#initialize ctc and rnn model
ctc = CTCImplementation.apply
rnn_mdl = RNNModel(input_dim, hidden_dim, num_layers, output_dim)



#run training loop

epochs = 50

optimizer = optim.Adam(rnn_mdl.parameters(), lr=0.01)

for epoch in range(epochs):
      optimizer.zero_grad()
      rnn_output_logits = rnn_mdl(example_input);
      loss = ctc(rnn_output_logits, example_target, example_target_lengths, blank_index)
      loss.backward()
      print("Epoch",epoch + 1," -- ","Loss: ",loss.item())
      optimizer.step()
