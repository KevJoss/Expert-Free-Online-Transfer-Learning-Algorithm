import math
import random
import numpy as np
from utils import transpose, scale


def estimate_surprise(target_agent,batch):
    max_batch_size=32
    idxs= [x for x in range(0, len(batch), max_batch_size)]
    losses=[]
    for idx in idxs[:-1]:
        losses= np.hstack((losses,target_agent.get_loss(batch[0][idx:idx+max_batch_size],batch[1][idx:idx+max_batch_size],
                    batch[2][idx:idx+max_batch_size],batch[3][idx:idx+max_batch_size],
                    batch[4][idx:idx+max_batch_size])))
    idx=idxs[-1]
    losses= np.hstack( ( losses, target_agent.get_loss(batch[0][idx:],batch[1][idx:],
                    batch[2][idx:],batch[3][idx:],
                    batch[4][idx:])))
    return losses


def estimate_uncertainty(target_estimator, batch):
    """
    Estimates the epistemic uncertainty that the target agent (student) has
    regarding a batch of transitions collected by the source agent (teacher).

    Splits the full batch into smaller chunks (max_batch_size = 32) to perform
    batched inference on the RND (Random Network Distillation) network without
    exceeding memory limits.

    Args:
        target_estimator (SARSrnd): Uncertainty estimator instance associated
            with the target agent.
        batch (np.ndarray): Array of concatenated [s, a, r, s'] transitions
            extracted from the transfer buffer.

    Returns:
        np.ndarray: 1D array containing the computed uncertainty values for
        each transition in the batch.
    """
    max_batch_size=32
    idxs= [x for x in range(0, len(batch), max_batch_size)]
    uncertainties=[]
    for idx in idxs[:-1]:
        uncertainties= np.hstack((uncertainties,target_estimator._get_multi_uncertainty(batch[idx:idx+max_batch_size]).detach().cpu().numpy()))
    idx=idxs[-1]
    uncertainties= np.hstack( ( uncertainties, target_estimator._get_multi_uncertainty(batch[idx:]).detach().cpu().numpy()))
    return uncertainties
    






def select_teacher(agents, mode, transfer_buffers, ep_start_sharing, evaluation_range):
    """
    Dynamically selects which concurrent agent will act as the teacher 
    (knowledge source) during the transfer event.

    Supported selection modes ('mode'):
        - 'best_performance': Selects the agent with the highest recent average 
          reward over the last 'evaluation_range' episodes.
        - 'avg_uncertainty': Selects the agent with the LOWEST average uncertainty 
          in its transfer buffer, identifying it as the agent with the highest 
          certainty and best consolidated model over its explored environment.

    Args:
        agents (list): List of concurrently training RL agents.
        mode (str): Source selection strategy ('best_performance' or 'avg_uncertainty').
        transfer_buffers (list): List of transfer buffer instances (one per agent) 
            containing recorded transitions and computed uncertainties.
        ep_start_sharing (int): Minimum episode threshold required before enabling 
            knowledge transfer between agents.
        evaluation_range (int): Number of recent episodes to consider when computing 
            mean reward for 'best_performance' mode.

    Returns:
        agent or None: The selected teacher agent, or None if the warm-up threshold 
        'ep_start_sharing' has not yet been reached.
    """
    if(mode=="best_performance"):
        maxR= -math.inf
        teacher=None
        for p in agents:
            if(p.episode < ep_start_sharing):
                return None #to soon to share
            r=np.mean(p.avg_reward[-evaluation_range:])
            if(maxR<r):
                maxR=r
                teacher=p
    elif(mode=="avg_uncertainty"):
        for p in agents:
            if(p.episode < ep_start_sharing):
                return None  # too soon to share
        avg_uncertainty=[]
        for i in range(len(agents)):
            avg_uncertainty.append(transfer_buffers[i].get_avg_uncertainty())

        print("  [UNCERTAINTY] " + " | ".join(
            f"A{i}={u:.6f}" for i, u in enumerate(avg_uncertainty)))
        print(f"  [UNCERTAINTY] min=A{avg_uncertainty.index(min(avg_uncertainty))}")
        
        teacher= agents[avg_uncertainty.index(min(avg_uncertainty))]
    else:
        raise BaseException("Invalid Source Selection Method")
    return teacher



def random_transfer(sourceBuffer, target_agent,B):
    try:
        interactions_to_transfer= random.sample(sourceBuffer,B)
        target_agent.learn_from_samples(interactions_to_transfer)     
    except:
        print("No transfer. - Not enough samples")
    return len(sourceBuffer)

def filter_delta_confidence(sourceBuffer, target_estimator):
    batch= np.array([np.hstack(x) for x in transpose(sourceBuffer.get_sars())])
    target_conf = estimate_uncertainty(target_estimator,batch)
    deltaconfs=np.array(sourceBuffer.get_all_uncertainties())- target_conf
    return deltaconfs, target_conf
      
     
    
#random transfer from those that, afte filtering as random_transfer_from_delta_confidence eprforms a further filterin over he median.
def random_transfer_delta_confidence(sourceBuffer, target_agent, target_estimator, B,threshold="median"):
    deltaconfs,_=filter_delta_confidence(sourceBuffer, target_estimator)
    sorted_by_deltaconfs= [x for _, x in sorted(zip(deltaconfs,sourceBuffer), key=lambda pair: pair[0])]
    if(threshold=="median"):
        sorted_by_deltaconfs= sorted_by_deltaconfs[:(1+len(sorted_by_deltaconfs))//2]
    else:
        raise(Exception("not implemented"))        
    if(len(sorted_by_deltaconfs)>=B):
        random_transfer(sorted_by_deltaconfs, target_agent, None, B)
    return len(sorted_by_deltaconfs)
        
"""
Transfer the value with higher delta confidence.
"""
def transfer_higher_delta_confidence(sourceBuffer, target_agent, target_estimator, B):
    deltaconfs,_=filter_delta_confidence(sourceBuffer, target_estimator)
    sorted_by_deltaconfs= [x for _, x in sorted(zip(deltaconfs,sourceBuffer), key=lambda pair: pair[0])]
    if(len(sorted_by_deltaconfs)>=B):
        target_agent.learn_from_samples(sorted_by_deltaconfs[:B])
    return len(sorted_by_deltaconfs)
        





def transfer_by_higher_loss_and_confidence_mixed(sourceBuffer, target_agent, target_estimator,B):
    deltaconfs,_=filter_delta_confidence(sourceBuffer, target_estimator)
    deltaconfs,filtered_by_conf= transpose([ [_, x] for _, x in filter(lambda pair: pair[0]<0, zip(deltaconfs,sourceBuffer) ) ])

    states, actions, reward, states1, done,source_uncertainties=transpose(filtered_by_conf)
    target_surprises= estimate_surprise(target_agent, [states,actions,reward,states1,done])
    scaled_target_surprises=scale(target_surprises)
    scaled_deltaconfs = scale(deltaconfs)

    sort_by_loss_and_confidence= [x for x,_,_ in sorted(zip(filtered_by_conf,scaled_target_surprises,scaled_deltaconfs), key=lambda x: x[1]+x[2], reverse=True)]
    #each weighs half
    if(len(sort_by_loss_and_confidence)>=B):
        target_agent.learn_from_samples(sort_by_loss_and_confidence[:B])
    return len(sort_by_loss_and_confidence)


