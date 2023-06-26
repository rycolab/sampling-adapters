import torch
from collections import namedtuple

Metrics = namedtuple('Metrics', ['tvd', 'forward_kl', 'backward_kl', 'eps_forward_kl', 'eps_backward_kl',
                      'js', 'forward_js', 'backward_js', 'token_counts', 'entropy', 'reference_entropy', 'cross_entropy',
                      'inf_counts', 'perplexity', 'eps_perplexity'], defaults=[None, None, None])


def jensen_shannon(log_p, log_q):
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    log_m = torch.log(0.5*(p+q))
    kl1 = torch.nansum(p*log_p, dim=1) - torch.nansum(p*log_m, dim=1)
    kl2 = torch.nansum(q*log_q, dim=1) - torch.nansum(q*log_m, dim=1)                     
    js = 0.5 *(kl1 + kl2)
    return js, (kl1, kl2)

def tvd(p, q):
    return torch.sum(torch.abs(p-q), axis=1)

def entropy(log_p):
    return -(torch.exp(log_p)*log_p).nansum(dim=1)

def cross_entropy(log_p, log_q):
    return -(torch.exp(log_p)*log_q).sum(dim=1)
                                         
def epsilon_perplexity(log_q, reference, epsilon=1e-6):
    q_eps = (torch.nn.functional.softmax(log_q.view(-1, log_q.size(-1)), dim=1) + epsilon)/(1+epsilon*log_q.shape[-1])
    neg_eps_log_probs = torch.nn.functional.cross_entropy(torch.log(q_eps.view(-1, q_eps.size(-1))), reference.view(-1), reduction='none')
    return torch.exp(neg_eps_log_probs.mean())  
                       
def perplexity(log_q, reference):
    neg_log_probs = torch.nn.functional.cross_entropy(log_q.view(-1, log_q.size(-1)), reference.view(-1), reduction='none')
    return torch.exp(neg_log_probs.mean())

def avg_nll(log_q, reference):
    neg_log_likelihood = torch.nn.functional.cross_entropy(log_q.view(-1, log_q.size(-1)), reference.view(-1), reduction='none')
    return sum(neg_log_likelihood)/len(neg_log_likelihood)

def inf_counts(log_q, reference):
    log_probs = torch.nn.functional.cross_entropy(log_q.view(-1, log_q.size(-1)), reference.view(-1), reduction='none')
    return sum(torch.isinf(log_probs))

def forward_kl(log_p, log_q):
    return cross_entropy(log_p, log_q) - entropy(log_p)

def eps_forward_kl(log_p, log_q, epsilon=1e-6):
    q_eps = (torch.nn.functional.softmax(log_q.view(-1, log_q.size(-1)), dim=1) + epsilon)/(1+epsilon*log_q.shape[-1])
    return cross_entropy(log_p, torch.log(q_eps)) - entropy(log_p)

def backward_kl(log_p, log_q):
    return forward_kl(log_q, log_p)

def eps_backward_kl(log_p, log_q, epsilon=1e-8):
    return eps_forward_kl(log_q, log_p, epsilon)
