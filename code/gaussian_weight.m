function y = gaussian_weight(X,mu,sigma)
y = exp((-1/(2*sigma^2))*(X-mu)^2);
