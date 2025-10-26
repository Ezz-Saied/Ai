No.	        Optimizer	                                                One-Line Explanation

1	SGD (Stochastic Gradient Descent)	        Updates model parameters for each training sample to minimize loss.
2	Momentum	                                Adds a fraction of the previous update to the current one for faster convergence.
3	Nesterov Accelerated Gradient (NAG)	        Improves momentum by looking ahead before updating parameters.
4	Adagrad	                                        Adapts learning rate for each parameter based on historical gradients.
5	RMSProp	                                        Uses a moving average of squared gradients to adjust learning rate
6	Adam (Adaptive Moment Estimation)	        Combines momentum and RMSProp for adaptive, efficient updates.
7	AdaDelta	                                An improvement of Adagrad that limits the accumulated gradient to avoid decay.
8	AdamW	                                        Adam optimizer with decoupled weight decay for better regularization.
9	Nadam	                                        Adam optimizer with Nesterov momentum for faster convergence.
10	FTRL (Follow The Regularized Leader)	        Optimizer used for sparse data; balances learning speed and regularization.
