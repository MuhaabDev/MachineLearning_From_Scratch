class LinearRegression:
    
    def __init__(self , weight = 0 , bias = 0 , alpha = 0.1 , iteration = 100):
        self.weight = weight
        self.bias = bias
        self.alpha = alpha
        self.iteration = iteration
    
    
    
    def train(self , x , y ):
        m = x.shape[0]
        
        for i in range(self.iteration):
            sum_w = 0
            sum_b = 0  
                   
            cost_function = self.compute_cost_function( x , y , m)
            print(f"At Iteration : {i} \t Cost Function : {cost_function}")
            
            for j in range(m):
                tmp_w = ((self.weight * x[j] + self.bias) - y[j])*(x[j])
                tmp_b = (self.weight * x[j] + self.bias) - y[j]
                sum_w += tmp_w
                sum_b += tmp_b
                
            self.weight = self.weight - (self.alpha * (1/m) *sum_w)
            self.bias = self.bias - (self.alpha * (1/m) * sum_b)

        return self.weight , self.bias
    
    
    
    
    def compute_cost_function(self , x , y , m):
        sum_error = 0
        for i in range(m):
            y_pred = self.weight * x[i] + self.bias
            error = pow( y_pred - y[i] , 2 )
            sum_error += error
        J_w_b = (1/(2*m))*(sum_error)
        return J_w_b
      
      
      
            
    def predict(self , input):
        prediction = self.weight * input + self.bias
        print(f"Input : {input} \t Predicted Output :{prediction}")
     
     
        
    def draw_line(self):
        y1 = (self.weight * 0) + self.bias
        y2 = (self.weight * 1) + self.bias
        y3 = (self.weight * 2) + self.bias
        return [y1,y2,y3]
        
