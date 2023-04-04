
## Forward equation

- $net_{H_t} = X_t*W_x + out_{H_{t-1}}*W_h + B_x$
- $out_{H_t} = tanh(net_{H_t})$
- $\hat Y_t = out_{H_t}*W_y + B_y$
- $Loss_t = 0.5*(\hat {Y_t}-Y_t)^2$

## Gradients (BPTT)
1. Wy
```
aLt      aLt        a(hat_Yt)
-----  = -------  x -------- = (hat_Yt-Yt)  * out_Ht
a(Wy)    a(hat_Yt)  a(Wy)

aLt      aLt        a(hat_Yt)
-----  = -------  x --------  = (hat_Yt-Yt)  * 1
a(By)    a(hat_Yt)  a(By)
```

2. Wx, Bx (TimeWindow: 3)
```
aLt     aLt         a(hat_Yt)   a(out_Ht)   a(net_Ht)      aLt         a(hat_Yt)   a(out_Ht)   a(net_Ht)     a(out_Ht-1)    a(net_Ht-1)
----- = --------- x --------- x --------- x ---------   +  -------   x --------  x --------- x ----------- x ------------ x -----------
a(Wx)   a(hat_Yt)   a(out_Ht)   a(net_Ht)   a(Wx)          a(hat_Yt)   a(out_Ht)   a(net_Ht)   a(out_Ht-1)   a(net_Ht-1)    a(Wx)

          aLt         a(hat_Yt)   a(out_Ht)   a(net_Ht)     a(out_Ht-1)    a(net_Ht-1)   a(out_Ht-2)   a(net_Ht-2)
        + -------   x --------  x --------- x ----------- x ------------ x ----------- x ----------- x ----------- 
          a(hat_Yt)   a(out_Ht)   a(net_Ht)   a(out_Ht-1)   a(net_Ht-1)    a(out_Ht-2)   a(net_Ht-2)   a(Wx)

      = (hat_Yt-Yt) * Wy * (1-out_Ht)(1+out_Ht) * X_t + (hat_Yt-Yt) * Wy * (1-out_Ht)(1+out_Ht) * Wh * (1-out_Ht-1)(1+out_Ht-1) * X_t-1
      
        + (hat_Yt-Yt) * Wy * (1-out_Ht)(1+out_Ht) * Wh * (1-out_Ht-1)(1+out_Ht-1) * Wh * (1-out_Ht-2)(1+out_Ht-2) * X_t-2

aLt     aLt         a(hat_Yt)   a(out_Ht)   a(net_Ht)
----- = --------- x --------- x --------- x --------- + ...
a(Bx)   a(hat_Yt)   a(out_Ht)   a(net_Ht)   a(Bx) 

      = (hat_Yt-Yt) * Wy * (1-out_Ht)(1+out_Ht) * 1 + (hat_Yt-Yt) * Wy * (1-out_Ht)(1+out_Ht) * Wh * (1-out_Ht-1)(1+out_Ht-1) * 1 + (hat_Yt-Yt) * Wy * (1-out_Ht)(1+out_Ht) * Wh * (1-out_Ht-1)(1+out_Ht-1) * Wh * (1-out_Ht-2)(1+out_Ht-2) * 1

```

3. Wh (TimeWindow: 3)
```
aLt     aLt         a(hat_Yt)    a(out_Ht)   a(net_Ht)      aLt         a(hat_Yt)    a(out_Ht)   a(net_Ht)     a(out_Ht-1)   a(net_Ht-1)
----- = --------- x ---------- x --------- x ---------  +   --------- x ---------- x --------- x ----------- x ----------  x -----------
a(Wh)   a(hat_Yt)   a(out_Ht)    a(net_Ht)   a(Wh)          a(hat_Yt)   a(out_Ht)    a(net_Ht)   a(out_Ht-1)   a(net_Ht-1)   a(Wh)

            aLt         a(hat_Yt)    a(out_Ht)   a(net_Ht)     a(out_Ht-1)   a(net_Ht-1)   a(out_Ht-2)   a(net_Ht-2)
        +   --------- x ---------- x --------- x ----------- x ----------  x ----------- x ----------- x -----------
            a(hat_Yt)   a(out_Ht)    a(net_Ht)   a(out_Ht-1)   a(net_Ht-1)   a(out_Ht-2)   a(net_Ht-2)   a(Wh)

      = (hat_Yt-Yt) * Wy * (1-out_Ht)(1+out_Ht) * out_Ht-1 +  (hat_Yt-Yt) * Wy * (1-out_Ht)(1+out_Ht) * Wh * (1-out_Ht-1)(1+out_Ht-1) * out_Ht-2
        
        +  (hat_Yt-Yt) * Wy * (1-out_Ht)(1+out_Ht) * Wh * (1-out_Ht-1)(1+out_Ht-1) * Wh * (1-out_Ht-2)(1+out_Ht-2) * out_Ht-3
```

## Graph Strategy 
- Save 10 heap (Queue)
    - [@forward_step] out_Ht, out_Ht-1, out_Ht-2, out_Ht-3, out_Ht-4, out_Ht-5, out_Ht-6, out_Ht-7, out_Ht-8, out_Ht-9
    - X_t, Xt-1, Xt-2, Xt-3, Xt-4, Xt-5, Xt-6, Xt-7, Xt-8, Xt-9


