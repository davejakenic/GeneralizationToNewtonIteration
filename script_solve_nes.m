% script_solve_nes

% script for solving non-linear equation systems

% method k is DJN iteration with k linsolves.

close all;
clear all;
clc;

index   = 2;
maxit   = 10; 

func_f  = @(x) func_residual(x,index);
func_J  = @(x) func_jacobian(x,index);
x0      =      func_iniguess(  index);
xF      =      func_solution(  index);

[X1]    = Iteration1(x0,func_f,func_J,maxit);
[X2]    = Iteration2(x0,func_f,func_J,maxit);
[X3]    = Iteration3(x0,func_f,func_J,maxit);
[X4]    = Iteration4(x0,func_f,func_J,maxit);

err1    = X1-xF*ones(1,maxit);
err2    = X2-xF*ones(1,maxit);
err3    = X3-xF*ones(1,maxit);
err4    = X4-xF*ones(1,maxit);

ne1     = zeros(1,maxit);
ne2     = zeros(1,maxit);
ne3     = zeros(1,maxit);
ne4     = zeros(1,maxit);
for i=1:size(x0,1)
    ne1 = ne1 + err1(i,:).^2;
    ne2 = ne2 + err2(i,:).^2;
    ne3 = ne3 + err3(i,:).^2;
    ne4 = ne4 + err4(i,:).^2;
end
ne1 = ne1.^0.5 + 1e-20;
ne2 = ne2.^0.5 + 1e-20;
ne3 = ne3.^0.5 + 1e-20;
ne4 = ne4.^0.5 + 1e-20;

figure;
hold all;
plot(1:maxit,log10(ne1),'.-');
plot(1:maxit,log10(ne2),'.-');
plot(1:maxit,log10(ne3),'.-');
plot(1:maxit,log10(ne4),'.-');
grid on;
box on;
xlabel("iterations k")
ylabel("lg10|x_k-eF|_2")
legend("method1","method2","method3")


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

function [ X] = Iteration1(x0,func_f,func_J,maxit)
    %
    x   = x0;
    X   = [];
    %
    for rep=1:maxit
        %
        mJ      = -func_J(x);
        %
        v1      = mJ \ func_f(x);
        %
        x       = x + v1;
        %
        X       = [X,x];
    end
end
function [ X] = Iteration2(x0,func_f,func_J,maxit)
    %
    x   = x0;
    X   = [];
    %
    for rep=1:maxit
        %
        mJ      = -func_J(x);
        %
        v1      = mJ \ func_f(x);
        v2      = mJ \ func_f(x+v1);
        %
        x       = x + v1 + v2;
        %
        X       = [X,x];
    end
end
function [ X] = Iteration3(x0,func_f,func_J,maxit)
    %
    x   = x0;
    X   = [];
    %
    for rep=1:maxit
        %
        mJ      = -func_J(x);
        %
        v1      = mJ \ func_f(x);
        v2      = mJ \ func_f(x+v1);
        v3      = mJ \ func_f(x+v1+v2);
        %
        x       = x + v1 + v2 + v3;
        %
        X       = [X,x];
    end
end
function [ X] = Iteration4(x0,func_f,func_J,maxit)
    %
    x   = x0;
    X   = [];
    %
    for rep=1:maxit
        %
        mJ      = -func_J(x);
        %
        v1      = mJ \ func_f(x);
        v2      = mJ \ func_f(x+v1);
        v3      = mJ \ func_f(x+v1+v2);
        v4      = mJ \ func_f(x+v1+v2+v3);
        %
        x       = x + v1 + v2 + v3 + v4;
        %
        X       = [X,x];
    end
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

function [x0] = func_iniguess(  index)
    %
    switch(index)
        case 1
            x0  = 3;
        case 2
            x0  = [3;4];
    end
    %
end
function [xF] = func_solution(  index)
    %
    switch(index)
        case 1
            xF  = 2;
        case 2
            xF  = [3*sqrt(2);3];
    end
    %
end
function [ y] = func_residual(x,index)
    %
    switch(index)
        case 1
            y   = x^2 - 4;
        case 2
            y   = [6*x(1)-2*x(1)*x(2);6*x(2)-x(1)^2];
    end
    %
end
function [Dy] = func_jacobian(x,index)
    n   = size(x,1);
    Dy  = zeros(n,n);
    for i=1:n
        xi      = x;
        xi(i,1) = xi(i) + sqrt(-1) * 1e-20;
        Dy(:,i) = 1e20 * imag( func_residual(xi,index) );
    end
end