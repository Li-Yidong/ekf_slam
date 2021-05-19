close all;
clear all;

angle_l = pi/10;%ランドマークを配置する円上の間隔[rad]
r = 8.5;%ランドマークを配置する円の半径
global lm_num;
lm_num = 20; %ランドマークの個数

time = 0;
endtime = 40; % シミュレーション終了時間[sec]
global dt;
dt = 0.1; % シミュレーション刻み時間[sec]
nSteps = ceil((endtime - time)/dt);%シミュレーションのステップ数


MaxRange = 5; %センサの観測最大距離

toRadian =@(degree)degree/180*pi;% degree to radian

% 観測ノイズの共分散行列　(12)式
global R;
R=diag([0.1 toRadian(1)]).^2;%range[m], Angle[rad]

global J;
J = [0 -1;1 0];%90度の回転行列

%制御入力
w = pi/20;                      %転向速度 9deg/s
v = [1;0]; % ロボットの前進速度 （ロボット座標系で v_x = 1m/s, v_y = 0）

delta = 0.02;
w_w = delta*2*sqrt(0.1);%角度誤差
w_v = [delta*sqrt(0.1)/2;delta*sqrt(0.1)/2];%速度誤差
w_Q = [toRadian(w_w);w_v];% プロセスノイズの共分散行列　(10)式
Q = diag([toRadian(w_w);w_v].^2);
global R_t;
R_t = @(theta) [cos(theta) -sin(theta);sin(theta) cos(theta)];%回転行列
f = @(theta,x,wdt,vdt)[ theta + wdt; x + R_t(theta)*vdt];%ロボット運動学モデル
global h_tilde
h_tilde = @(x) [sqrt(x'*x); atan2(x(2),x(1))];%ロボット局所座標xでの観測値(距離と角度） Remark2の式
Dh = @(x) [x(1)/sqrt(x'*x) x(2)/sqrt(x'*x); -x(2)/(x'*x) x(2)/(x'*x)];%h_tildeのヤコビ行列 (13)式

x = zeros(2,nSteps+1);%状態変数(x,y)
x(:,1) = [0 0]';%状態変数(x,y)の初期値
theta = zeros(1,nSteps+1);%状態変数 θ
theta(1) = 0;%状態変数 θの初期値
Y = zeros(2*lm_num, nSteps+1);%ランドマークの観測データ
X_hat = zeros(3+2*lm_num,1);

%ランドマークの座標を定義
p = zeros(lm_num,2);%ランドマーク座標変数
th = 0;
for k = 1:lm_num
    p(k,1) = r*sin(th);
    p(k,2) = r-r*cos(th)-2;
    th = angle_l+th;
end

%ロボットの走行と観測シミュレーション　　ーーーーーーーーーーーーーーーーーーーーーーーー
Y(:,1) = observation(x(:,1), theta(1), p, MaxRange);
figure
show_state(x(:,1), theta(1), p, Y(:,1));
for k = 2:nSteps+1
    noise = w_Q(1:3).*randn(3,1);
    X = f(theta(k-1),x(:,k-1),w*dt + noise(1),v*dt + noise(2:3)); %(9)式
    theta(k) = X(1);
    x(:,k) = X(2:3);
    Y(:,k) = observation(x(:,k), theta(k), p, MaxRange);
    show_state(x(:,k), theta(k), p, Y(:,k));
end

%推定ベクトルの定義 
x_hat = zeros(2,nSteps);
theta_hat = zeros(1,nSteps);
x_hat(:,1) = [0;0];
theta_hat(1) = 0;
P = zeros(3+2*lm_num,3+2*lm_num);%誤差共分散行列の初期値
id = [];
lm_seq=[];
% EKF-SLAM　ーーーーーーーーーーーーーーーーーーーーーーーー
figure
for k=2:nSteps
    %Propagation
    X_hat(1:3) = f(theta_hat(k-1),x_hat(:,k-1),w*dt,v*dt);%(9)式、ランドマークの位置をロボットの状態ベクトルに追加
    theta_hat(k) = X_hat(1);
    x_hat(:,k) = X_hat(2:3);
    [F_e,G_e,P] = jacob_FGQ(theta_hat(k-1),X_hat,w*dt,v*dt,Q,P);%式13
    id = X_hat(4:end);
    %Update
    z=[];
    H=[];
    Rn=[];
    lm_id=[];
    is_exist=[];
    for i=1:lm_num
        if Y(2*i-1,k) < Inf %見えているランドマークについてのみベクトルz 、ヤコビ行列H、共分散行列Rnを構成する
            is_exist=[is_exist;i];
            z=[z; Y((2*i-1):2*i,k) - h_tilde(R_t(theta_hat(k))'*(X_hat(3+i*2-1:3+i*2) - x_hat(:,k)))];
            jacob_h=Dh(R_t(theta_hat(k))'*(X_hat(3+i*2-1:3+i*2) - x_hat(:,k)));%(13)式の∇h
            H=measurmentH(theta_hat,p,x_hat,jacob_h,i,k,H,X_hat);
            if length(Rn)==0
                Rn = R;
            else
                Rn = blkdiag(Rn, R);
            end
        end
    end
    S = H*P*H' + Rn;
    K = P*H'/S;
    P = (eye(size(K*H))-K*H)*P;
    X_hat = X_hat + K*z;
    theta_hat(k) = X_hat(1);
    x_hat(:,k) = X_hat(2:3);
    for i=1:length(is_exist)
        was_exist = ~(lm_seq - is_exist(i));
        idy = find(was_exist);
        if isempty(idy)
            lm_seq = [lm_seq; is_exist(i)];
            x_L = x_hat(:,k)+R_t(theta_hat(k))*Y((2*is_exist(i)-1):2*is_exist(i),k);
            X_hat(2*is_exist(i)+2:2*is_exist(i)+3) = x_L;
            HR = - R_t(theta_hat(k))'*[eye(2) J*R_t(theta_hat(k))*(p(is_exist(i),:)'-x_hat(:,k))];
            HL = R_t(theta_hat(k))';
            rng = 2*is_exist(i)+2:2*is_exist(i)+3;
            P(rng,rng)= inv(HL)*HR*P(1:3,1:3)*HR'*inv(HL)' + inv(HL)*R*inv(HL)'; % 対応しているランドマークの共分散行列（シグマMnMn）
            P(rng,1:3)= -inv(HL)*HR*P(1:3,1:3); % 対応しているランドマークのシグマMnx
            P(1:3,rng)= P(rng,1:3)';%対応しているランドマークのシグマxMn
            if length(P)>3
                rnm= 4:2*lm_num;
                P(rng,rnm)= -inv(HL)*HR*P(1:3,rnm);%シグマM1Mn
                P(rnm,rng)= P(rng,rnm)';%シグマMnM1
            end
            P = 0.5*(P+P');
        end
    end
    show_state_estimated(x(:,k), theta(k), p, Y(:,k), x_hat(:,k), theta_hat(k),id);
end

figure
ph=plot(p(:,1),p(:,2), 'bp');
ph.DisplayName = 'landmark';
hold on
ph=plot(x(1,:),x(2,:), 'ko');
ph.DisplayName = 'robot location';
ph=plot(x_hat(1,:),x_hat(2,:), 'g-o');
ph.DisplayName = 'estimated';
xlim([-10 +10])
ylim([-4 +16])
grid on;
axis equal;
legend('show')




function [F_e,G_e,P]= jacob_FGQ(theta_hat,X_hat,wdt,vdt,Q,P) % ヤコビ行列 (13)式のFn, Gn
global J;
global R_t;
G = [1 0 0;zeros(2,1) R_t(theta_hat)];%Gの初期化
                                        %[1 0 0
                                        % 0 R(theta_hat)
                                        % 0 0 0]

F = [1 0 0; R_t(theta_hat)*J*vdt eye(2)];%Fの初期化
                                           %[1 0 0
                                           % R(theta_hat)*J*vdt 1 0 
                                           %   0 1]
Qprime = blkdiag(Q,zeros(size(X_hat,1)-3))*blkdiag(Q,zeros(size(X_hat,1)-3))';
F_e = blkdiag(F,eye(size(X_hat,1)-3));
G_e = blkdiag(G,zeros(size(X_hat,1)-3));
P = F_e*P*F_e'+G_e*Qprime*G_e';
end


function y=observation(x, theta, p, MaxRange)
% 観測方程式
% 状態(x,theta)のロボットからランドマーク集合 p の観測したデータベクトル y をつくる（センサの最大検知距離:MaxRange）
global lm_num
global R
global h_tilde
global R_t
y = zeros(2*lm_num,1);
for k=1:lm_num
    h=h_tilde(R_t(theta)'*(p(k,:)'- x));%　(11)式
    if h(1) <= MaxRange
        y(2*k-1:2*k) = h + R*randn(2,1);%　(11)式
    else
        y(2*k-1:2*k) = [Inf Inf]';
    end
end
end

function show_state(x, theta, p, Y) %走行シミュレーションの描画
global lm_num
ph=plot(p(:,1),p(:,2), 'bp');
xlim([-10 +10])
ylim([-4 +16])
pbaspect([1 1 1])
hold on
ph=plot(x(1),x(2), 'ko');
ph=plot([x(1) x(1)+cos(theta)],[x(2) x(2)+sin(theta)], 'k-');
for i=1:lm_num
    if Y(2*i-1)<Inf
        plot([x(1) x(1)+Y(2*i-1)*cos(Y(2*i)+theta)],[x(2) x(2)+Y(2*i-1)*sin(Y(2*i)+theta)],'r-')
    end
end
grid on;
hold off
drawnow
end

function show_state_estimated(x, theta, p, Y, x_hat, theta_hat,id)%EKF-SLAM シミュレーションの描画
global lm_num
ph=plot(p(:,1),p(:,2), 'bp');
xlim([-10 +10])
ylim([-4 +16])
pbaspect([1 1 1])
hold on
for i=1:2:size(id,1)
    ph=plot(id(i),id(i+1),'rp');
end
ph=plot(x(1),x(2), 'ko');
ph=plot([x(1) x(1)+cos(theta)],[x(2) x(2)+sin(theta)], 'k-');
for i=1:lm_num
    if Y(2*i-1)<Inf
        plot([x(1) x(1)+Y(2*i-1)*cos(Y(2*i)+theta)],[x(2) x(2)+Y(2*i-1)*sin(Y(2*i)+theta)],'r-')
    end
end
ph=plot(x_hat(1),x_hat(2), 'go');
ph=plot([x_hat(1) x_hat(1)+cos(theta_hat)],[x_hat(2) x_hat(2)+sin(theta_hat)], 'g-');
grid on;
hold off
drawnow
end

function [H]=measurmentH(theta_hat,p,x_hat,jacob_h,i,k,H,X_hat)
global J;
global R_t;
Xsize = size(X_hat,1);
H_k = zeros(2,Xsize);
H_k(:,1:3) = [-J*R_t(theta_hat(k))'*(X_hat(3+i*2-1:3+i*2)-x_hat(:,k)) -R_t(theta_hat(k))'];
H_k(:,i*2+2:i*2+3) = R_t(theta_hat(k))';
H = [H; jacob_h*H_k]; %(13)式
end
