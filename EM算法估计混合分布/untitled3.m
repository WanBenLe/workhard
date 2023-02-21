data = [1.04,
1.16,
1.28,
1.40,
1.51,
1.63,
1.75,
1.87,
1.99,
2.10,
2.22,
2.34]

Mdl = arima(1, 1, 1);
EstMdl = estimate(Mdl,data);
[forData,YMSE] = forecast(EstMdl,29,'Y0',data);  
lower = forData - 1.96*sqrt(YMSE); %95置信区间下限
upper = forData + 1.96*sqrt(YMSE); %95置信区间上限
