function plotCsv(filename)

data = csvread(filename);
figure;
hold on;
plot(data(:,1));
plot(data(:,2));
grid on

end