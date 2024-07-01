data <- read.csv("dataset.csv")
data

data$person <- as.factor(data$person)
data$Article <- as.factor(data$Article)
data$accuracy <- as.factor(data$accuracy)

# Create dummy variables for each class
data$accuracy_0 <- ifelse(data$accuracy == 0, 0, 1)
data$accuracy_1 <- ifelse(data$accuracy == 1, 1, 0)

model.1.1 <- glm(data$accuracy_0 ~ data$LLM_dummy + data$RB_dummy + data$person + data$Article + data$time.spent.total, family = binomial)
summary(model.1.1)

model.1.2 <- glm(data$accuracy_1 ~ data$LLM_dummy + data$RB_dummy + data$person + data$Article + data$time.spent.total, family = binomial)
summary(model.1.2)

model.2 <- lm(data$time.spent.total ~ data$LLM_dummy + data$RB_dummy + data$person + data$Article + data$accuracy)
summary(model.2)

