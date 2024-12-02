TRAIN?=train.py
INFER?=inference.py
TARGET?=ljk-dao-153.ljk
TARGET_DIR?="~/Documents/LLM/cvx_py_finetune"
TARGET_USER?=azizianw
SOCKET?=./socket.sock

.PHONY:init_ssh
init_ssh:
	eval $(ssh-agent) && ssh-add
	ssh -M -S $(SOCKET) -o ControlPersist=600m $(TARGET_USER)@$(TARGET) exit

.PHONY:deploy
deploy:
	rsync -Pavu -e "ssh -S $(SOCKET)"  --include='*.py' --include='configs/*.yaml' --include="*.sh" --include="*/" --include=".project-root" --include="dataset_hf_datasets_python_cvxpy/*" --prune-empty-dirs --exclude='*' . $(TARGET_USER)@$(TARGET):$(TARGET_DIR)

.PHONY:connect
connect:
	ssh -S $(SOCKET) -t $(TARGET_USER)@$(TARGET) "cd $(TARGET_DIR) && bash --login"

.PHONY:train
train:deploy
	ssh -S $(SOCKET) $(TARGET_USER)@$(TARGET) "cd $(TARGET_DIR) && python3 $(TRAIN)"

.PHONY:infer
infer:deploy
	ssh -S $(SOCKET) $(TARGET_USER)@$(TARGET) "cd $(TARGET_DIR) && python3 $(INFER)"

.PHONY:kill
kill:
	ssh -S $(SOCKET) $(TARGET_USER)@$(TARGET) "pkill -f $(TRAIN)"

.PHONY:clear_logs
clear_logs:
	trash ./logs/*
	ssh -S $(SOCKET) $(TARGET_USER)@$(TARGET) "rm -rf $(TARGET_DIR)/logs/*"

.PHONY:fetch_logs
fetch_logs:
	rsync -Pavu -e "ssh -S $(SOCKET)" $(TARGET_USER)@$(TARGET):$(TARGET_DIR)/logs/* ./logs/

format:
	pre-commit run -a
