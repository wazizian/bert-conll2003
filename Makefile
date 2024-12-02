TRAIN?=train/train.py
TARGET?=ljk-dao-153.ljk
TARGET_DIR?="~/Documents/LLM/bert_finetune"
TARGET_USER?=azizianw
SOCKET?=./socket.sock

.PHONY:init_ssh
init_ssh:
	rm -f $(SOCKET)
	eval $(ssh-agent) && ssh-add
	ssh -M -S $(SOCKET) -o ControlPersist=600m $(TARGET_USER)@$(TARGET) exit

.PHONY:deploy
deploy:
	rsync -Pavu -e "ssh -S $(SOCKET)"  --include='*.py' --include='*.yaml' --include="*.sh" --include="*/" --include=".project-root" --include="data/*/*" --include="data/*/*/*" --prune-empty-dirs --exclude='*' . $(TARGET_USER)@$(TARGET):$(TARGET_DIR)

.PHONY:connect
connect:
	ssh -S $(SOCKET) -t $(TARGET_USER)@$(TARGET) "mkdir -p $(TARGET_DIR) && cd $(TARGET_DIR) && bash --login"

.PHONY:train
train:deploy
	ssh -S $(SOCKET) $(TARGET_USER)@$(TARGET) "cd $(TARGET_DIR) && python3 $(TRAIN)"

.PHONY:kill
kill:
	ssh -S $(SOCKET) $(TARGET_USER)@$(TARGET) "pkill -f $(TRAIN)"

format:
	pre-commit run -a
