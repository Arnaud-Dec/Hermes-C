.PHONY: all clean data train full

CC := gcc
CFLAGS := -Wall -Wextra -I src/c
SRC := src/c/main.c
TARGET ?= hermes
PYTHON ?= .venv/bin/python

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "[MAKE] Compiling C Engine..."
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	@echo "[MAKE] Delete hermes"
	rm -f $(TARGET)

data:
	@echo "[MAKE] Downloading Data..."
	$(PYTHON) src/python/get_data.py

train:
	@echo "[MAKE] Training Model..."
	$(PYTHON) src/python/train_model.py

full: data train all
	@echo "[MAKE] Launching Hermes..."
	./$(TARGET)
