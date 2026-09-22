CC      ?= cc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -Iinclude
LDLIBS  := -lm
SRC     := src/har_resnet.c src/har_resnet_params.c src/har_golden.c test/test_inference.c
BUILD   := build

all: $(BUILD)/test_inference

$(BUILD)/test_inference: $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDLIBS)

$(BUILD):
	mkdir -p $(BUILD)

test: $(BUILD)/test_inference
	./$(BUILD)/test_inference

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
