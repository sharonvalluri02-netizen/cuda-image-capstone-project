NVCC ?= nvcc
TARGET = image_processing

all: $(TARGET)

$(TARGET): image_processing.cu
	$(NVCC) -O2 -std=c++17 image_processing.cu -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) input output

dataset:
	python3 generate_dataset.py
