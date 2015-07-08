#CODE VARIABLES
CXX=g++
CFLAGS=-c -Wall -std=c++0x -g3 -Ofast -msse2  -I.
LDFLAGS=-L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_calib3d  -lopencv_flann -lopencv_objdetect \
	-lboost_filesystem -lboost_system -lboost_program_options -lopencv_ml -lboost_serialization -lboost_iostreams 
SOURCES=${wildcard *.cpp model/*.cpp}
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=deepface

all:  $(SOURCES) $(EXECUTABLE)


clean:
	rm -f *.o
	rm -f $(EXECUTABLE)
	find ./ -name \*.o  -delete

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CXX) $(CFLAGS) $< -o $@