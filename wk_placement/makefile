SCR_PATH := ./src
MY_INCLUDE_PATH := ./include
NNIE_INCLUDE_PATH := ./multi-core/include
NNIE_LIB_PATH := ./multi-core/lib
LIBRARIES := -lmpi -lnnie -lsecurec -lVoiceEngine -lupvqe -ldnvqe -lpthread
#-lmali
CROSS := aarch64-himix100-linux-
CXX := $(CROSS)g++
CC := gcc
CFLAGES := -Wall -I $(MY_INCLUDE_PATH) -I $(NNIE_INCLUDE_PATH) -c 

OBJS := auxiliary.o basic_function.o DeepModel.o wk_run.o

wk_run: $(OBJS)
	$(CXX) -o $@ $^ -L $(NNIE_LIB_PATH) $(LIBRARIES)

auxiliary.o: 	$(MY_INCLUDE_PATH)/auxiliary.h \
				$(SCR_PATH)/auxiliary.cpp
	$(CXX) $(CFLAGES) $(SCR_PATH)/auxiliary.cpp

basic_function.o: 	$(MY_INCLUDE_PATH)/basic_function.h \
					$(MY_INCLUDE_PATH)/resource_manager.hpp \
					$(MY_INCLUDE_PATH)/auxiliary.h \
					$(SCR_PATH)/basic_function.cpp
	$(CXX) $(CFLAGES) $(SCR_PATH)/basic_function.cpp

DeepModel.o: 	$(MY_INCLUDE_PATH)/auxiliary.h \
				$(MY_INCLUDE_PATH)/DeepModel.hpp \
				$(MY_INCLUDE_PATH)/basic_function.h \
				$(MY_INCLUDE_PATH)/resource_manager.hpp \
				$(SCR_PATH)/DeepModel.cpp
	$(CXX) $(CFLAGES) $(SCR_PATH)/DeepModel.cpp

wk_run.o:		$(MY_INCLUDE_PATH)/DeepModel.hpp \
				$(MY_INCLUDE_PATH)/basic_function.h \
				$(MY_INCLUDE_PATH)/resource_manager.hpp \
				$(SCR_PATH)/wk_run.cpp
	$(CXX) $(CFLAGES) $(SCR_PATH)/wk_run.cpp

.PHONY: clean
clean:
	@-rm wk_run *.o
