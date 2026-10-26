CC = clang
CFLAGS = -fobjc-arc -O2 -I include -I src
FRAMEWORKS = -framework Foundation -framework IOSurface -framework Accelerate \
             -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
VERSION := $(shell sed -n 's/.*ANER_VERSION_STRING "\([0-9.]*\)".*/\1/p' include/aner.h)

PREFIX ?= /usr/local
BINDIR = $(PREFIX)/bin
INCLUDEDIR = $(PREFIX)/include
LIBDIR = $(PREFIX)/lib
PKGCONFIGDIR = $(LIBDIR)/pkgconfig

LIB_OBJS = build/graph.o build/cpu.o build/program.o build/ane.o build/gpu.o build/lut6.o

.PHONY: all test python-test install uninstall clean

all: build/libaner.a build/libaner.dylib build/aner-test build/lut6-test build/modal

build:
	mkdir -p build

build/graph.o: src/graph.c include/aner.h src/internal.h | build
	$(CC) $(CFLAGS) -c src/graph.c -o $@

build/cpu.o: src/cpu.c include/aner.h src/internal.h | build
	$(CC) $(CFLAGS) -c src/cpu.c -o $@

build/program.o: src/program.c include/aner.h src/internal.h | build
	$(CC) $(CFLAGS) -c src/program.c -o $@

build/ane.o: src/ane.m include/aner.h src/internal.h | build
	$(CC) $(CFLAGS) -c src/ane.m -o $@

build/gpu.o: src/gpu.m include/aner.h src/internal.h | build
	$(CC) $(CFLAGS) -c src/gpu.m -o $@

build/lut6.o: src/lut6.c include/aner.h include/aner_lut6.h | build
	$(CC) $(CFLAGS) -c src/lut6.c -o $@

build/libaner.a: $(LIB_OBJS)
	ar rcs $@ $(LIB_OBJS)

build/libaner.0.dylib: $(LIB_OBJS)
	$(CC) -dynamiclib -install_name @rpath/libaner.0.dylib \
		-compatibility_version $(VERSION) -current_version $(VERSION) \
		-o $@ $(LIB_OBJS) $(FRAMEWORKS)

build/libaner.dylib: build/libaner.0.dylib
	ln -sfn libaner.0.dylib $@

build/aner-test: src/test_main.c build/libaner.a
	$(CC) $(CFLAGS) src/test_main.c build/libaner.a $(FRAMEWORKS) -o $@

build/lut6-test: src/lut6_test.c build/libaner.a
	$(CC) $(CFLAGS) src/lut6_test.c build/libaner.a $(FRAMEWORKS) -o $@

build/modal: examples/modal.c build/libaner.a
	$(CC) $(CFLAGS) examples/modal.c build/libaner.a $(FRAMEWORKS) -o $@

test: build/aner-test build/lut6-test build/modal python-test
	./build/aner-test
	./build/lut6-test
	./build/modal

python-test: build/libaner.dylib
	cd .. && PYTHONPATH=python python3 -m unittest discover -s python/tests -v

install: build/libaner.a build/libaner.0.dylib build/libaner.dylib build/modal
	install -d $(DESTDIR)$(BINDIR) $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(LIBDIR) $(DESTDIR)$(PKGCONFIGDIR)
	install -m 644 include/aner.h $(DESTDIR)$(INCLUDEDIR)/aner.h
	install -m 644 include/aner_lut6.h $(DESTDIR)$(INCLUDEDIR)/aner_lut6.h
	install -m 644 build/libaner.a $(DESTDIR)$(LIBDIR)/libaner.a
	install -m 755 build/libaner.0.dylib $(DESTDIR)$(LIBDIR)/libaner.0.dylib
	ln -sfn libaner.0.dylib $(DESTDIR)$(LIBDIR)/libaner.dylib
	sed -e 's|@prefix@|$(PREFIX)|g' -e 's|@version@|$(VERSION)|g' aner.pc.in > $(DESTDIR)$(PKGCONFIGDIR)/aner.pc
	install -m 755 build/modal $(DESTDIR)$(BINDIR)/aner-modal

uninstall:
	rm -f $(DESTDIR)$(INCLUDEDIR)/aner.h $(DESTDIR)$(INCLUDEDIR)/aner_lut6.h
	rm -f $(DESTDIR)$(LIBDIR)/libaner.a $(DESTDIR)$(LIBDIR)/libaner.dylib $(DESTDIR)$(LIBDIR)/libaner.0.dylib
	rm -f $(DESTDIR)$(PKGCONFIGDIR)/aner.pc
	rm -f $(DESTDIR)$(BINDIR)/aner-modal

clean:
	rm -rf build
