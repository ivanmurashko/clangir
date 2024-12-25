// RUN: %clang_cc1 -fclangir -emit-cir -o - %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-LE %s

#include <stdarg.h>

// Obviously there's more than one way to implement va_arg. This test should at
// least prevent unintentional regressions caused by refactoring.

va_list the_list;

int simple_int(void) {
  return va_arg(the_list, int);
// CHECK: [[GR_OFFS:%[a-z_0-9]+]] = load i32, ptr getelementptr inbounds nuw (%struct.__va_list, ptr @the_list, i32 0, i32 3)
// CHECK: ret i32 [[RESULT]]
}

typedef struct {} empty;
empty empty_record_test(void) {
// CHECK: entry
// CHECK-NEXT: ret void
  return va_arg(the_list, empty);
}
