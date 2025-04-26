package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompareIgnCase_3_0_Test {

    @Test
    public void testStringCompareIgnCase() {
        String lhs = "Hello";
        String rhs = "hello";
        // Test case 1: Normal case
        assertEquals(0, Util.StringCompareIgnCase(lhs, rhs));
        // Test case 2: Upper case
        assertEquals(0, Util.StringCompareIgnCase("Hello", rhs));
        // Test case 3: Lower case
        assertEquals(0, Util.StringCompareIgnCase(rhs, lhs));
        // Test case 4: Title case
        assertEquals(0, Util.StringCompareIgnCase("hello", lhs));
        // Test case 5: Mixed case
        assertEquals(0, Util.StringCompareIgnCase("HelloWorld", rhs));
        // Test case 6: Title case with spaces
        assertEquals(0, Util.StringCompareIgnCase("Hello World", rhs));
        // Test case 7: Title case with special characters
        assertEquals(0, Util.StringCompareIgnCase("Hello! World", rhs));
        // Test case 8: Lower case with spaces
        assertEquals(0, Util.StringCompareIgnCase("hello", "Hello World"));
        // Test case 9: Lower case with special characters
        assertEquals(0, Util.StringCompareIgnCase("hello!", "Hello World"));
        // Test case 10: Title case with special characters
        assertThrows(NullPointerException.class, () -> Util.StringCompareIgnCase(null, rhs));
        // Test case 11: Null input
        assertThrows(NullPointerException.class, () -> Util.StringCompareIgnCase("Hello", null));
    }
}
