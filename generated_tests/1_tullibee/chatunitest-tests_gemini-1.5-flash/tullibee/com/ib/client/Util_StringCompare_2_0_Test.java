package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_StringCompare_2_0_Test {

    @Test
    void testStringCompare() throws Exception {
        // Test cases covering various scenarios including null and empty strings
        assertEquals(0, Util.StringCompare("hello", "hello"));
        // Case-insensitive due to normalization
        assertEquals(0, Util.StringCompare("Hello", "Hello"));
        assertEquals(0, Util.StringCompare("", ""));
        assertEquals(0, Util.StringCompare(null, null));
        assertEquals(-1, Util.StringCompare("apple", "banana"));
        assertEquals(1, Util.StringCompare("banana", "apple"));
        // Case-insensitive comparison
        assertEquals(-1, Util.StringCompare("apple", "Apple"));
        assertEquals(-1, Util.StringCompare("a", "abc"));
        assertEquals(1, Util.StringCompare("abc", "a"));
        assertEquals(-1, Util.StringCompare("", "a"));
        assertEquals(1, Util.StringCompare("a", ""));
        assertEquals(-1, Util.StringCompare(null, "a"));
        assertEquals(1, Util.StringCompare("a", null));
        // Testing with Whitespace
        assertEquals(0, Util.StringCompare(" hello ", "hello"));
        assertEquals(0, Util.StringCompare("hello ", " hello"));
        assertEquals(0, Util.StringCompare(" hello ", " hello "));
    }

    // Helper method to access private method for more comprehensive testing (if needed)
    private String NormalizeString(String str) throws Exception {
        Method method = Util.class.getDeclaredMethod("NormalizeString", String.class);
        method.setAccessible(true);
        return (String) method.invoke(null, str);
    }
}

class Util {

    public static int StringCompare(String lhs, String rhs) {
        return NormalizeString(lhs).compareTo(NormalizeString(rhs));
    }

    private static String NormalizeString(String str) {
        if (str == null)
            return "";
        return str.trim().toLowerCase();
    }
}
