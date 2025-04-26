package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_StringCompare_2_0_Test {

    @ParameterizedTest
    @CsvSource({ "abc,abc,0", "abc,def,-1", "def,abc,1", "aBc,abc,1", "abc,  abc,0", "abc,abc ,0", "abc,,1", ",abc,-1", "null,null,0", "abc,null,1", "null,abc,-1", "abc,abc,0", "null,null,0", "abc,null,1", "null,abc,-1", "aBc,abc,1", "abc,def,-1", "def,abc,1", "abc,  abc,0", "abc,abc ,0" })
    void testStringCompare(String lhs, String rhs, int expected) {
        int actual;
        try {
            actual = Util.StringCompare(lhs, rhs);
        } catch (IllegalArgumentException e) {
            fail("Unexpected IllegalArgumentException: " + e.getMessage());
            // Crucial: Exit the method if an exception occurs.
            return;
        }
        assertEquals(expected, actual);
    }

    static class Util {

        static int StringCompare(String lhs, String rhs) {
            if (lhs == null || rhs == null) {
                if (lhs == null && rhs == null) {
                    return 0;
                } else if (lhs == null) {
                    return -1;
                } else {
                    return 1;
                }
            }
            return lhs.compareToIgnoreCase(rhs);
        }
    }
}
