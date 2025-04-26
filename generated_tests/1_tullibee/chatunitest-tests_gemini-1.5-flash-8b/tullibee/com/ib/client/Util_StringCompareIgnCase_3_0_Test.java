package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_StringCompareIgnCase_3_0_Test {

    @ParameterizedTest
    @CsvSource({ "abc,abc,0", "abc,def,-1", "def,abc,1", "ABC,abc,0", "aBc,abc,0", "  abc,abc,0", "abc  ,abc,0", "abc, ,1", ",abc,-1", "null,abc,-1", "abc,null,1" })
    void testStringCompareIgnCase(String lhs, String rhs, int expected) {
        int actual = Util.StringCompareIgnCase(lhs, rhs);
        assertEquals(expected, actual);
    }

    // Helper method to normalize a string (replace with actual implementation if available)
    private static String NormalizeString(String str) {
        return (str == null) ? "" : str.trim().toLowerCase();
    }
}
