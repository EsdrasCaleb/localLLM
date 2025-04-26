package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_StringCompareIgnCase_3_0_Test {

    @Test
    void testStringCompareIgnCase() {
        String lhs = "Test";
        String rhs = "test";
        int result = Util.StringCompareIgnCase(lhs, rhs);
        assertEquals(0, result);
        lhs = "Test";
        rhs = "Test1";
        result = Util.StringCompareIgnCase(lhs, rhs);
        assertTrue(result > 0);
        lhs = "Test";
        rhs = "Test1";
        result = Util.StringCompareIgnCase(lhs, rhs);
        assertTrue(result < 0);
        lhs = null;
        rhs = "Test";
        result = Util.StringCompareIgnCase(lhs, rhs);
        assertEquals(0, result);
        lhs = "Test";
        rhs = null;
        result = Util.StringCompareIgnCase(lhs, rhs);
        assertEquals(0, result);
        lhs = null;
        rhs = null;
        result = Util.StringCompareIgnCase(lhs, rhs);
        assertEquals(0, result);
    }
}
