package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_IntMaxString_5_0_Test {

    @Test
    void testIntMaxString() {
        // Test with Integer.MAX_VALUE
        String expected = "";
        String result = Util.IntMaxString(Integer.MAX_VALUE);
        assertEquals(expected, result);
        // Test with a normal integer
        int normalInteger = 123;
        expected = "" + normalInteger;
        result = Util.IntMaxString(normalInteger);
        assertEquals(expected, result);
        // Test with 0
        int zero = 0;
        expected = "" + zero;
        result = Util.IntMaxString(zero);
        assertEquals(expected, result);
        // Test with negative integer
        int negativeInteger = -123;
        expected = "" + negativeInteger;
        result = Util.IntMaxString(negativeInteger);
        assertEquals(expected, result);
    }
}
