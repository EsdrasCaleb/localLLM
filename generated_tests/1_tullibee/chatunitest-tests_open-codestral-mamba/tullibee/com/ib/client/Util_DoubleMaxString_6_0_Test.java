package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_DoubleMaxString_6_0_Test {

    @Test
    void testDoubleMaxString() {
        // Test with a value equal to Double.MAX_VALUE
        assertEquals("", Util.DoubleMaxString(Double.MAX_VALUE));
        // Test with a value not equal to Double.MAX_VALUE
        assertEquals("123.45", Util.DoubleMaxString(123.45));
    }
}
