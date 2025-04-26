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
        // Test with a positive value
        assertEquals("", Util.DoubleMaxString(Double.MAX_VALUE));
        // Test with a negative value
        assertNotEquals("", Util.DoubleMaxString(-Double.MAX_VALUE));
        // Test with zero
        assertEquals("0.0", Util.DoubleMaxString(0.0));
    }
}
