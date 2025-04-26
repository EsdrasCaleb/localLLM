package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_DoubleMaxString_6_0_Test {

    @Test
    public void testDoubleMaxString() {
        // Test when the input value is Double.MAX_VALUE
        assertEquals("", Util.DoubleMaxString(Double.MAX_VALUE));
        // Test when the input value is not Double.MAX_VALUE
        assertEquals("1.0", Util.DoubleMaxString(1.0));
        assertEquals("0.0", Util.DoubleMaxString(0.0));
        assertEquals("-1.0", Util.DoubleMaxString(-1.0));
        assertEquals("1.7976931348623157E308", Util.DoubleMaxString(Double.MAX_VALUE - 1));
    }
}
