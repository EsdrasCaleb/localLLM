package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_IntMaxString_5_0_Test {

    @Test
    public void testIntMaxString() {
        // Test when the input value is Integer.MAX_VALUE
        assertEquals("", Util.IntMaxString(Integer.MAX_VALUE));
        // Test when the input value is less than Integer.MAX_VALUE
        assertEquals("100", Util.IntMaxString(100));
        // Test when the input value is greater than Integer.MAX_VALUE
        assertEquals("-100", Util.IntMaxString(-100));
        // Test when the input value is Integer.MIN_VALUE
        assertEquals("" + Integer.MIN_VALUE, Util.IntMaxString(Integer.MIN_VALUE));
    }
}
