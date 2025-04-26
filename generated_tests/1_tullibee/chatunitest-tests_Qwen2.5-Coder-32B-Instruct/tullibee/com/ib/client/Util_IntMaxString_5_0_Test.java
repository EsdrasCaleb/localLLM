package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_IntMaxString_5_0_Test {

    @Test
    public void testIntMaxString() throws Exception {
        // Test case when value is Integer.MAX_VALUE
        assertEquals("", Util.IntMaxString(Integer.MAX_VALUE));
        // Test case when value is less than Integer.MAX_VALUE
        assertEquals("100", Util.IntMaxString(100));
        // Test case when value is greater than Integer.MAX_VALUE - 1 (to ensure boundary condition is handled correctly)
        assertEquals("2147483646", Util.IntMaxString(Integer.MAX_VALUE - 1));
        // Test case when value is zero
        assertEquals("0", Util.IntMaxString(0));
        // Test case when value is negative
        assertEquals("-100", Util.IntMaxString(-100));
    }
}
