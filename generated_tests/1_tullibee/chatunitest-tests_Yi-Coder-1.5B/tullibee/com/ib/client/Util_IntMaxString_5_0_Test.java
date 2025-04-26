package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_IntMaxString_5_0_Test {

    // Test class
    @Test
    void testIntMaxString() {
        assertEquals("123", Util.IntMaxString(123));
        assertEquals("123456789", Util.IntMaxString(123456789));
        assertEquals("", Util.IntMaxString(Integer.MAX_VALUE));
    }
}
