package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_IntMaxString_5_2_Test {

    @Test
    void testIntMaxString() {
        assertEquals("", Util.IntMaxString(Integer.MAX_VALUE));
        assertEquals("", Util.IntMaxString(Integer.MIN_VALUE));
        assertEquals("1234567890", Util.IntMaxString(1234567890));
        assertEquals("1234567890", Util.IntMaxString(-1234567890));
    }
}
