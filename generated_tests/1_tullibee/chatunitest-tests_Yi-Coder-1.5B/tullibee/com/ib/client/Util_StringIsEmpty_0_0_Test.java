package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_StringIsEmpty_0_0_Test {

    @Test
    void testStringIsEmpty() {
        assertTrue(Util.StringIsEmpty(""));
        assertFalse(Util.StringIsEmpty("a"));
        assertFalse(Util.StringIsEmpty(null));
    }
}
