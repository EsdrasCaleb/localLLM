package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringIsEmpty_0_0_Test {

    @Test
    public void testStringIsEmpty_withNull() {
        assertTrue(Util.StringIsEmpty(null));
    }

    @Test
    public void testStringIsEmpty_withEmptyString() {
        assertTrue(Util.StringIsEmpty(""));
    }

    @Test
    public void testStringIsEmpty_withNonEmptyString() {
        assertFalse(Util.StringIsEmpty("test"));
    }
}
