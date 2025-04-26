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
    void testStringIsEmpty_nullString() {
        boolean result = Util.StringIsEmpty(null);
        assertTrue(result);
    }

    @Test
    void testStringIsEmpty_emptyString() {
        boolean result = Util.StringIsEmpty("");
        assertTrue(result);
    }

    @Test
    void testStringIsEmpty_stringWithContent() {
        boolean result = Util.StringIsEmpty("hello");
        assertFalse(result);
    }

    @Test
    void testStringIsEmpty_stringWithWhitespace() {
        boolean result = Util.StringIsEmpty(" ");
        assertFalse(result);
    }
}
