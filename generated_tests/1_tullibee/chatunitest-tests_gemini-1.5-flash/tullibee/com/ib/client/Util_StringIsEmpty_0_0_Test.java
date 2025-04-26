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
    void testStringIsEmpty_NullString() {
        assertTrue(Util.StringIsEmpty(null));
    }

    @Test
    void testStringIsEmpty_EmptyString() {
        assertTrue(Util.StringIsEmpty(""));
    }

    @Test
    void testStringIsEmpty_NonEmptyString() {
        assertFalse(Util.StringIsEmpty("hello"));
    }

    @Test
    void testStringIsEmpty_StringWithSpaces() {
        assertFalse(Util.StringIsEmpty("   "));
    }
}
