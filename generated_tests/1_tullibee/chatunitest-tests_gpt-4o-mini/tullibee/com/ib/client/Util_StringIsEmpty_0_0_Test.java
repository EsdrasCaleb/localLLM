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
    public void testStringIsEmpty_NullString() {
        assertTrue(Util.StringIsEmpty(null), "Expected true for null string");
    }

    @Test
    public void testStringIsEmpty_EmptyString() {
        assertTrue(Util.StringIsEmpty(""), "Expected true for empty string");
    }

    @Test
    public void testStringIsEmpty_NonEmptyString() {
        assertFalse(Util.StringIsEmpty("Hello"), "Expected false for non-empty string");
    }

    @Test
    public void testStringIsEmpty_WhitespaceString() {
        assertFalse(Util.StringIsEmpty(" "), "Expected false for whitespace string");
    }
}
