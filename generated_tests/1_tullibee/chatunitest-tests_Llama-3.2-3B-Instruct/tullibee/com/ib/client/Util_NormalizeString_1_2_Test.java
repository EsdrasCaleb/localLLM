package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_NormalizeString_1_2_Test {

    @Test
    public void testNormalizeString_NullInput_ReturnsEmptyString() {
        String input = null;
        String expected = "";
        String actual = Util.NormalizeString(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testNormalizeString_NotNullInput_ReturnsOriginalString() {
        String input = "Hello";
        String expected = "Hello";
        String actual = Util.NormalizeString(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testNormalizeString_EmptyInput_ReturnsEmptyString() {
        String input = "";
        String expected = "";
        String actual = Util.NormalizeString(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testNormalizeString_WhitespaceInput_ReturnsEmptyString() {
        String input = "   ";
        String expected = "";
        String actual = Util.NormalizeString(input);
        assertEquals(expected, actual);
    }
}
