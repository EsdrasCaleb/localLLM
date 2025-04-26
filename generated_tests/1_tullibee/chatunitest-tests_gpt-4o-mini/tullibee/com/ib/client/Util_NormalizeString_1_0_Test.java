package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_NormalizeString_1_0_Test {

    @Test
    public void testNormalizeString_WithNonNullString() {
        String input = "Hello, World!";
        String result = Util.NormalizeString(input);
        Assertions.assertEquals(input, result);
    }

    @Test
    public void testNormalizeString_WithNullString() {
        String input = null;
        String result = Util.NormalizeString(input);
        Assertions.assertEquals("", result);
    }

    @Test
    public void testNormalizeString_WithEmptyString() {
        String input = "";
        String result = Util.NormalizeString(input);
        Assertions.assertEquals(input, result);
    }

    @Test
    public void testNormalizeString_WithWhitespaceString() {
        String input = "   ";
        String result = Util.NormalizeString(input);
        Assertions.assertEquals(input, result);
    }
}
