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
    public void testEmptyString_ReturnsTrue() {
        String input = "";
        boolean result = Util.StringIsEmpty(input);
        assertTrue(result);
    }

    @Test
    public void nullString_ReturnsTrue() {
        String input = null;
        boolean result = Util.StringIsEmpty(input);
        assertTrue(result);
    }

    @Test
    public void nonEmptyString_ReturnsFalse() {
        String input = "Hello";
        boolean result = Util.StringIsEmpty(input);
        assertFalse(result);
    }

    @Test
    public void nullInput_ThrowsNullPointerException() {
        assertThrows(NullPointerException.class, () -> Util.StringIsEmpty(null));
    }
}
