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
    public void testStringIsEmpty() {
        String testString = "test";
        boolean result = Util.StringIsEmpty(testString);
        Assertions.assertFalse(result);
    }

    @Test
    public void testStringIsNotEmpty() {
        String testString = "test";
        boolean result = Util.StringIsEmpty(testString);
        Assertions.assertFalse(result);
    }

    @Test
    public void testNullString() {
        String testString = null;
        boolean result = Util.StringIsEmpty(testString);
        Assertions.assertTrue(result);
    }

    @Test
    public void testEmptyString() {
        String testString = "";
        boolean result = Util.StringIsEmpty(testString);
        Assertions.assertTrue(result);
    }
}
