package com.ib.client;

import java.lang.reflect.Method;
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
        Util util = new Util();
        // null should be empty
        assertTrue(util.StringIsEmpty(null));
        // empty string should be empty
        assertFalse(util.StringIsEmpty(""));
        // empty string should be empty
        assertTrue(util.StringIsEmpty(""));
        // non-empty string should be empty
        assertFalse(util.StringIsEmpty("a"));
        // null should be empty
        assertFalse(util.StringIsEmpty(null));
    }
}
