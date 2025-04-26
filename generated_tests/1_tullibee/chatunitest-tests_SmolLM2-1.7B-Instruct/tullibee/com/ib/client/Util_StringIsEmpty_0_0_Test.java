package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_StringIsEmpty_0_0_Test {

    // Test method
    @Test
    public void testStringIsEmpty() {
        String str = "";
        assertTrue(Util.StringIsEmpty(str));
    }
}
