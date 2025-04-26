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
    void testStringIsEmpty() {
        String str1 = "";
        assertTrue(Util.StringIsEmpty(str1));
        String str2 = "  ";
        assertTrue(Util.StringIsEmpty(str2));
        String str3 = "Hello";
        assertFalse(Util.StringIsEmpty(str3));
    }
}
