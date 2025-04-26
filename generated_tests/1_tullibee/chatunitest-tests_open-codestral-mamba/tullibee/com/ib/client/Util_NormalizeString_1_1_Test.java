package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_NormalizeString_1_1_Test {

    @Test
    public void testNormalizeString() {
        String input = "Hello";
        String expectedOutput = "Hello";
        String actualOutput = Util.NormalizeString(input);
        assertEquals(expectedOutput, actualOutput);
        input = null;
        expectedOutput = "";
        actualOutput = Util.NormalizeString(input);
        assertEquals(expectedOutput, actualOutput);
    }
}
