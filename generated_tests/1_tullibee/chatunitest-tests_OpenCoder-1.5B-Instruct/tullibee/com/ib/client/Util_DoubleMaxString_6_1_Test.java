// Test method
package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Util_DoubleMaxString_6_1_Test {

    @Mock
    private Util util;

    @Test
    public void testDoubleMaxString() {
        double input = Double.MAX_VALUE;
        String expectedOutput = "";
        String actualOutput = util.DoubleMaxString(input);
        Assertions.assertEquals(expectedOutput, actualOutput);
    }
}
