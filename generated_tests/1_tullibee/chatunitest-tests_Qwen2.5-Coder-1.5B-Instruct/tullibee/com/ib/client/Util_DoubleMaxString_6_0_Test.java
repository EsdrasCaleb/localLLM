package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class // You can add more test cases here if needed
Util_DoubleMaxString_6_0_Test {

    @Test
    public void testDoubleMaxString() throws Exception {
        // Create mock objects for any dependencies
        // If there are no dependencies, you can leave them out
        // Create a new instance of the class to test
        Util util = new Util();
        // Define the input and expected output
        double input = Double.MAX_VALUE;
        String expectedOutput = "";
        // Call the method under test
        String result = util.DoubleMaxString(input);
        // Assert the result
        assertEquals(expectedOutput, result, "The method did not return the expected output.");
    }
}
