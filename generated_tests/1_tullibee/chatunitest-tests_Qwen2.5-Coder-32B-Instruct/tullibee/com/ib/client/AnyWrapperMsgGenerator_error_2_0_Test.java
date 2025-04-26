package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_2_0_Test {

    @Test
    public void testErrorMethod() {
        // Test case 1: Normal case with positive integers and non-empty string
        String result1 = AnyWrapperMsgGenerator.error(100, 404, "Not Found");
        assertEquals("100 | 404 | Not Found", result1);
        // Test case 2: Zero values for id and errorCode
        String result2 = AnyWrapperMsgGenerator.error(0, 0, "No Error");
        assertEquals("0 | 0 | No Error", result2);
        // Test case 3: Negative values for id and errorCode
        String result3 = AnyWrapperMsgGenerator.error(-1, -2, "Negative Values");
        assertEquals("-1 | -2 | Negative Values", result3);
        // Test case 4: Empty string for errorMsg
        String result4 = AnyWrapperMsgGenerator.error(1, 1, "");
        assertEquals("1 | 1 | ", result4);
        // Test case 5: Large values for id and errorCode
        String result5 = AnyWrapperMsgGenerator.error(Integer.MAX_VALUE, Integer.MIN_VALUE, "Large Values");
        assertEquals("2147483647 | -2147483648 | Large Values", result5);
    }
}
