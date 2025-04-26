package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_tickPrice_0_2_Test {

    @Test
    public void testTickPrice() {
        // Test case 1: canAutoExecute is non-zero
        String result1 = EWrapperMsgGenerator.tickPrice(1, 1, 100.50, 1);
        assertEquals("id=1  1=100.5 canAutoExecute", result1);
        // Test case 2: canAutoExecute is zero
        String result2 = EWrapperMsgGenerator.tickPrice(2, 2, 200.75, 0);
        assertEquals("id=2  2=200.75 noAutoExecute", result2);
        // Test case 3: Edge case with minimum values
        String result3 = EWrapperMsgGenerator.tickPrice(Integer.MIN_VALUE, Integer.MIN_VALUE, Double.MIN_VALUE, Integer.MIN_VALUE);
        assertEquals("id=-2147483648  -2147483648=1.4E-45 canAutoExecute", result3);
        // Test case 4: Edge case with maximum values
        String result4 = EWrapperMsgGenerator.tickPrice(Integer.MAX_VALUE, Integer.MAX_VALUE, Double.MAX_VALUE, Integer.MAX_VALUE);
        assertEquals("id=2147483647  2147483647=1.7976931348623157E308 canAutoExecute", result4);
    }
}
