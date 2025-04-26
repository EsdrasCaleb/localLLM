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

public class EWrapperMsgGenerator_tickPrice_0_0_Test {

    @Test
    public void testTickPrice() {
        // Test with valid input
        String result = EWrapperMsgGenerator.tickPrice(1, 2, 3.0, 0);
        assertEquals("id=1  TickType.FIELDS[2]=3.0 noAutoExecute", result);
        // Test with canAutoExecute = 1
        String result2 = EWrapperMsgGenerator.tickPrice(2, 3, 4.0, 1);
        assertEquals("id=2  TickType.FIELDS[3]=4.0 canAutoExecute", result2);
        // Test with canAutoExecute = 0
        String result3 = EWrapperMsgGenerator.tickPrice(3, 4, 5.0, 0);
        assertEquals("id=3  TickType.FIELDS[4]=5.0 noAutoExecute", result3);
    }
}
