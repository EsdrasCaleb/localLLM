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

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickSize_1_1_Test {

    // Test class
    @Test
    public void testTickSize() {
        int tickerId = 1;
        int field = TickType.BID;
        int size = 10;
        String expected = "id=1  BID=10";
        String actual = EWrapperMsgGenerator.tickSize(tickerId, field, size);
        assertEquals(expected, actual);
    }
}
